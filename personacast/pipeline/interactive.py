"""
the interactive session orchestrator

retrieve + curate ipelie runs once, we synthesize from the resulting pool
"""


from __future__ import annotations

import re
from pydantic import BaseModel

from .. import config
from ..llm.client import LLMClient
from ..models import (
    CuratedItem,
    InteractiveTurn,
    Persona,
    Reaction,
    ReactionType,
    SessionState,
    PersonaMemory,
    CoveredSource
)
from . import curation, dedup, memory, qa, queries, script, state, topics
from .retrieval import retrieve

def build_source_pool(persona: Persona, llm: LLMClient, on_stage = None,
                       *, memory: PersonaMemory | None = None, on_topic_done = None) -> dict[str, list[CuratedItem]]: 
    """
    plan_topics -> for each topic (generate query -> retrieve -> curate sources) -> deduplicate and return curated pool 
    keyed by topic
    memory --> read only , used for agentic method to not reshow sources listener has already seen/discussed 
    on_topic_done --> can see what agent decided
    """

    if config.AGENTIC_RETRIEVAL: 
        from ..agents.graph import build_source_pool_agentic
        return build_source_pool_agentic( # build source pool using retrieval agent node pipeline
                    persona, llm, on_stage=on_stage, memory=memory, on_topic_done=on_topic_done
                )

    def stage(label): # for cli usage and streamlit so u pass the onstage as a callback function defined as inline labmda, so it preints updates in particular way 
        if on_stage: 
            on_stage(label)
    
    stage("Planning Topics") # set label 
    plans = topics.plan_topics(persona) # create the plan for each topic 

    curated : dict[str, list[CuratedItem]] = {} ## set up dictionary to hold curated items for each topic key

    for plan in plans: # generate queries, retrieve items, curate items
        stage("Retrieving and Curating")
        query = queries.generate_queries(plan, llm)
        retrieve_items = retrieve.retrieve_topic(plan, query)
        curated[plan.topic] = curation.curate_topic(plan, retrieve_items, persona, llm)
    
    stage('Removing Duplicates')
    curated, _notes = dedup.dedup_across_topics(curated)

    return curated


### turn summarization to create gist for current turn to be fed into future turns for context

## gist generation system prompt 

_GIST_SYSTEM = (
    "Condense the following podcast turn into ONE short plain sentence capturing only the key "
    "point(s) it covered — it will be used as a 'what we already said' note so later turns don't "
    "repeat it. No preamble, no quotes, just the sentence."
)

def  summarize_turn(text: str, llm: LLMClient): 
    """
    condense turn content into 1 sentence gist we feed to future 
    if config.SUMMARIZE_TURNS flag is on -> no llm call just return first sentence truncated for testing
    """
    if not config.SUMMARIZE_TURNS: 
        sentences = re.split(r"(?<=[.!?])\s+", text.strip()) # split sentences 
        return sentences[0] if sentences and sentences[0] else text[:160]
    
    return llm.complete(_GIST_SYSTEM, text, temperature=0.2).strip()

_SUMMARY_MAX_LINES = 12 # how many lines of topic gist the cross session summary can keep 

def _update_summary(summary:str, turn: InteractiveTurn): 
    """
    append turn gist to rolling summary, capped at _SUMMARY_MAX_LINES
    """
    lines = [line for line in summary.splitlines() if line.strip()]
    lines.append(f"- [{turn.topic}] {turn.gist}")
    return "\n".join(lines[-_SUMMARY_MAX_LINES:])

### mapping pause time to sentence being spoken

def locate_snippet(text, seconds:float, total_seconds:float) -> str: 
    """
    given the audio pause time, find the sentence podcast was at
    seconds/total -> word index -> sentence containing word returns "" if nothing
    """

    text = (text or "").strip() # given full turn text  remove surrounding whitespace 
    if not text or total_seconds <= 0: #if no text return nothing
            return ""

    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()] # split sentences into list elements

    if not sentences: # another guard if splitting created nothing
        return ""

    fraction = min(max(seconds / total_seconds, 0.0), 1.0) # how far through audio the pause has happened as fraction if halfway through text 0.5

    target_word_position = fraction * len(text.split()) #estimated word number position 

    seen_word_count = 0 # accumulating word count
    ## logic: target word is at 5 word ocount psition, sentence A -> 4 words seen = 4, 5< 4 dalse move sentence B -> 3 words seen = 7 5 <7  true in sentence B!

    for sentence in sentences: 
        seen_word_count += len(sentence.split())
        if target_word_position < seen_word_count: 
            return sentence.strip()

    return sentences[-1].strip() # return sentence target word is in so we know sentence interrupted at 


### given the anchoring sentence make llm call to find the curated source that is associated with what that sentence is talking abt
_LINK_SYSTEM = (
    "You are given ONE sentence from a podcast turn and a numbered list of the source summaries "
    "that turn was synthesized from. Return the index of the single source that sentence is most "
    "directly based on. If no source clearly supports it, return -1."
)

class _SourceLink(BaseModel):
    index: int

def link_source(snippet: str, sources: list[CuratedItem], llm: LLMClient) -> CuratedItem | None: 
    """
    recover curated source from pinned sentence, returns None if nothing is surfaced
    the llm returns as an index so we can directly index the sources!
    """
    if not snippet.strip() or not sources: 
        return None 

    listing = "\n".join(f"[{i}] {s.title} — {s.summary}" for i, s in enumerate(sources)) # using enumerate give index to source [0] Attention Is All You Need — Introduces the transformer architecture...
    user = f"Sentence:\n{snippet}\n\nSources:\n{listing}" # user prompt

    try:
        link = llm.structured(_LINK_SYSTEM, user, _SourceLink, temperature=0.0)
    except Exception:
            return None
    return sources[link.index] if 0 <= link.index < len(sources) else None


### recording what sources listener has been shown, for cross session memory
def  _record_covered(session_state: SessionState, topic: str) -> None:
    """
    remember sources of this turn, turn in particular rather than source pool because there might be some items in source pool never seen by user
    this is to make sure a future session can filter these seen sources out
    """
    shown_sources_memory = session_state.memory.covered.setdefault(topic, []) # get shown sources 
    seen = {source.url for source in shown_sources_memory} # recover url for each shhown source
    
    for item in session_state.pool.get(topic, []):
        if item.url not in seen:
            seen.add(item.url) # add in seen if not already 
            shown_sources_memory.append(CoveredSource(url= item.url, title= item.title)) # append in covered source model 


### session step object 

class InteractiveSession: # interactive session object
    
    def __init__(self, persona: Persona, llm: LLMClient |None = None,*, on_stage = None): 
        self.persona = persona 
        self.llm = llm or LLMClient()
        self.on_stage = on_stage 
        self.state: SessionState | None = None
        ## topics that steer current session order is the tie breaker 
        self._active_topics = [interest.topic for interest in persona.interests]

        ## what retrieval agent decided per topic and filled in at start() 
        self.retrieval_trace: dict[str, dict] = {}

    def _note_topic(self, topic: str, final_state: dict) -> None:
        """
        compact record of one topic's graph run, for the UI
        """
        self.retrieval_trace[topic] = {
            "sources": final_state.get("sources", []),
            "queries": final_state.get("search_queries", []),
            "arxiv_queries": final_state.get("arxiv_queries", []),
            "retrieved": len(final_state.get("raw_items", [])),
            "kept": len(final_state.get("curated", [])),
            "notes": final_state.get("notes", []),
        }
    

    
    def start(self) -> SessionState: 
        """
        load/ merge memory, build the source pool 
        """
        mem = memory.load_memory(self.persona)
        pool = build_source_pool(self.persona, self.llm, on_stage= self.on_stage, memory = mem, on_topic_done= self._note_topic)

        self.state = SessionState(
            run_id = state.new_run_id(), persona = self.persona, memory = mem, pool = pool
        )

        ## snapshot of choices of retrieval agent
        if self.retrieval_trace:
            state.log_retrieval(self.state.run_id, self.retrieval_trace)

        return self.state 
    
    @property # call methods like attributes!
    def done(self) -> bool: 
        """
        tell us when we are done, basically when we have generated MAX_ITERATIONS turns
        """
        return self.state is not None and len(self.state.turns) >= config.MAX_ITERATIONS
    
    def _choose_topic(self, session_state : SessionState, last_reaction : Reaction | None): 
        """
        decide next turns topic, we hold topic till 
        explicit switch request in last reaction, or disengagment and points drop in current topic below otehr topic move to next highest topic
        first turn is the highest engagment point topic 
        othereise if user had good reaction, stay on current topic
        """

        if last_reaction and last_reaction.requested_topic in self._active_topics: 
            return last_reaction.requested_topic # if requested_topic exists in active topics return that requested topic to be next topic 
        
        if session_state.current_topic is None: 
            return memory.next_topic(session_state.memory, self._active_topics)
        
        if last_reaction and last_reaction.type == ReactionType.none: 
            # round robin next listed topic when everything is at 0 
            # find next highest engagment topic if exist 
            best = memory.next_topic(session_state.memory, self._active_topics)

            if session_state.memory.engagement.get(best, config.ENGAGE_BASE)  > session_state.memory.engagement.get(session_state.current_topic, config.ENGAGE_BASE):
                # if there is a more engaging topic than current topic
                return best 
            
            #none past conditions are met then we are in a tie in engagement so we round robin
            i = self._active_topics.index(session_state.current_topic) # get position of current topic in list 
            return self._active_topics[(i + 1) % len(self._active_topics)]
        
        return session_state.current_topic
    
    def _require_started(self) -> SessionState:
        """
        check for start, if not raise error to call start before generating turn
        """
        if self.state is None:
            raise RuntimeError("call start() before generating turns")
        return self.state
    
    def next_segment(self): 
        """
        choose the topic and generate the 60s turn
        """
        session_state = self._require_started()
        last_reaction = session_state.turns[-1].reaction if session_state.turns else None # if exists pull most recent turn reaction or return None 
        topic = self._choose_topic(session_state, last_reaction) # choose topic and assign to topic 
        session_state.current_topic = topic # set sesion state current topic 
        recent_gists = [turn.gist for turn in session_state.turns[-config.RECENT_TURNS_CONTEXT:] if turn.gist] # iterate for 4 turn defuault, and return the most 4 recent turn gists config sets default 4

        ## if listener interrupted at a certain sentence, link back to source so turn expands on that source specifically
        focus_source = None 
        if last_reaction and last_reaction.anchor_snippet:
            focus_source = link_source(last_reaction.anchor_snippet, session_state.pool.get(last_reaction.topic, []), self.llm,)
            last_reaction.anchor_source = focus_source.title if focus_source else ""

        text = script.generate_turn(
            topic, session_state.pool.get(topic, []), self.persona, session_state.memory, recent_gists, self.llm,
            last_reaction=last_reaction, focus_source= focus_source,
        ) # generate turn text
        turn = InteractiveTurn(iteration=len(session_state.turns) + 1, topic=topic, text=text)
        session_state.turns.append(turn)
        _record_covered(session_state, topic) # note sources feeding this turn so a future session filters them out
        return turn

    def _read_reaction(self, reaction_text, turn, session_state, anchor_snippet):
        """
        semantic read of reaction 
        """
        if not config.AGENTIC_INTERACTION:
            return None, None
        
        from ..agents import interaction 
        try:
            read = interaction.read_reaction( # have llm read reaction and interpret 
                reaction_text, turn, session_state, self.persona, self.llm,
                active_topics=self._active_topics, anchor_snippet=anchor_snippet.strip(),
            )
        except Exception:
            return None, None

        #  switch only if requested topic is in topic list/ not this topic
        requested = (read.requested_topic or "").strip() or None
        if requested == turn.topic or requested not in self._active_topics:
            requested = None
        is_switch = requested is not None

        # returns reaction read 
        return read, {
            "type": interaction.to_reaction_type(read.intent),
            "delta": interaction.clamp_delta(read.engagement_delta, is_switch=is_switch),
            "requested": requested,
            "needs_answer": read.needs_answer,
        }

    
    def submit_reaction(self, reaction_text: str, *, anchor_snippet: str = "") : 
        """
        attach the listern reaction to the current turn object, type of reaction inferred from text
        question is answered through qa function , web_fallback if not in sources, memory updated and turn summarized
        
        """ 

        session_state = self._require_started() # load session state
        
        if not session_state.turns: # raise error in order
            raise RuntimeError("submit_reaction called before next_segment")

        turn = session_state.turns[-1] # pull recent turn

        # read the reaction either with llm agentic method or original method
        read, norm = self._read_reaction(reaction_text, turn, session_state, anchor_snippet)

        if read is not None and norm is not None: # fill in details 
            reaction_type = norm["type"]
            delta = norm["delta"]
            requested = norm["requested"]
            needs_answer = norm["needs_answer"]
        else:
            reaction_type = memory.classify_reaction(reaction_text)
            requested = memory.detect_requested_topic(reaction_text, self._active_topics, exclude=turn.topic)
            delta = None 
            needs_answer = reaction_type == ReactionType.question

        ## only honor the anchoring if there is a reaction if no reaction its treated as just a pause!
        anchor = "" if reaction_type == ReactionType.none else anchor_snippet.strip()
        reaction = Reaction( iteration=turn.iteration, topic=turn.topic, type=reaction_type, text=reaction_text.strip(), requested_topic=requested,answer=None, anchor_snippet= anchor)

        if read is not None:
            # persist what the agent read memory.reactions accumulates labelled behaviour over time
            reaction.intent = read.intent
            reaction.sentiment = read.sentiment
            reaction.engagement_delta = delta

        if needs_answer:
            answer = qa.answer_question( # answer question using qa utility
                reaction.text, self.persona, qa.flatten_curated(session_state.pool), self.llm, allow_web=True,
            )
            reaction.answer = answer.answer

        turn.reaction = reaction

        # add to memory
        memory.apply_reaction(session_state.memory, reaction, delta=delta)

        # gist for the cross session summary. the agent already produced one as part of the same call above
        turn.gist = read.gist.strip() if read is not None and read.gist.strip() else summarize_turn(turn.text, self.llm)
        session_state.memory.summary = _update_summary(session_state.memory.summary, turn)

        # persist memory to disk and snapshot the growing session
        memory.save_memory(session_state.memory)
        state.log_turn(session_state, turn.iteration)
        return turn
    
    def finish(self) -> SessionState:
        """
        final memory save and write fill session transcript to run directory
        """
        session_state = self._require_started()
        memory.save_memory(session_state.memory)
        transcript = "\n\n".join(f"[{t.topic}] {t.text}" for t in session_state.turns)
        (state.run_dir(session_state.run_id) / "session.txt").write_text(transcript)
        
        return session_state



        



    



