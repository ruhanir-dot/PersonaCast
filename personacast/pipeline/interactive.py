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
)
from . import curation, dedup, memory, qa, queries, script, state, topics
from .retrieval import retrieve

def build_source_pool(persona: Persona, llm: LLMClient, on_stage = None) -> dict[str, list[CuratedItem]]: 
    """
    plan_topics -> for each topic (generate query -> retrieve -> curate sources) -> deduplicate and return curated pool 
    keyed by topic
    """

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


### session step object 

class InteractiveSession: # interactive session object
    
    def __init__(self, persona: Persona, llm: LLMClient |None = None,*, on_stage = None): 
        self.persona = persona 
        self.llm = llm or LLMClient()
        self.on_stage = on_stage 
        self.state: SessionState | None = None
        ## topics that steer current session order is the tie breaker 
        self._active_topics = [interest.topic for interest in persona.interests]
    
    def start(self) -> SessionState: 
        """
        load/ merge memory, build the source pool 
        """
        mem = memory.load_memory(self.persona)
        pool = build_source_pool(self.persona, self.llm, on_stage= self.on_stage)

        self.state = SessionState(
            run_id = state.new_run_id(), persona = self.persona, memory = mem, pool = pool
        )

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
        return turn
    
    def submit_reaction(self, reaction_text: str, *, anchor_snippet: str = "") : 
        """
        attach the listern reaction to the current turn object, type of reaction inferred from text
        question is answered through qa function , web_fallback if not in sources, memory updated and turn summarized
        
        """ 

        session_state = self._require_started() # load session state
        
        if not session_state.turns: # raise error in order
            raise RuntimeError("submit_reaction called before next_segment")

        turn = session_state.turns[-1] # pull recent turn

        reaction_type = memory.classify_reaction(reaction_text)
        requested = memory.detect_requested_topic(reaction_text, self._active_topics, exclude=turn.topic)

        ## only honor the anchoring if there is a reaction if no reaction its treated as just a pause!
        anchor = "" if reaction_type == ReactionType.none else anchor_snippet.strip()
        reaction = Reaction( iteration=turn.iteration, topic=turn.topic, type=reaction_type, text=reaction_text.strip(), requested_topic=requested,answer=None, anchor_snippet= anchor)


        if reaction_type == ReactionType.question:
            answer = qa.answer_question( # answer question using qa utility
                reaction.text, self.persona, qa.flatten_curated(session_state.pool), self.llm, allow_web=True,
            )
            reaction.answer = answer.answer

        turn.reaction = reaction

        # add to memory
        memory.apply_reaction(session_state.memory, reaction)

        # create gist and add to cross session summary gists
        turn.gist = summarize_turn(turn.text, self.llm)
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



        



    



