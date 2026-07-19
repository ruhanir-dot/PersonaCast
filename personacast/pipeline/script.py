"""
Final per topic script generation as well as a stitch pass
 we do two different LLM jobs

1. write segments where we do one call per topic
v2 tries to make it more synthesized than v1 and more flowing

2. stitch the seperate segments for each topic

"""

from __future__ import annotations

import re

from pydantic import BaseModel

from .. import config
from ..llm.client import LLMClient
from ..models import CuratedItem, Persona, PersonaMemory, Reaction, ReactionType, TopicSegment

_SEGMENT_SYSTEM = (
    "You write ONE spoken-podcast segment that SYNTHESIZES the given sources into a single "
    "narrative with a clear throughline — NOT a list of 'this source said X, that source said Y'. "
    "First identify the common thread across the sources, then write one arc around it.\n\n"
    "DEPTH — calibrate to the listener's expertise, but ONLY using what the sources actually say. "
    "Depth means precision and correct terminology, NOT inventing mechanisms, numbers, or causes "
    "the sources don't state:\n"
    "- advanced: use precise domain terminology, assume fundamentals are known, and draw out "
    "implications the sources support — but do not manufacture explanations or figures to sound "
    "deep. If the sources don't explain 'why', don't invent a 'why'.\n"
    "- intermediate: convey what's new and why it matters, lightly, staying within the sources.\n"
    "- beginner: prioritize intuition and significance over technical detail.\n\n"
    "Respect the AVOID list as STYLE guidance (e.g. 'basic ML 101 explanations' => skip "
    "fundamentals).\n\n"
    "FACTUAL FIDELITY — this is critical. Every number, name, and attribution in the sources comes "
    "with a CONTEXT (e.g. 'regular-season average' vs 'Finals average'; which player; which team; "
    "which year). Preserve that context EXACTLY. Never transfer a statistic, quote, or attribution "
    "from one context to another — e.g. do not report a regular-season stat line as a playoff or "
    "Finals one, and do not reassign an action to a different player than the source states. If two "
    "sources give different numbers for different contexts, keep them distinct; do not merge or "
    "average them. Do not invent specifics (biography, exact figures, physics) that the sources do "
    "not state — synthesize only what is actually in the summaries.\n"
    "NEVER INVENT A NAME OR SPECIFIC TO FILL A GAP. If a summary refers to someone or something "
    "WITHOUT naming it (e.g. 'a blind Ukrainian veteran'), keep that generic phrasing — do NOT "
    "make up a name, number, date, or place to sound concrete. A generic-but-true reference is "
    "required; a specific-but-invented one is a failure.\n\n"
    "DO NOT write an introduction, a recap of what you're about to say, or a concluding/summary "
    "sentence — those belong to the episode's stitch pass, not the segment. Start directly in the "
    "substance. Output prose only: no headings, no bullet points.\n\n"
    "LENGTH: aim to USE THE FULL budget of about {budget} words — develop the real facts fully "
    "(context, implications the sources support, why they matter) so the segment lands near "
    "{budget} words, not well short. Hard ceiling: never exceed {budget}. But do NOT pad with "
    "filler or invent detail to reach the target — if you genuinely run out of sourced material, a "
    "slightly shorter segment is fine; fabricating to hit a word count is never acceptable."
)

_STITCH_SYSTEM = (
    "You write ONLY the connective material for a podcast episode. You are given the persona "
    "and the full text of each segment in order. Produce exactly: a brief intro (≤40 words) "
    "that previews the episode topics; one short transition (≤25 words) for each gap between "
    "consecutive segments (N segments → N-1 transitions, in order); and a brief outro (≤30 "
    "words). Match the tone from the persona's tone field exactly.\n\n"
    "DO NOT reproduce, summarize, paraphrase, or rewrite any segment body — only write the "
    "intro, the between-segment transitions, and the outro.\n\n"
    "DO NOT invent thematic connections between topics that the segment content does not "
    "support. If no genuine link exists, write a simple honest pivot ('now let's turn to X') "
    "rather than fabricating one."
)

class _Stitch(BaseModel): 
    intro: str
    transitions: list[str]
    outro:str

# how far bidget a segment can run over before we trim
_BUDGET_TOLERANCE = 0.10

def _trim_to_budget(text, word_budget) -> str:
    """
    make sure the segment fits allotted word budget
    trim when text exceeds budget*(1+tolerance), cut at last sentence so no mid sentence cuts
    """
    words = text.split() # split into list of words and count

    if len(words) <= word_budget * (1 + _BUDGET_TOLERANCE): # text within budget change nothing
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) # split text into sentences at punctuation 
    
    kept = [] #kept sentences accumulator
    count = 0

    for sentence in sentences:
        n = len(sentence.split()) # split
        if count + n > word_budget and kept:
            break # if adding sentence exceed budget break
        kept.append(sentence)
        count += n
    return " ".join(kept) # join kept sentences in a string text 

def write_segment( topic: str, expertise: str, word_budget: int, curated: list[CuratedItem], persona: Persona, llm: LLMClient,) -> TopicSegment: 
    ### build formatted string of all curated sources for this topic
    sources_block = "\n\n".join(
        f"[{c.source}] {c.title} ({c.url})\n{c.summary}" for c in curated
    ) or "(no sources survived curation for this topic)"
    
    ### assemble user prompt passed to LLM
    user = (
        f"Topic: {topic}\n"
        f"Listener expertise: {expertise}\n"
        f"Listener tone: {persona.tone}\n"
        f"AVOID (style): {persona.avoid}\n"
        f"Hard word limit: {word_budget} words\n\n"
        f"Sources to synthesize:\n{sources_block}"
    )
    
    # build at cooler temp 
    text = llm.complete(_SEGMENT_SYSTEM.format(budget=word_budget), user, temperature=0.3)

    # return topic segment object 
    return TopicSegment(topic=topic, text=_trim_to_budget(text, word_budget))

def stitch(segments: list[TopicSegment], persona: Persona, llm: LLMClient) -> str:
    """
    generate intro/transitions/outro, then assemble the final script
    """
    topics = [segment.topic for segment in segments] # get topic list
    
    user = ( # user prompt
        f"Listener tone: {persona.tone}\n"
        f"Topics in order ({len(topics)}): {topics}\n"
        f"So produce {max(len(topics) - 1, 0)} transition(s)."
    )
    
    parts = llm.structured(_STITCH_SYSTEM, user, _Stitch) # stitch parts

    pieces = [parts.intro.strip()] # start with intro
    
    for i, segment in enumerate(segments): # loop through each topic segment 
        pieces.append(segment.text.strip())
        if i < len(segments) - 1: # if not last segment add a transition after
            if i < len(parts.transitions): # guard in case less transitions than expected
                pieces.append(parts.transitions[i].strip()) # add transition between this segment and next

    pieces.append(parts.outro.strip()) # end with outro
    return "\n\n".join(p for p in pieces if p) # join everything


### Interactive turn generation

## Turn generation system prompt
_TURN_SYSTEM = (
    "You are voicing ONE short spoken-podcast turn — about {budget} words, roughly 60 seconds — in an "
    "ONGOING, interactive session with a single listener. This is NOT a full episode; it is one live "
    "beat that continues from what came before and will be followed by more.\n\n"
    "SYNTHESIS & DEPTH — synthesize the given sources into a single clear throughline calibrated to the "
    "listener's expertise (advanced: precise domain terms, assume fundamentals; intermediate: what's "
    "new and why it matters; beginner: intuition over technical detail). Respect the AVOID list as "
    "STYLE guidance.\n\n"
    "FACTUAL FIDELITY — use ONLY what the sources actually state. Never invent a name, number, date, "
    "statistic, or mechanism to fill a gap, and never transfer a fact from one context to another. A "
    "generic-but-true reference beats a specific-but-invented one.\n\n"
    "SITUATION — the listener's current situation is: {context}. Match your energy and pacing to it.\n\n"
    "CONTINUITY — you are given short gists of the most recent turns and a running summary of the "
    "session. Do NOT repeat what was already covered; move the conversation forward.\n\n"
    "{reaction_instr}\n\n"
    "{mode_instr}\n\n"
    "Start directly in the substance — no 'welcome back', no meta narration, no sign-off. Output spoken "
    "prose only: no headings, no bullet points. Hard ceiling: never exceed {budget} words."
)

## At a turn after a user reaction, prompt on how to reac on this turn to users reaction to last turn 
# used as dictionary so reaction prompt is keyed by reaction type based on the reaction user gave
_REACTION_INSTR = {
    None: "This is the OPENING turn of the session — set the scene in a sentence, then dive in.",
    ReactionType.none: (
        "The listener did NOT react to your last turn — treat that as mild disengagement. Open with a "
        "fresh hook or a different angle to re-earn their attention."
    ),
    ReactionType.comment: (
        "The listener just commented: \"{text}\". Acknowledge or adapt to it naturally, then continue."
    ),
    ReactionType.question: (
        "The listener just asked: \"{text}\". OPEN this turn by answering it directly in your own "
        "spoken voice — a brief, natural acknowledgement (e.g. 'Good question —') then the answer — "
        "grounded in the retrieved answer below. Weave it into the narration; do NOT read it verbatim, "
        "label it 'Answer:', or dump it as a Q&A block. After addressing it, continue the episode.\n"
        "Retrieved answer to ground your response in: {answer}"
    ),
}

###  turn generation style just an idea to make it more personalized to kind of adjust on making it variety focused, versus deep dove in one topic
_MODE_INSTR = {
    "variety": (
        "STYLE: prefer VARIETY — bring a fresh angle and keep the session feeling wide-ranging rather "
        "than dwelling."
    ),
    "consistent": (
        "STYLE: the listener has shown strong interest in this topic — LEAN IN and go a level deeper "
        "rather than broadening."
    ),
}

def _expertise_for(topic: str, persona: Persona) -> str: 
    """
    method to look up listeners expertise level for a certain topic
    """

    for interest in persona.interests: 
        if interest.topic == topic: 
            return interest.expertise.value
    return "intermediate" # default value

def generate_turn(topic: str, 
                  sources: list[CuratedItem], 
                  persona: Persona, 
                  memory: PersonaMemory,
                  recent_gists: list[str], 
                  llm: LLMClient, 
                  *,
                  last_reaction: Reaction | None = None,
                  mode: str | None = None,
                  target_words: int | None = None, ): 
    
    """
    Method to generate oen around 60s turn on `topic`, grounded in `sources` based on 
    inputted `persona` similar logic to write segment 
        - add persona.additional _context attribute
        - continuity across sessions recent gists from past turns past 4 gists, rolling memory.summary past 12 gists 
        - variety and consistency toggle 
    Returns text for the turn, later will wrap into InteractiveTurn object 
    """

    budget = target_words or config.WORDS_PER_TURN # word budget set in method, or config if not entered 
    mode = mode or config.TURN_MODE

    if last_reaction is None: 
        reaction_instr = _REACTION_INSTR[None] # opening turn bc no reaction recorded ye 
    elif last_reaction.type == ReactionType.question: 
        reaction_instr = _REACTION_INSTR[ReactionType.question].format(
            text = last_reaction.text, # inject text of reaction into prompt
            answer = last_reaction.answer or "no answer found")
        
    else: 
        reaction_instr = _REACTION_INSTR[last_reaction.type].format(text = last_reaction.text)

    system = _TURN_SYSTEM.format( 
        budget=budget,
        context=persona.additional_context or "(not specified)",
        reaction_instr=reaction_instr,
        mode_instr=_MODE_INSTR.get(mode, _MODE_INSTR["variety"]),
    ) # inject and enter into system prompt 

    sources_block = "\n\n".join(
        f"[{c.source}] {c.title} ({c.url})\n{c.summary}" for c in sources
    ) or "(no sources available for this topic)" # formatting text block of sources iterating through curated items 


    gists_block = "\n".join(f"- {g}" for g in recent_gists) or "(none yet — this is the first turn)" # creating recent gists block so past 4 turn gists as text block 

    ### user prompt construction 

    user = (
        f"Topic: {topic}\n"
        f"Listener expertise: {_expertise_for(topic, persona)}\n" # pull expertise for topic 
        f"Listener tone: {persona.tone}\n"
        f"AVOID (style): {persona.avoid}\n"
        f"Session summary so far: {memory.summary or '(just starting)'}\n"
        f"Recent turn gists:\n{gists_block}\n\n" # input gist block
        f"Sources to draw on:\n{sources_block}" #sources block
    )

    text = llm.complete(system, user, temperature=0.4) # generate

    return _trim_to_budget(text,budget) # trim to budget

