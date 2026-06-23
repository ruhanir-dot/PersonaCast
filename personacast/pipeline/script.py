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

from ..llm.client import LLMClient
from ..models import CuratedItem, Persona, TopicSegment

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


