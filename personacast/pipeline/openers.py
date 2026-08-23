
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from .. import config
from ..llm.client import BudgetExceeded, LLMClient
from ..models import InteractiveTurn, Persona
from . import tts


QUESTION= "question"
COMMENT= "comment"
SWITCH ="switch"
NONE = "none"

OPENER_CLASSES = (QUESTION, COMMENT, SWITCH, NONE)


OPENER_WORDS = config.OPENER_WORDS


class OpenerSet(BaseModel):
    question: str
    comment: str
    switch: str
    none: str


_SYSTEM = (
    "You are voicing a podcast host in a live one-on-one session. The listener has just PAUSED "
    "you mid-turn and is about to react. You do not know yet what they will say, so you are "
    "writing FOUR short bridges — one for each way they might react. Whichever one fits gets "
    "played the instant they finish speaking, while the real response is still being written.\n\n"
    f"Each bridge is about {OPENER_WORDS} words, six or seven seconds spoken.\n\n"
    "STAY GENERIC. This is the hard part and the most common way to get it wrong. You do not "
    "know what they asked, which sentence they paused on, or what the answer turns out to be. "
    "So a bridge must NOT name a specific angle, guess at what they mean, claim you rushed "
    "something, or promise a particular direction — every one of those is a guess that the "
    "real response will contradict, and the contradiction is what makes it obvious this was "
    "pre-written. Buy a few seconds gracefully and hand off. Nothing more.\n\n"
    "WRITE ONE FOR EACH:\n"
    "  question  They asked something. Register it as a good question and signal you are going "
    "to answer it properly. Do not guess what it was about.\n"
    "  comment   They reacted or agreed but asked nothing. Take the agreement and signal you "
    "are about to go further on the same thread.\n"
    "  switch    They want a different topic. Agree easily and signal the move, naming the new "
    "topic only if you were given one.\n"
    "  none      They said nothing, or sounded flat. Do NOT call this out or ask if they are "
    "still there. Signal a change of angle on the same topic.\n\n"
    "THE HARD RULE — FACTUAL SAFETY. These are spoken BEFORE anyone knows what the listener "
    "asked or what the sources say, so state NO facts at all: no names, numbers, dates, "
    "findings, or claims. You may refer to the topic in general terms and match the energy of "
    "what you were just saying, but assert nothing.\n\n"
    "Never say you are looking something up, thinking, checking, or need a moment. Never "
    "reference being an AI or a system. Do not end with a question. Output spoken prose only, "
    "no headings, no labels, no quotes around the text."
)


_FALLBACK = OpenerSet(
    question=(
        "That's a good question, and I'd rather take it properly than give you the quick "
        "version of it. Let me do that."
    ),
    comment=(
        "Right and that's worth staying with for a moment rather than moving straight on. "
        "Let me take it a bit further."
    ),
    switch=(
        "Sure, we can leave this here and move on. Let me pick the thread up somewhere else."
    ),
    none=(
        "Let me come at this from a different angle, because I think there's a better way into "
        "it than the one I was taking."
    ),
)


def fallback_openers() -> dict[str, str]:
    return _FALLBACK.model_dump()


def generate_openers(turn: InteractiveTurn, persona: Persona, llm: LLMClient, *, likely_switch_topic: str | None = None) -> dict[str, str]:
    """
    using llm call to generate opening statements, this is done while current turn is playing
    """
    user = (
        f"TOPIC: {turn.topic}\n"
        f"Listener tone: {persona.tone}\n"
        f"AVOID (style): {persona.avoid}\n"
        f"Their situation right now: {persona.additional_context or '(not specified)'}\n"
        + (f"If they ask to switch, the most likely topic is: {likely_switch_topic}\n"
           if likely_switch_topic else "")
        + f"\nTHE TURN YOU JUST DELIVERED (they paused somewhere in this):\n{turn.text}"
    )

    try:
        openers = llm.structured(_SYSTEM, user, OpenerSet, temperature=0.7, priority="background")
    except BudgetExceeded:
        return fallback_openers()
    except Exception:
        return fallback_openers()

    produced = openers.model_dump()
    return {key: (produced.get(key) or "").strip() or _FALLBACK.model_dump()[key]
            for key in OPENER_CLASSES}


def synthesize_openers(openers: dict[str, str], out_dir: Path, iteration: int, *, on_error=None) -> dict[str, str]:
    """
    pre redner in tts each opener after generating
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}
    failures: list[str] = []
    for name in OPENER_CLASSES:
        text = openers.get(name, "").strip()
        if not text:
            continue
        try:
            target = out_dir / f"turn_{iteration}_open_{name}.wav"
            paths[name] = str(tts.synthesize(text, target))
        except Exception as err:
            failures.append(f"{name}: {type(err).__name__}")

    if failures and on_error:
        on_error(f"opener synthesis failed for {', '.join(failures)}")
    return paths


_STRONG_WH = {"what", "why", "how", "when", "where", "who", "whose", "which"}

_SWITCH_PATTERNS = tuple(re.compile(p) for p in (
    r"\b(can|could|shall|should|may)\s+we\s+(switch|move|change|talk|do|go)\b",
    r"\blet'?s\s+(switch|move|change|talk about|do|go)\b",
    r"\b(switch|change|changing)\s+(the\s+|to\s+(a|another)\s+)?(topic|subject|gears)\b",
    r"^\s*switch\b",
    r"\bmov(e|ing)\s+on\b",
    r"\b(different|another|new|next)\s+(topic|subject|thing)\b",
    r"\bsomething\s+else\b",
    r"\benough\s+(about|of)\b",
    r"\b(bored|done|over)\s+(of|with)\s+(this|that|it)\b",
    r"\bnot\s+interested\b",
))

_FLAT = {
    "yeah", "yeah ok", "yeah okay", "ok", "okay", "sure", "sure ok", "mm", "mhm", "mmhm",
    "meh", "right", "i guess", "yeah i guess", "fine", "cool", "k", "uh huh", "alright",
}


def classify_for_opener(reaction_text: str, active_topics: list[str], current_topic: str) -> str:
    """
    classify opener using simple regex
    """
    from . import memory

    stripped = (reaction_text or "").strip()
    if not stripped:
        return NONE

    lowered_full = stripped.lower()

    if memory.detect_requested_topic(stripped, active_topics, exclude=current_topic):
        return SWITCH

    if any(pattern.search(lowered_full) for pattern in _SWITCH_PATTERNS):
        return SWITCH

    lowered = stripped.lower()

    if lowered.strip(".!") in _FLAT:
        return NONE

    if "?" in lowered:
        return QUESTION

    from .interactive import _WH_WORDS
    words = [w.strip(".,!") for w in lowered.split()]

    if words and words[0] in _WH_WORDS:
        return QUESTION

    if any(word in _STRONG_WH for word in words[:5]):
        return QUESTION

    return COMMENT


def select_opener(pool: dict[str, str], audio: dict[str, str], reaction_text: str, active_topics: list[str], current_topic: str) -> tuple[str, str, str | None]:
    """
    given reaction classification, select opener
    """
    name = classify_for_opener(reaction_text, active_topics, current_topic)
    text = (pool or {}).get(name) or fallback_openers()[name]
    return name, text, (audio or {}).get(name)
