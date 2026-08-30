
from __future__ import annotations

import re
from pathlib import Path
import math
from pydantic import BaseModel, Field

from .. import config
from ..llm.client import BudgetExceeded, LLMClient
from ..models import Persona
from . import tts



### MODEL DEFINITIONS AND PROMPTS ###

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


_FALLBACK = OpenerSet( # a set of fallback responses incase our candidate bridges are 
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

def fallback_openers() -> dict[str,str]: 
    return _FALLBACK.model_dump()


### persona aligned bridge bank and schemas
## each bridging statement is scored at generation on the same axes as the the persona style vector so the persona vector and the candidate vector are in the same space 

class BridgeCandidate(BaseModel): 
    text: str
    scores: list[float] # the scores given by the llm on the axes to score on for each candidate bridge 


class BridgeBank(BaseModel): 
    """
    the bridge bank generated for the session which is generated and scored by one llm call
    the candidates are generated based on the session vibe to add context of what user is doing 
    """
    question: list[BridgeCandidate] = Field(default_factory = list)
    comment: list[BridgeCandidate] = Field(default_factory = list)
    switch: list[BridgeCandidate] = Field(default_factory = list)
    none: list[BridgeCandidate] = Field(default_factory = list)

    def by_class(self, name:str) -> list[BridgeCandidate]: 
        """
        looks up one of the 4 lists by name its basically getattr
        """
        return getattr(self, name, [])

class StyleSeed(BaseModel): 
    persona_style: list[float] # persona style vector

_AXES = ", ".join(config.STYLE_AXES)


### Axes and their rubric, since these come from 2 independent llm calls  they should be on the same rubric 


## look up for a 0 and 1 on each of tehse axes scores
_AXIS_POLES: dict[str, tuple[str, str]] = {
    "formality": (
        "loose and contracted, slangy, sentence fragments fine",
        "measured and composed, full sentences, no slang",
    ),
    "energy": (
        "low and unhurried, relaxed pacing, room to breathe",
        "brisk and propulsive, driving straight ahead",
    ),
    "warmth": (
        "detached and matter-of-fact, no direct address to the listener",
        "personal and encouraging, speaks WITH the listener not AT them",
    ),
    "technical_register": (
        "plain everyday words, no jargon, nothing assumed",
        "domain vocabulary used unapologetically, assumes fluency",
    ),
    "brevity": (
        "long flowing multi-clause sentences that take their time",
        "clipped and terse, as few words as the thought allows",
    ),
}

def _axis_rubric() -> str: 
    lines = []
    ### pulling in each axisa pair of what a 0 is and what 1 is as the rubric for the llm calls to look back on how to score
    for axis in config.STYLE_AXES: 
        low, high = _AXIS_POLES.get(axis, ("low on this quality", "high on this quality"))
        lines.append(f"  - {axis}:  0.0 = {low}   |   1.0 = {high}")

    return "\n".join(lines)


_RUBRIC = (
    "THE STYLE AXES — these definitions are fixed and are used identically everywhere in this "
    "system, so score against THESE meanings and no other reading of the words:\n"
    f"{_axis_rubric()}\n"
    "Use the full range. 0.5 means genuinely mid, not 'unsure' — if a line truly sits at an "
    "extreme, score it there.\n"
    f"Return the scores as a list of {len(config.STYLE_AXES)} floats in EXACTLY this order, "
    f"one per axis, no names and no reordering: {_AXES}."
)

### BANK GENERATION SYSTEM PROMPT

_BANK_SYSTEM = (
    "You are voicing a podcast host in a live one-on-one session. At any point the listener may "
    "PAUSE you mid-turn and react. You do not know yet what they will say, so ahead of time you "
    "are writing a BANK of short bridges — several per reaction class — for this whole session. "
    "Whichever one fits the moment gets played the instant the listener finishes speaking, while "
    "the real response is still being written.\n\n"
    f"Each bridge is about {OPENER_WORDS} words, six or seven seconds spoken.\n\n"
    "STAY GENERIC. This is the hard part and the most common way to get it wrong. You do not "
    "know what they will ask, which sentence they will pause on, what turn number this is, or "
    "what the answer turns out to be. So a bridge must NOT name a specific angle, guess at what "
    "they mean, claim you rushed something, or promise a particular direction — every one of "
    "those is a guess that the real response will contradict, and the contradiction is what "
    "makes it obvious this was pre-written. Buy a few seconds gracefully and hand off. Nothing "
    "more.\n\n"
    "WRITE SEVERAL FOR EACH CLASS:\n"
    "  question  They asked something. Register it as a good question and signal you are going "
    "to answer it properly. Do not guess what it was about.\n"
    "  comment   They reacted or agreed but asked nothing. Take the agreement and signal you "
    "are about to go further on the same thread.\n"
    "  switch    They want a different topic. Agree easily and signal the move. You are given "
    "the session's topic list below, but you must STILL NOT name a specific destination topic: "
    "the bridge is picked and played before anyone knows which one they asked for, so naming "
    "one is a coin flip the real response will contradict. Refer to the shift itself, or to "
    "there being plenty else on the list.\n"
    "  none      They said nothing, or sounded flat. Do NOT call this out or ask if they are "
    "still there. Signal a change of angle on the same topic.\n\n"
    "WHAT YOU KNOW VS. WHAT YOU DON'T — this is the key distinction that governs everything "
    "below. You do NOT know what the listener will say, ask, or pause on — that is truly "
    "unknown, and guessing at it is what makes a bridge obviously pre-written when it's wrong. "
    "But several things ARE known facts about this listener rather than guesses, all given to "
    "you directly below, and you are free and encouraged to use them concretely:\n"
    "  - the SESSION VIBE — what their listening situation is like right now\n"
    "  - the TOPICS on the table this session — you know the whole set, just not which one "
    "they will pick (see 'switch' above)\n"
    "  - their EXPERTISE on each of those topics\n"
    "  - what earlier sessions already COVERED, if this is a returning listener\n\n"
    "EXPERTISE CALIBRATES VOCABULARY. A beginner's bridges stay in plain words with nothing "
    "assumed; an advanced listener's may use domain vocabulary unapologetically. This is known, "
    "not guessed, so use it — and score it honestly in technical_register rather than writing "
    "every candidate at the same register.\n\n"
    "IF EARLIER SESSIONS ARE LISTED, this is a returning listener, and a few candidates may "
    "lean on that continuity in general terms ('we've been circling this one for a while now'). "
    "Never assert a specific fact, finding, or claim from those sessions — continuity only.\n\n"
    "THE HARD RULE — FACTUAL SAFETY (about the UNKNOWN part only). These are spoken BEFORE "
    "anyone knows what the listener asked or what the sources say, so state NO facts about the "
    "topic/content: no names, numbers, dates, findings, or claims, and do not guess what they "
    "asked or which sentence they paused on. You may refer to the topic in general terms, but "
    "assert nothing about it.\n\n"
    "Never say you are looking something up, thinking, checking, or need a moment. Never "
    "reference being an AI or a system. NO CANDIDATE MAY CONTAIN A QUESTION MARK, anywhere — "
    "not as a rhetorical aside about their situation ('nice out there, isn't it?'), not as a "
    "check-in ('you still there?'), nowhere. This applies even when you reference the vibe "
    "below — describe their situation as a statement, never ask about it. Output spoken prose "
    "only, no headings, no labels, no quotes around the text.\n\n"
    "THE SESSION VIBE IS THE MOST IMPORTANT INPUT FOR HOW THESE SOUND — AND YOU MAY NAME IT "
    "DIRECTLY. You are given a short free-text description of what this listening session is "
    "actually like right now (e.g. 'walking in the park, relaxed' or 'quick distracted commute' "
    "or 'focused evening deep dive'). This is known, not guessed, so don't just match its energy "
    "in the abstract — actually reference the concrete situation in the words themselves when it "
    "fits naturally, especially for 'none' (their attention drifting is exactly what the "
    "situation would predict) and 'switch'. For example, if the vibe is 'walking in the park, "
    "relaxed' and the reaction class is none, a good line is something like 'Ah, I bet you're "
    "busy enjoying the walk right now — let's come at this from a different angle.' If the vibe "
    "is a rushed commute, you might say something like 'Right, quick one before you get where "
    "you're going.' Don't force this into every single candidate if it would feel repetitive or "
    "contrived, but it should show up often enough that the bank clearly reads as written for "
    "THIS specific vibe rather than any generic session. Beyond that direct reference, the vibe "
    "should also shape every candidate's WORD CHOICE, SENTENCE LENGTH, ENERGY, and FORMALITY — a "
    "relaxed/casual vibe should sound short, loose, and plain, not corporate-podcast-host filler "
    "like 'let me break down how that fits into the broader picture.' If your candidates would "
    "read the same regardless of what vibe was given, you have not used it — rewrite them.\n\n"
    "DIVERSITY — THE CANDIDATES IN A CLASS MUST GENUINELY DIFFER, and this matters more than "
    "it sounds. Downstream, one candidate is chosen by comparing its style scores against a "
    "listener profile; if the candidates in a class are near-duplicates of each other, that "
    "comparison has nothing to choose between and the same line gets played every single time. "
    "So the vibe fixes some qualities and leaves others open, and you should move deliberately "
    "in the open ones. If the vibe says relaxed and casual, then formality stays low across the "
    "whole class — but within that, genuinely range: brisk and clipped versus unhurried and "
    "rambling, warm and personal versus dry and matter-of-fact, four words versus twenty. "
    "Vary sentence shape and rhythm too, not just synonyms. What you must NOT do is break the "
    "register the vibe pins — no formal candidate in a casual class — but everything the vibe "
    "does not speak to is yours to spread across.\n\n"
    f"STYLE SCORING — for every candidate you write, also score it on the axes below, each a "
    f"float from 0.0 to 1.0.\n\n{_RUBRIC}\n\n"
    "Score the candidate's ACTUAL voice as written, which should "
    "already reflect the vibe above — a relaxed/casual vibe's candidates should score low on "
    "formality, a focused/technical vibe's candidates should score higher on technical_register. "
    "Score honestly: the numbers must describe the line you actually wrote, never a target you "
    "wish it hit. But because you varied the writing as instructed above, honest scores will "
    "naturally differ — NO TWO CANDIDATES IN THE SAME CLASS MAY HAVE AN IDENTICAL SCORE VECTOR. "
    "If two of your candidates would score the same, that is a sign they are too alike; rewrite "
    "one of them to be genuinely different rather than nudging its numbers."
)


### INTIAL SEED FOR PERSONA VECTOR: COLD USER
_STYLE_SEED_SYSTEM = (
    f"You are scoring a podcast listener's requested delivery style on a small set of axes, each "
    f"a float from 0.0 to 1.0.\n\n{_RUBRIC}\n\n"
    "You are given their tone request, an avoid-list (style "
    "constraints), and their per-topic expertise. Infer where their preferred delivery sits on "
    "each axis — e.g. 'technical but conversational, like talking to a peer' implies high "
    "technical_register, mid formality, mid-high warmth; advanced expertise pushes "
    "technical_register up, beginner pushes it down.\n\n"
    "You are deliberately NOT told what this particular listening session is like (walking, "
    "commuting, a focused evening). That is withheld on purpose: this vector is the listener's "
    "STABLE cross-session baseline and is saved to disk and reused every future session, so "
    "letting one evening's circumstances into it would permanently bias every session that "
    "follows. Score only the durable preference."
)


### UPDATE PROMPT FOR PERSONA STYLE VECTOR AT END OF SESSION
_STYLE_UPDATE_SYSTEM = (
    "You are revising a podcast listener's learned delivery-style profile at the end of one "
    "listening session, using what they actually said during it.\n\n"
    f"{_RUBRIC}\n\n"
    "You are given their CURRENT profile and a log of this session's reactions. Return the "
    "UPDATED profile. This is a REVISION, not a fresh judgment — start from the current "
    "values and move an axis only where this session gives you real evidence. Most axes "
    "should come back unchanged or nearly so. A session with no clear stylistic signal "
    "should return the current profile essentially untouched. This vector persists across "
    "sessions and accumulates, so a small correct nudge beats a large speculative one.\n\n"
    "WEIGH THE EVIDENCE IN THIS ORDER — this ranking matters more than anything else here:\n\n"
    "  1. EXPLICIT REQUESTS ABOUT DELIVERY (strongest by far). The listener directly saying "
    "'can you be more concise', 'that's too technical', 'skip the basics', 'slow down', "
    "'explain it simply'. These are statements about how they want to be spoken to, which is "
    "exactly what this vector encodes. One clear request outweighs everything below it, and "
    "justifies moving that axis decisively.\n\n"
    "  2. ENGAGEMENT PATTERNS BY CONTENT TYPE (real behavioural evidence). Did they ask "
    "follow-up questions on the technical material and go flat on the breezy overview, or the "
    "reverse? Did dense turns earn engagement and light ones lose it? Sustained patterns "
    "across several reactions are meaningful; a single reaction is not.\n\n"
    "  3. HOW THEY THEMSELVES WRITE OR SPEAK (weakest — use only as a tiebreak). Their "
    "vocabulary and sentence length hint at the register they are comfortable in. TREAT THIS "
    "WITH GREAT CAUTION: these reactions are short spoken interruptions or hurried typing, so "
    "they are terse for reasons that have NOTHING to do with how the listener wants to be "
    "spoken TO. Never raise 'brevity' merely because their reactions are short — that is the "
    "single most common way to get this wrong. How someone talks is not how they want to be "
    "talked to.\n\n"
    "USING THE ENGAGEMENT SCORE: each reaction carries an engagement value (roughly -2 to +2). "
    "Use it to decide HOW MUCH TO TRUST that reaction as a sample of this listener, not as a "
    "reward signal for anything. A substantive, engaged reaction tells you far more about them "
    "than a flat 'meh' does. Do NOT try to attribute engagement to any particular thing — you "
    "are not being asked what caused it, only what these reactions reveal about the listener's "
    "preferred delivery.\n\n"
    "Do not invent evidence. If this session simply does not speak to an axis, leave it alone."
)


### METHODS FOR UTILIZING PROMPTS TO CREATE BANK, SYNTHESIZE BANK TTS ###

def _sanitize_bank(bank:BridgeBank, *, on_error = None) -> BridgeBank: 
    """
    drop candidates that we can't rank meaning the LLM generator added/removes a field to the style axes or not at same scale we will drop to not conflate 
    """

    n_axes = len(config.STYLE_AXES)
    dropped = 0 

    for name in OPENER_CLASSES: 
        kept = [] # to keep kept bridge candidates 

        for candidate in bank.by_class(name): # grab the candidates under each opener type 

            if not candidate.text.strip() or len(candidate.scores) != n_axes: 
                dropped += 1 
                continue 

            candidate.scores = [min(1.0, max(0.0, float(score))) for score in candidate.scores]

            kept.append(candidate)

        setattr(bank, name, kept) # setting the attriubute to kept

    if dropped and on_error:
            on_error(f"bridge bank: dropped {dropped} candidate(s) with empty text or "
                     f"score vectors that weren't {n_axes} long")

    return bank

def infer_persona_style(persona:Persona, current_style: list[float], reaction_digest: str, llm:LLMClient) -> list[float]: 
    """
    So after their initial seeded persona style vector given their intial given context on first session, we update their style vector given the past sessions reactions to turns 
    this way we have a vector to compare to
    """
    axes = ", ".join(config.STYLE_AXES)
    current = ", ".join(f"{a}={v:.2f}" for a, v in zip(config.STYLE_AXES, current_style))
    user = (
        f"Listener tone (their own words): {persona.tone}\n"
        f"AVOID (style): {persona.avoid}\n"
        f"Per-topic expertise: "
        + (", ".join(f"{i.topic}: {i.expertise.value}" for i in persona.interests) or "(none)")
        + f"\n\nCURRENT PROFILE: {current}\n\n"
        f"THIS SESSION'S REACTIONS (oldest first):\n{reaction_digest}\n\n"
        f"Return the updated profile as {len(config.STYLE_AXES)} floats in this order: {axes}."
    )
    try:
        seed = llm.structured(_STYLE_UPDATE_SYSTEM, user, StyleSeed,
                                temperature=0.2, priority="background")
    except BudgetExceeded:
        return []
    except Exception:
        return []

    return seed.persona_style

def generate_bridge_bank(persona:Persona, vibe_text:str, llm:LLMClient, *, n_per_class: int = config.BRIDGE_BANK_SIZE, active_topics: list[str] | None = None, memory_summary: str = "", on_error = None) -> BridgeBank:
    """
    structured llm call producing n_per class candidates for each of the opener class 
    and we score on the config.STYLE_AES prior set in this same call upon generation
    """
    topics = list(active_topics) if active_topics else [i.topic for i in persona.interests]
    expertise = ", ".join(f"{i.topic} ({i.expertise.value})" for i in persona.interests)
    summary = (memory_summary or "").strip()

    user = (
            f"THIS SESSION'S VIBE (the main thing that should shape how these sound): "
            f"{vibe_text or '(not specified — use the listener tone below instead)'}\n"
            f"Listener tone: {persona.tone}\n"
            f"AVOID (style): {persona.avoid}\n"
            f"TOPICS ON THE TABLE THIS SESSION (the full set — they may switch to any of "
            f"these, so do not name one): {', '.join(topics) or '(none)'}\n"
            f"THEIR EXPERTISE PER TOPIC: {expertise or '(none)'}\n"
            + (f"COVERED IN EARLIER SESSIONS: {summary}\n" if summary else "")
            + f"Write {n_per_class} candidates for EACH of: {', '.join(OPENER_CLASSES)}, "
            f"all clearly written in the vibe above."
        )

    try: 
        bank = llm.structured(_BANK_SYSTEM, user, BridgeBank, temperature = 0.9, priority= 'background')

    except BudgetExceeded: 
        if on_error:
            on_error("bridge bank generation skipped: LLM request budget exceeded — running on static fallback bridges this session")
        return BridgeBank()
    
    except Exception as err:
        if on_error:
            on_error(f"bridge bank generation failed: {type(err).__name__}: {err} — running on static fallback bridges this session")

        return BridgeBank()

    return _sanitize_bank(bank, on_error = on_error)

def seed_persona_style(persona: Persona, llm: LLMClient) -> list[float]: 
    """
    a call scoring the personas initial style vector based on the config.STYLE_AXES and the intially entered  topics and expertise, avoid list and listener tone 
    This the llms initial read on the user
    """
    levels = ", ".join(f"{i.topic}: {i.expertise.value}" for i in persona.interests) or "(none)" # string set up with each topics expertise. 
    user = (
        f'Listener tone: {persona.tone}\n'
        f'AVOID (style): {persona.avoid}\n'
        f'Per-topic expertise: {levels}'
    )

    try:
        seed = llm.structured(_STYLE_SEED_SYSTEM, user, StyleSeed, temperature=0.2, priority="background")

    except BudgetExceeded:
        return []
    
    except Exception:
        return []

    return seed.persona_style


def synthesize_bridge_bank(bank: BridgeBank, out_dir: Path, target: dict[str,dict[int, str]], *, priority_order: tuple[str, ...] = (COMMENT, QUESTION, SWITCH, NONE), on_error=None) -> None:
    """
    this is the background tts over the 40 candidates we have and we give priority to comment and question as they are the most como 
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents= True, exist_ok = True)

    failures = [] # holding the failed attempts 

    for name in priority_order: # we define teh priority order in the params with comment question first

        candidates = bank.by_class(name) # get candidates for each reaction class
        class_paths = target.setdefault(name, {}) # set path 

        for index, candidate in enumerate(candidates): 
            text = candidate.text.strip() 
            if not text: 
                continue 
            try: 
                wav_path = out_dir/ f'bridge_{name}_{index}.wav'
                class_paths[index] = str(tts.synthesize(text, wav_path))

            except Exception as err: 
                failures.append(f"{name}[{index}]: {type(err).__name__}")


    if failures and on_error: 
        on_error(f'bridge synthesis fialed for {len(failures)} candidates')


def _cosine(a: list[float], b: list[float]) -> float: 
    """
    cosine similarity, with guards 
    """

    if not a or not b or len(a) != len(b): 
        return 0.0 

    norm = math.hypot(*a) * math.hypot(*b)
    dot_product = math.sumprod(a, b)

    if norm: 
        return dot_product / norm
    else: 
        return 0.0


### CLASSIFYING WHAT OPENER CLASS TO LOOK AT GIVEN REACTION REGEX ###

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


def select_opener(bank: BridgeBank | None, bank_audio: dict[str, dict[int, str]], fallback_audio: dict[str, str], persona_style: list[float], reaction_text: str, active_topics: list[str], current_topic: str, *, 
                  recent_indices: dict[str, list[int]] | None = None) -> tuple[str, str, str | None , int| None]:
    """
    given reaction classification, select opener using argmax of cosine of candidate narrative style vector and persona style vector 
    so we choose the best fitting stylized candidate within that class and if the bank isnt ready we move to the static preloaded generic candidates
    dont want repeated candidate bridging narratives so we keep a list of recent indices that already have been used and use othe ones besides those ones 
    """

    name = classify_for_opener(reaction_text, active_topics, current_topic)

    candidates = bank.by_class(name) if bank is not None else []

    if not candidates or not persona_style: 
        text = fallback_openers()[name]
        wav = (fallback_audio or {}).get(name)

        return name, text, wav, None 

    ranked = sorted(
        range(len(candidates)), key = lambda i: _cosine(candidates[i].scores, persona_style), reverse = True # ranking using simple cosine similarity
    )

    excluded = set((recent_indices or {}).get(name, [])) # the excluded set of indices used recently

    class_audio = (bank_audio or {}).get(name) or {} # get audio files for that class 

    playable = [i for i in ranked if i in class_audio] # filter for playable audio files if in ranked 

    if playable:
         best_index  = next((i for i in playable if i not in excluded), playable[0]) # walk the list for best candidate that is in playable and not in the excluded recent indexes
         winner = candidates[best_index]

         return name, winner.text, class_audio[best_index], best_index

    fallback_wav = (fallback_audio or {}).get(name)

    if fallback_wav is not None: 
        return name, fallback_openers()[name], fallback_wav, None

    ## No audio fallback
    best_index = next((i for i in ranked if i not in excluded), ranked[0])
    winner = candidates[best_index]

    return name, winner.text, None, best_index
