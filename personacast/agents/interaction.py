"""
The interaction agent reads what listener just did and semantically understands it 
"""

from __future__ import annotations

from pydantic import BaseModel

from .. import config
from ..llm.client import LLMClient
from ..models import CuratedItem, Persona, ReactionType, SessionState, InteractiveTurn

class ReactionPlan(BaseModel):

    intent: str
    sentiment: float | None
    engagement_delta: float | None
    requested_topic: str
    reason: str
    gist: str

    anchor_source_index: int

    needs_answer: bool
    answered: bool
    answer: str
    sources_used: list[int]



### richer inten mapping to. the existing three value reaction type 
## so it interprets the reaction type even its not distinctly looks like a question and if its asking in some since can interpret as question and if its a bored response we treat as none
_INTENT_TO_TYPE = {
    "question":     ReactionType.question,
    "confusion":    ReactionType.question,   
    "deeper":       ReactionType.question,   
    "simpler":      ReactionType.comment,
    "agreement":    ReactionType.comment,
    "disagreement": ReactionType.comment,
    "switch":       ReactionType.comment,
    "boredom":      ReactionType.none,       
    "none":         ReactionType.none,
}


_SYSTEM = (
    "You are reading ONE listener reaction to ONE turn of a personalized podcast, and deciding "
    "what it means. You are given the turn they reacted to, the sentence they paused on (if "
    "any), and their history with this show.\n\n"
    "INTENT — pick exactly one:\n"
    "  question      they asked something, explicitly or implicitly. NOTE: a question mark is "
    "NOT required. 'wait why does that matter' is a question.\n"
    "  confusion     they did not follow it ('you lost me', 'what?')\n"
    "  deeper        they want more on this specific thing\n"
    "  simpler       too technical, or too much jargon for them\n"
    "  agreement     positive, engaged, but not asking anything\n"
    "  disagreement  they push back on the content\n"
    "  boredom       disengaged, dismissive, flat ('yeah ok', 'sure', 'meh')\n"
    "  switch        they want a different topic\n"
    "  none          empty or contentless\n\n"
    "SENTIMENT: -1.0 to 1.0. Judge genuine engagement, not politeness. 'yeah ok' is mildly "
    "negative — it is a listener tuning out. 'that's fascinating' is strongly positive. A "
    "pointed question is POSITIVE: they cared enough to ask.\n\n"
    "ENGAGEMENT_DELTA: points to add to this topic's running score, in the range -2.0 to +2.0. "
    "For reference, the hand-tuned system this replaces used a flat +2 for any question, +1 for "
    "any comment and -1 for silence. You have more information than that, so USE THE RANGE — a "
    "lukewarm 'sure, ok' and a genuinely excited follow-up question must not score the same. Do "
    "not simply reproduce those three constants.\n\n"
    "HOW TO WEIGH A PAUSE. If you are shown a sentence they paused on, they stopped playback "
    "mid-turn at that exact point. Read it like this:\n"
    "  paused AND said something  -> a strong positive engagement signal on top of whatever the "
    "reaction itself says. They interrupted a playing podcast to speak, which takes effort. "
    "Push the delta further from zero in the direction the reaction already points.\n"
    "  paused AND said nothing    -> still negative. A stop with nothing after it more likely "
    "means they walked away or got distracted than that they were gripped. Do NOT treat the "
    "pause alone as engagement.\n"
    "  no pause shown             -> judge on the reaction text alone.\n"
    "The paused sentence also tells you WHICH part of the turn they mean, so use it to "
    "disambiguate a vague reaction like 'wait, that bit'.\n\n"
    "REQUESTED_TOPIC: if they want a different topic, return the one from the ACTIVE TOPICS list "
    "they mean, matched by MEANING and not by wording — 'can we do hoops instead' means "
    "'basketball'. Never return the topic they are currently on. Return \"\" if they are not "
    "asking to switch.\n\n"
    "IMPORTANT — WHICH TOPIC THE DELTA APPLIES TO. If you set requested_topic, the "
    "engagement_delta is applied to THAT topic, not the one they are currently on (the topic "
    "they are leaving is penalised separately and automatically). So on a switch, "
    "engagement_delta must be POSITIVE and reflect how much they want the NEW topic — asking "
    "for something is by definition interest in it. Do not score their boredom with the current "
    "topic here; that is already handled.\n\n"
    "GIST: ONE short plain sentence capturing what the TURN (not the reaction) covered. It "
    "becomes a 'what we already said' note so later turns do not repeat it. No preamble.\n\n"
    "--- You are ALSO doing two source-grounded jobs in this same pass. ---\n\n"
    "ANCHOR_SOURCE_INDEX — you are given the numbered SOURCES the turn was synthesized from. "
    "If you were shown a sentence they paused on, return the index of the single source that "
    "sentence is most directly based on, so the next turn can go deeper on exactly it. If no "
    "source clearly supports it, or no pause was shown, return -1.\n\n"
    "ANSWERING — set needs_answer=true only if there is a real question to answer (intent "
    "question, confusion or deeper). When it is false, set answered=false, answer=\"\", "
    "sources_used=[] and web_query=\"\".\n"
    "  When needs_answer is true, try to answer it from the SOURCES below. This is a short "
    "spoken interjection, not an essay — a few sentences, then stop.\n"
    "  FACTUAL FIDELITY — this is critical. Use ONLY what the sources actually state. Never "
    "invent a name, number, date, statistic, or mechanism to fill a gap, and never transfer a "
    "fact from one context to another. A generic-but-true answer beats a specific-but-invented "
    "one.\n"
    "  GROUNDING — if at least one source supports an answer, set answered=true, write the "
    "answer, and list ONLY the indices you actually drew on in sources_used. If NO source "
    "addresses it, set answered=false, leave answer and sources_used empty.\n"
    "  PERSONALIZATION — calibrate depth and terminology to their expertise (advanced: precise "
    "domain terms, assume fundamentals; beginner: intuition over technical detail), and honor "
    "the AVOID list as STYLE guidance. Match their tone."
)

_WEB_ANSWER_SYSTEM = (
    "You are answering a single listener question that interrupted a podcast, grounded ONLY in "
    "the web results provided. This is a short spoken interjection, not an essay — a few "
    "sentences, then stop.\n\n"
    "FACTUAL FIDELITY — use ONLY what the results actually state. Never invent a name, number, "
    "date, statistic, or mechanism to fill a gap. If the results still do not answer it, say so "
    "honestly in one line and set answered=false.\n\n"
    "Calibrate depth to the listener's expertise and honor the AVOID list as STYLE guidance."
)

def source_listing(sources: list[CuratedItem]) -> str:
    return "\n".join(
        f"[{i}] {s.title} — {s.summary}" for i, s in enumerate(sources)
    ) or "(no sources available for this topic)"


def interpret(reaction_text: str, turn: InteractiveTurn, session_state: SessionState,
              persona: Persona, llm: LLMClient, *, active_topics: list[str],
              sources: list[CuratedItem], anchor_snippet: str = "") -> ReactionPlan:

    memory = session_state.memory

    recent = [
        f"- turn {r.iteration} [{r.topic}] {r.type.value}: {r.text[:80]}"
        for r in memory.reactions[-4:]
        ]

    user = (
            f"CURRENT TOPIC: {turn.topic}\n"
            f"ACTIVE TOPICS (the only ones they can switch to): {active_topics}\n"
            f"Listener expertise on this topic: "
            f"{next((i.expertise.value for i in persona.interests if i.topic == turn.topic), 'intermediate')}\n"
            f"Tone they asked for: {persona.tone}\n"
            f"AVOID (style): {persona.avoid}\n"
            f"\nTHE TURN THEY REACTED TO:\n{turn.text}\n"
            + (f"\nTHEY PAUSED ON THIS SENTENCE:\n{anchor_snippet}\n" if anchor_snippet else "")
            + (f"\nRECENT HISTORY:\n" + "\n".join(recent) + "\n" if recent else "")
            + (f"\nRUNNING SUMMARY OF THE SESSION SO FAR:\n{memory.summary}\n" if memory.summary else "")
            + f"\nTHEIR REACTION:\n{reaction_text.strip() or '(said nothing)'}\n"
            + f"\nSOURCES (this turn was synthesized from these):\n{source_listing(sources)}"
        )

    return llm.structured(_SYSTEM, user, ReactionPlan, temperature=0.0)


class WebAnswer(BaseModel):
    answered: bool
    answer: str


def answer_from_web(question: str, persona: Persona, web_items: list[CuratedItem], llm: LLMClient) -> WebAnswer:
    user = (
        f"Listener question: {question}\n"
        f"Listener interests + expertise: "
        f"{[(i.topic, i.expertise.value) for i in persona.interests]}\n"
        f"Listener tone: {persona.tone}\n"
        f"AVOID (style): {persona.avoid}\n\n"
        f"Web results:\n{source_listing(web_items)}"
    )
    return llm.structured(_WEB_ANSWER_SYSTEM, user, WebAnswer, temperature=0.2)



def to_reaction_type(intent: str) -> ReactionType:
    """
    map intent to type based on prior defined dict
    """
    return _INTENT_TO_TYPE.get(intent, ReactionType.comment)

def clamp_delta(delta: float, *, is_switch: bool = False) -> float:
    """
    clamp delta to -2 to 2 range, 

    when switch is true and user asked for sitch 
    """
    delta = max(-2.0, min(2.0, float(delta)))

    ### previously topic getting sitched to would get the engagement delta of preior topic
    ### upon switch engage_comment +1 is added to next topic, instead of it recieving the current topics enagagemnt delta llm produces
    ### topic getting switched away from is done by engage_switch_away -2 which we dont touch
    if is_switch: 
        return max(delta, config.ENGAGE_COMMENT)
    return delta
