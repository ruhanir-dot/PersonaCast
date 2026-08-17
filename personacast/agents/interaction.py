"""
The interaction agent reads what listener just did and semantically understands it 
"""

from __future__ import annotations

from pydantic import BaseModel

from .. import config
from ..llm.client import LLMClient
from ..models import Persona, ReactionType, SessionState, InteractiveTurn

class ReactionRead(BaseModel): 
    """
    what the aget decides after recieving a reaction, we append this schema to system prompt! when generating response to reaaction on next turn
    """

    ### as mentioned this appends to the reaction model in models.py as well, this model is more for request one is storage record
    intent: str = "" # question, comment, NOne --> fills type when agent appends 
    sentiment: float | None = None # -1 to 1 engagment signal determined by agent
    engagement_delta: float | None = None # the engagemnet point accumulation the agent decides hat poijt value this reaction got -2 to 2 
    requested_topic: str  # keep as string to make sure, llm wont fill string with None string "" if none 
    reason: str # one line for trace


    needs_answer: bool # should we run grounded qa on this 
    gist: str # one sentence summarizing turn, replacing old summarize turn


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
    "becomes a 'what we already said' note so later turns do not repeat it. No preamble."
)

def read_reaction(reaction_text: str, turn : InteractiveTurn, session_state:SessionState, persona:Persona, llm:LLMClient, *, active_topics: list[str],  anchor_snippet: str = "") -> ReactionRead: 
    """
    structured LLM call, given inputs we return ReactionRead  object 
      actual turn text --> turn, rest of params are exlanatory
    """

    memory = session_state.memory # our growing session memory

    recent = [ # grab 4 most recent historic reactions the user has from their persona memory file
        f"- turn {r.iteration} [{r.topic}] {r.type.value}: {r.text[:80]}"
        for r in memory.reactions[-4:]
        ]

    user = ( # user prompt
            f"CURRENT TOPIC: {turn.topic}\n"
            f"ACTIVE TOPICS (the only ones they can switch to): {active_topics}\n"
            f"Listener expertise on this topic: "
            f"{next((i.expertise.value for i in persona.interests if i.topic == turn.topic), 'intermediate')}\n"
            f"Tone they asked for: {persona.tone}\n"
            f"\nTHE TURN THEY REACTED TO:\n{turn.text}\n"
            + (f"\nTHEY PAUSED ON THIS SENTENCE:\n{anchor_snippet}\n" if anchor_snippet else "")
            + (f"\nRECENT HISTORY:\n" + "\n".join(recent) + "\n" if recent else "")
            + (f"\nRUNNING SUMMARY OF THE SESSION SO FAR:\n{memory.summary}\n" if memory.summary else "")
            + f"\nTHEIR REACTION:\n{reaction_text.strip() or '(said nothing)'}"
        )

    return llm.structured(_SYSTEM, user, ReactionRead, temperature=0.0)

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
