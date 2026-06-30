"""
naive qa implementation after script generation a listener can "pause" their episode and ask a question
we first attempt to give an answer grounded in the sources.json curates sources, if its not present in our curated sources 
we put the query in to out web search method get new sources and ground the LLM response in the related serarch results

"""

from __future__ import annotations 
import json
from pathlib import Path
from pydantic import BaseModel, Field

from ..llm.client import LLMClient
from ..models import CuratedItem, Persona, RetrievedItem
from .retrieval.tavily import search_web

# configure how much of the fetched web pages content we want as grounding context
_WEB_SUMMARY_CHARS = 1200 

_QA_SYSTEM = (
    "You answer a single listener question that interrupts a podcast, grounded ONLY in the "
    "sources provided below. This is a short spoken interjection, not an essay — answer in a "
    "few sentences, then stop.\n\n"
    "FACTUAL FIDELITY — this is critical. Use ONLY what the sources actually state. Never invent "
    "a name, number, date, statistic, or mechanism to fill a gap, and never transfer a fact from "
    "one context to another. If the sources do not answer the question, say so honestly rather "
    "than guessing.\n\n"
    "GROUNDING — if at least one source supports an answer, set answered=true, write the answer, "
    "and list ONLY the sources you actually drew on in sources_used. If NO source addresses the "
    "question, set answered=false, give a one-line honest note that it isn't covered, and leave "
    "sources_used empty.\n\n"
    "PERSONALIZATION — calibrate depth and terminology to the listener's expertise (advanced: "
    "precise domain terms, assume fundamentals; beginner: intuition over technical detail), and "
    "honor the AVOID list as STYLE guidance (e.g. 'basic ML 101 explanations' => skip "
    "fundamentals). Match the listener's tone."
)

### pydantic object structure 
class AnswerSource(BaseModel):
    title: str
    url: str

class Answer(BaseModel): 
    answered: bool # true if the LLM determined that the sources cover the question and use previously curated sources as grounding
    answer: str # our answer
    sources_used: list[AnswerSource] = Field(default_factory = list) # list of the sources used
    used_web: bool = False # default is False, if we use web bc answered = False switch to true



### loading the curated sources method

def load_curated(sources_path) -> dict[str, list[CuratedItem]] :
    """
    looking at the source.json recreate the topic key and list of curated items structures
    """

    raw = json.loads(Path(sources_path).read_text())
    ## iterate through the topic, items pairs  extracted through the raw jason extraction =
    return {
        topic: [CuratedItem.model_validate(item )for item in items]
        for topic, items in raw.items()
    }

def flatten_curated(curated: dict[str, list[CuratedItem]]) -> list[CuratedItem]:
    """
    question isn't previously attached to a topic so the naive implementation just looks at the episode's whole curated set
    topics detached
    """
    ## double wrapped for loop where we first iterate through the items in the curated then break from there
    return[ item for items in curated.values() for item in items]

def select_context(question, persona: Persona, curated_items: list[CuratedItem]): 
    """
    simply return all curated items, this is a naive placeholder func want to maybe add things 
    """
    return curated_items

### grounded answering methods 

def answer_from_sources(question, persona, items : list[CuratedItem], llm: LLMClient): 
    """
    given the question, persona, and curated sources used in script we produce an LLM call to retrieve an answer
    """

    items = select_context(question, persona, items)
    sources_block = "\n\n".join(
        f"[{c.source}] {c.title} ({c.url})\n{c.summary}" for c in items
        ) or "(no sources available)"
    
    ## User prompt
    user = (
        f"Listener question: {question}\n"
        f"Listener interests + expertise: "
        f"{[(i.topic, i.expertise.value) for i in persona.interests]}\n"
        f"Listener tone: {persona.tone}\n"
        f"AVOID (style): {persona.avoid}\n\n"
        f"Sources you may use:\n{sources_block}"
    )

    return llm.structured(_QA_SYSTEM, user, Answer, temperature=0.2) 

def _web_to_curated(items : list[RetrievedItem], limit_chars: int= _WEB_SUMMARY_CHARS) -> list[CuratedItem]: 
    """
    we are getting our detched web results and turning into the CuratedItem shep so our web fallback is using same 
    answer_from_sources call for consitendcy
    """
    
    return [
        CuratedItem(source="web", title=item.title, url=item.url, summary=item.content[:limit_chars])
        for item in items
    ]


### qa pipeline function.

def answer_question(question:str, persona: Persona, curated_items: list[CuratedItem], llm: LLMClient, *,  allow_web: bool = True) -> Answer:
    """
    based on return llm output answered bool will tell us if it is pososble to use the curated source as grounding or not
    based on that premise we can search the web with no recency window filter and re-answer using those reults as grounding 
    """
    answer = answer_from_sources(question, persona, curated_items, llm)

    if answer.answered or not allow_web: # if answered true simply just return the answer at hand
        return answer
    
    ## else...
    web_items = search_web(question, topic="general", days=None) # grab web items if needed
    
    if not web_items:# just in case web search returns nothing to not waste llm call 
        return answer
    
    answer = answer_from_sources(question, persona, _web_to_curated(web_items), llm)
    answer.used_web = True

    return answer




