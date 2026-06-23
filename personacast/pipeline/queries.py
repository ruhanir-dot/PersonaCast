"""
Step: Per Topic search query generation 

"""




from __future__ import annotations
from datetime import date
from pydantic import BaseModel

from .. import config
from ..llm.client import LLMClient
from ..models import TopicPlan
from ..models import Expertise

class _QueryList(BaseModel): # query list pydantic object
    queries: list[str]

_SYSTEM = (
    "You generate search queries for a podcast research agent. Output PLAIN natural-language "
    "keyword queries — the kind you'd actually type into a search box.\n"
    "STRICT: no search operators of any kind — no site:, no AND/OR/NOT, no parentheses, no quotes, "
    "no year-range syntax. Source targeting is handled elsewhere; your query must be source-"
    "agnostic so it works on a general web search.\n"
    "Make queries concrete and recency-focused (favor 'recent', 'latest', 'this year'), NOT "
    "encyclopedic 'what is X' definitions. Work for ANY domain — sports, cooking, fashion, ML — so "
    "do NOT assume the topic is academic or that papers/preprints exist for it.\n"
    "The current year is {year}. If you put a year in a query, use {year} (or 'this year'); never "
    "use an older year from memory — stale years return stale results.\n"
    "Calibrate VOCABULARY (not operators) to expertise: an advanced listener gets precise domain "
    "terminology; a beginner gets plainer wording."
)

def generate_queries(plan: TopicPlan, llm:LLMClient) -> list[str]: 
    user = (
        f"Topic: {plan.topic}\n"  # set to plan topic
        f"Listener expertise: {plan.expertise.value}\n" # expertise 
        f"Generate {config.QUERIES_PER_TOPIC} search queries." # cap at how many queries per topic
    )
    result = llm.structured(_SYSTEM.format(year=date.today().year), user, _QueryList) # pass through structured which will return as _QueryList object 

    return result.queries[: config.QUERIES_PER_TOPIC] # cap at the queries defined in the config 


if __name__ == "__main__":

    llm = LLMClient() # instance of LLM client object

    ### test code to check out diff in topic level
    for level in (Expertise.beginner, Expertise.advanced):
        plan = TopicPlan(topic="recommender systems", expertise=level, word_budget=900)
        print(f"\n=== {level.value} ===")
        for q in generate_queries(plan, llm):
            print(" -", q)
