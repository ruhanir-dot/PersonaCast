"""
retrieval orchestration, decide sources per topic
based off our llm generated queries we use on web search, using our config stem topic triggers we know when to use arxiv 
doesnt make sense to go to arxov for something like food or basketball 
we also use async on the websearch calls (cant use reddit right now needs approval)
arxov is not parallizable
"""

from __future__ import annotations

import asyncio

from ... import config as cfg
from ...models import RetrievedItem, TopicPlan
from . import arxiv as arxiv_src
from . import tavily as web_src
from .clean import clean_query
from ...models import Expertise


def _wants_arxiv(topic) -> bool:
    """
    keyword checker to see if we need to use arxiv
    """
    topic_lower = topic.lower()
    return any(keyword in topic_lower for keyword in cfg.STEM_HINT_KEYWORDS) # return bool with is its contained in topic

async def _gather_parallel(queries: list[str]) -> list[RetrievedItem]:
    """
    Searching web using queries no reddit yet
    """

    tasks = []
    for query in queries:
        tasks.append(asyncio.to_thread(web_src.search_web, query)) # for ech query use tavily module and tavily api with query 
        # async allows concurrent queries not one at a time 
    
    results = await asyncio.gather(*tasks, return_exceptions=True) # runs all tasks in tasks concurrently
    items = [] # list of RetrievedItem objects
    for r in results:
        if isinstance(r, Exception): # skip exception objects in results
            continue
        items.extend(r)
    return items

def retrieve_topic(plan: TopicPlan, queries: list[str]) -> list[RetrievedItem]:
    """
    retrieval for one topic with parallel web search and conditional arxiv
    """
    queries = [clean_query(query) for query in queries] # strip any residual search operators
    
    items = asyncio.run(_gather_parallel(queries)) # run web searches concurrentl and collect results in items

    if _wants_arxiv(plan.topic): # seruial arxiv search
        for query in queries:  
            try:
                items.extend(arxiv_src.search_arxiv(query))
            except Exception: 
                continue
    
    return items

if __name__ == "__main__":
    for topic in ("recommender systems", "sports", "agentic AI", "cooking"):
        plan = TopicPlan(topic=topic, expertise=Expertise.intermediate, word_budget=900)
        srcs = "web" + (" + arxiv" if _wants_arxiv(plan.topic) else "")
        print(f"{topic:<22} -> {srcs}")
