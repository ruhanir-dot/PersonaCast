"""
helper functions that the retrrieval graph nodes call , graph.py is only. orchestration

we use a trhead pool for tavily http calls, parallelising calls for speed no more asyncio
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from ..models import CuratedItem, PersonaMemory, RetrievedItem
from ..pipeline.dedup import _norm_url
from ..pipeline.retrieval import arxiv as arxiv_src
from ..pipeline.retrieval import tavily as web_src
from ..pipeline.retrieval.clean import clean_query

def search_web_parallel(queries: list[str]) -> tuple[list[RetrievedItem], list[str]]:
    """
    reun web query concurenttly and collect results into a list, take in raw queries from planning agent
    returns the retrieved items and errors we encounter 
    """

    queries = [q for q in (clean_query(q) for q in queries) if q] # run through query clean query add cleaned query to query list

    if not queries: # if nothing return empty
        return [], []

    items = [] # list of RetrievedItem
    errors = [] # any errors 

    with ThreadPoolExecutor(max_workers = len(queries)) as pool: 
        futures = {pool.submit(web_src.search_web, q): q for q in queries}

        for future in as_completed(futures):
            try:
                items.extend(future.result())
            except Exception as e:
                errors.append(f"web search failed q={futures[future]!r}: {type(e).__name__}: {e}")

    return items, errors

def search_arxiv_serial(queries: list[str]) -> tuple[list[RetrievedItem], list[str]]:
    """
    query arxiv one at a time because of rate limiting!
    """
    items = [] # list of RetrievedItem
    errors = [] # any errors 

    for query in queries:
        try:
            items.extend(arxiv_src.search_arxiv(query))
        except Exception as e:
            errors.append(f"arxiv search failed q={query!r}: {type(e).__name__}: {e}")

    return items, errors

def novel_items(candidates: list[RetrievedItem], seen_urls: set[str]) -> list[RetrievedItem]:
    """
   filter against url and make sure we have candidates not seen before, we check memory.covered[topic]
   and we cover urls seen in the current candidate list of topics if queries of one topic uncover same source over and over
    """
    kept= [] # list of RetrievedItem
    seen = set(seen_urls)

    for item in candidates:
        key = _norm_url(item.url)
        if key in seen:
            continue
        seen.add(key)
        kept.append(item)

    return kept


### PERSONA MEMORY HELPERS FOR RETRIEVAL GRAPH

def covered_urls(memory: PersonaMemory | None, topic: str) -> set[str]:
    """
    normalized URLs of every source this listener was already shown for this topi
    """
    if memory is None:
        return set()
    return {_norm_url(source.url) for source in memory.covered.get(topic, [])}


def covered_block(memory: PersonaMemory | None, topic: str, *, max_lines: int = 12) -> str:
    """
    build the already covered text block handed to the planning agent, so it writes queries for a topic mentioned across session
    at a different angle than already covered
    """

    if memory is None:
        return ""

    lines = []
    seen = set()

    def add(text: str) -> None:
        text = (text or "").strip()
        if text and text.lower() not in seen:
            seen.add(text.lower())
            lines.append(f"- {text}")

    for source in memory.covered.get(topic, []):
        add(source.title)

    for line in (memory.summary or "").splitlines():
        if f"[{topic}]" in line:
            add(line.split("]", 1)[-1])

    for reaction in memory.reactions:
        if reaction.topic == topic and reaction.anchor_source:
            add(reaction.anchor_source)

    return "\n".join(lines[:max_lines])


