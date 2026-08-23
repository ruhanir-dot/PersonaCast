"""
file holds our search_arxiv which we use conditionally in retrieve.py only for topics that need it 
sleep between arxiv calls to abide to rate limit

fixes from v1: sort by relevance not submitted data or else unrelated papers leak into our content
"""

from __future__ import annotations

import time

import requests

from ... import config as cfg
from ...models import RetrievedItem
from .clean import clean_query
import arxiv


class _TimeoutSession(requests.Session):

    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", cfg.ARXIV_REQUEST_TIMEOUT_SECONDS)
        return super().request(*args, **kwargs)


_client = None


def _get_client() -> arxiv.Client:
    global _client

    if _client is None:
        _client = arxiv.Client(
            page_size=max(1, cfg.RESULTS_PER_QUERY),
            delay_seconds=cfg.ARXIV_RATE_LIMIT_SECONDS,
            num_retries=cfg.ARXIV_MAX_RETRIES,
        )
        _client._session = _TimeoutSession()

    return _client


def search_arxiv(query: str, max_results: int | None = None) -> list[RetrievedItem]:
    limit = max_results or cfg.RESULTS_PER_QUERY # can either be defined in func or config

    # define search params
    search = arxiv.Search( 
        query=clean_query(query),
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance, 
    )
    items = [] # list of retrieved items

    try:
        for result in _get_client().results(search):
            items.append( # add into items retrieveditem structure
                RetrievedItem(
                    source="arxiv",
                    title=result.title,
                    url=result.entry_id,
                    content=result.summary,  #summary gives us the abstract
                    published=result.published.date().isoformat() if result.published else None,
                )
            )
    except Exception:
        time.sleep(cfg.ARXIV_RATE_LIMIT_SECONDS)
        raise

    return items