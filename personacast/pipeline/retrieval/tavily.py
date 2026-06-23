

from __future__ import annotations
from ... import config as cfg
from ...models import RetrievedItem


def search_web(query, max_results = None) -> list[RetrievedItem]:
    from tavily import TavilyClient

    client = TavilyClient(api_key = cfg.TAVILY_API_KEY)

    responses = client.search(
        query=query,
        max_results=max_results or cfg.RESULTS_PER_QUERY,
        include_raw_content=True,
        topic="news", # focus on news
        days=cfg.SEARCH_RECENCY_DAYS, # last N days for recenct
    )

    items = []
    for response in responses.get('results', []): 
        items.append(
            RetrievedItem(
                source="web",
                title=response.get("title", ""),
                url=response.get("url", ""),
                # want clean full text but will fall back to snippet if no raw content 
                content=response.get("raw_content") or response.get("content", ""),
                published=response.get("published_date"),
        )
        )

    return items
    
    
if __name__ == "__main__":
    for item in search_web("recommender systems arxiv preprint 2026"):
        print(f"- {item.title}\n  {item.url}\n  {len(item.content)} chars\n")
