import requests

SEARXNG_URL = "http://localhost:8080"


def search_web(query: str, limit: int = 5) -> list[dict]:
    response = requests.get(
        f"{SEARXNG_URL}/search",
        params={
            "q": query,
            "format": "json",
            "language": "en",
        },
        timeout=20,
    )
    response.raise_for_status()

    results = response.json().get("results", [])

    # 保留常見搜尋結果欄位，方便接回既有的文章擷取與整理程式
    return [
        {
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "content": item.get("content", ""),
        }
        for item in results[:limit]
    ]
