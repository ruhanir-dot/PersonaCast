

import html
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.utils import PROJECT_ROOT, llm_json


SOURCE_GROUP_FILE = Path(__file__).with_name("source_group.json")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
MAX_SEARCH_RESULTS = 8
MIN_ARTICLE_WORDS = 450
TAVILY_RESULTS_FILE = PROJECT_ROOT / "data" / \
    "output" / "tavily_search_results.json"

load_dotenv(PROJECT_ROOT / ".env")


def load_source_groups() -> dict[str, dict[str, Any]]:
    with SOURCE_GROUP_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    groups = payload.get("source_groups")
    if not isinstance(groups, dict):
        raise ValueError("source_group.json must contain source_groups.")
    return groups


def plan_web_search(
    seed: dict[str, Any],
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    group_names = list(groups)
    result = llm_json(
        prompt=(
            "A YouTube video has no usable transcript. Create a precise English "
            "web-search plan to find a trustworthy article about the same topic.\n\n"
            f"Video title: {seed.get('title', '')}\n"
            f"Description: {seed.get('description', '')}\n"
            f"Channel: {seed.get('channel', '')}\n\n"
            "Choose one to three source groups from the allowed list and write one "
            "search query. Do not include site: operators in the query."
        ),
        system="Return valid JSON only.",
        temperature=0.0,
        max_output_tokens=250,
        json_schema={
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "minLength": 3, "maxLength": 220},
                "source_groups": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 3,
                    "items": {"type": "string", "enum": group_names},
                },
            },
            "required": ["search_query", "source_groups"],
            "additionalProperties": False,
        },
    )
    return {
        "search_query": " ".join(str(result["search_query"]).split()),
        "source_groups": list(dict.fromkeys(result["source_groups"])),
    }


def allowed_domains(
    source_groups: list[str], groups: dict[str, dict[str, Any]]
) -> list[str]:
    return list(
        dict.fromkeys(
            domain
            for group_name in source_groups
            for domain in groups[group_name].get("domains", [])
            if isinstance(domain, str) and domain.strip()
        )
    )


def save_tavily_search_results(
    seed: dict[str, Any],
    plan: dict[str, Any],
    results: list[dict[str, str]],
) -> None:
    TAVILY_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    saved_results: list[dict[str, Any]] = []
    if TAVILY_RESULTS_FILE.exists():
        with TAVILY_RESULTS_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            saved_results = payload

    saved_results.append(
        {
            "title": seed.get("title"),
            "description": seed.get("description"),
            "channel": seed.get("channel"),
            "url": seed.get("url"),
            "search_query": plan["search_query"],
            "source_groups": plan["source_groups"],
            "results": results,
        }
    )
    with TAVILY_RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(saved_results, file, ensure_ascii=False, indent=2)


def tavily_search(search_query: str, domains: list[str]) -> list[dict[str, str]]:
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required.")

    search_payload: dict[str, Any] = {
        "query": search_query,
        "search_depth": "basic",
        "max_results": MAX_SEARCH_RESULTS,
        "include_answer": False,
    }
    if domains:
        search_payload["include_domains"] = domains

    request = Request(
        TAVILY_SEARCH_URL,
        data=json.dumps(search_payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # nosec B310: fixed HTTPS API URL
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload.get("results"), list):
        raise RuntimeError(
            f"Tavily did not return search results: {payload}"
        )

    results = []
    for item in payload.get("results", []):
        link = str(item.get("url") or "").strip()
        host = urlparse(link).hostname or ""
        if not link or (domains and not any(host == domain or host.endswith(f".{domain}") for domain in domains)):
            continue
        results.append(
            {
                "title": " ".join(str(item.get("title") or "").split()),
                "url": link,
                "snippet": " ".join(str(item.get("content") or "").split()),
            }
        )
    return results


def select_matching_result(seed: dict[str, Any], results: list[dict[str, str]]) -> dict[str, str] | None:
    if not results:
        return None
    result = llm_json(
        prompt=(
            "Choose one search result only when it is clearly about the same specific "
            "topic as the YouTube video. Reject results that are merely keyword-related.\n\n"
            f"YouTube title: {seed.get('title', '')}\n"
            f"YouTube description: {seed.get('description', '')}\n\n"
            f"Search results: {json.dumps(results, ensure_ascii=False)}"
        ),
        system="Return valid JSON only.",
        temperature=0.0,
        max_output_tokens=120,
        json_schema={
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "selected_index": {"type": "integer", "minimum": 0, "maximum": len(results) - 1},
            },
            "required": ["keep", "selected_index"],
            "additionalProperties": False,
        },
    )
    return results[int(result["selected_index"])] if result["keep"] else None


def extract_article_text(url: str) -> str:
    request = Request(
        url, headers={"User-Agent": "personalized-podcast-agent/1.0"})
    # nosec B310: source URL was returned by trusted-domain search
    with urlopen(request, timeout=30) as response:
        page = response.read().decode("utf-8", errors="replace")
    page = re.sub(
        r"<!--.*?-->|<(script|style|noscript|nav|footer|header|aside)[^>]*>.*?</\1>", " ", page, flags=re.IGNORECASE | re.DOTALL)
    text = html.unescape(re.sub(r"<[^>]+>", " ", page))
    return " ".join(text.split())


def search_web_source(seed: dict[str, Any]) -> dict[str, Any] | None:
    try:
        groups = load_source_groups()
        plan = plan_web_search(seed, groups)
        results = tavily_search(
            plan["search_query"],
            allowed_domains(plan["source_groups"], groups),
        )
        save_tavily_search_results(seed, plan, results)
        result = select_matching_result(seed, results)
        if result is None:
            return None
        article_text = extract_article_text(result["url"])
        if len(article_text.split()) < MIN_ARTICLE_WORDS:
            return None
        return {
            "source_type": "web_article",
            "title": result["title"],
            "url": result["url"],
            "article_text": article_text,
            "transcript_word_count": 0,
            "article_word_count": len(article_text.split()),
            "language": "English",
            "language_code": "en",
            "search_query": plan["search_query"],
            "source_groups": plan["source_groups"],
        }
    except (KeyError, RuntimeError, TypeError, ValueError, OSError) as exc:
        print(f"Web-source fallback failed: {exc}", flush=True)
        return None
