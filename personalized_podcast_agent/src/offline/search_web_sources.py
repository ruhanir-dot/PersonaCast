import html
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from requests.exceptions import RequestException

from src.searxng_search import search_web as searxng_search
from src.utils import PROJECT_ROOT, llm_json


SOURCE_GROUP_FILE = Path(__file__).with_name("source_group.json")
MAX_SEARCH_RESULTS = 8
MIN_ARTICLE_WORDS = 450
WEB_SEARCH_RESULTS_FILE = PROJECT_ROOT / \
    "data" / "output" / "web_search_results.json"

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
) -> dict[str, Any]:
    result = llm_json(
        prompt=(
            "A YouTube video has no usable transcript. Create a concise keyword-style "
            "web search query for the video's main topic. Keep important names and "
            "terms from the title or description. Do not add generic words such as "
            "article, analysis, review, or trustworthy. Do not invent details that "
            "are not in the video information.\n\n"
            f"Video title: {seed.get('title', '')}\n"
            f"Description: {seed.get('description', '')}\n"
            f"Channel: {seed.get('channel', '')}\n\n"
            "Return one concise search query. Do not include site: operators."
        ),
        system="Return valid JSON only.",
        temperature=0.0,
        max_output_tokens=120,
        json_schema={
            "type": "object",
            "properties": {
                "search_query": {"type": "string", "minLength": 3, "maxLength": 220},
            },
            "required": ["search_query"],
            "additionalProperties": False,
        },
    )
    return {
        "search_query": " ".join(str(result["search_query"]).split()),
        "source_groups": [],
    }


def plan_question_search(
    *,
    question: str,
    trunk_script: str,
    source_item: dict[str, Any],
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    group_names = list(groups)
    result = llm_json(
        prompt=(
            "The supplied source does not contain enough information to answer "
            "a question. Create a precise English web-search plan for a trustworthy "
            "source that directly answers the question about the same topic.\n\n"
            f"Source title: {source_item.get('title', '')}\n"
            f"Current trunk: {trunk_script}\n"
            f"Question: {question}\n\n"
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


def save_web_search_results(
    seed: dict[str, Any],
    plan: dict[str, Any],
    results: list[dict[str, str]],
    status: str | None = None,
) -> None:
    WEB_SEARCH_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    saved_results: list[dict[str, Any]] = []
    if WEB_SEARCH_RESULTS_FILE.exists():
        with WEB_SEARCH_RESULTS_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if isinstance(payload, list):
            saved_results = payload

    attempt = {
        "title": seed.get("title"),
        "description": seed.get("description"),
        "channel": seed.get("channel"),
        "url": seed.get("url"),
        "search_query": plan["search_query"],
        "source_groups": plan["source_groups"],
        "results": results,
    }
    if status:
        attempt["status"] = status
    saved_results.append(attempt)
    with WEB_SEARCH_RESULTS_FILE.open("w", encoding="utf-8") as file:
        json.dump(saved_results, file, ensure_ascii=False, indent=2)


def search_searxng(
    search_query: str, domains: list[str]
) -> list[dict[str, str]]:
    """Search the local SearXNG instance and keep approved source domains."""
    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    # SearXNG has no Tavily-style include_domains parameter. Search each
    # approved domain with a site: query, then enforce the allowlist locally.
    queries = [f"site:{domain} {search_query}" for domain in domains]
    if not queries:
        queries = [search_query]

    for query in queries:
        raw_results = searxng_search(query, limit=MAX_SEARCH_RESULTS)
        if not isinstance(raw_results, list):
            raise RuntimeError(
                "SearXNG search_web() must return a list of results.")

        for item in raw_results:
            if not isinstance(item, dict):
                continue
            link = str(item.get("url") or "").strip()
            host = (urlparse(link).hostname or "").lower()
            allowed = any(
                host == domain.lower() or host.endswith(f".{domain.lower()}")
                for domain in domains
            )
            if not link or (domains and not allowed) or link in seen_urls:
                continue
            seen_urls.add(link)
            snippet = item.get("snippet") or item.get("content") or ""
            results.append(
                {
                    "title": " ".join(str(item.get("title") or "").split()),
                    "url": link,
                    "snippet": " ".join(str(snippet).split()),
                }
            )
            if len(results) >= MAX_SEARCH_RESULTS:
                return results

    # Search once without site: if targeted engine syntax returned nothing;
    # still enforce the approved-domain allowlist on the returned URLs.
    if domains and not results:
        raw_results = searxng_search(search_query, limit=MAX_SEARCH_RESULTS)
        if not isinstance(raw_results, list):
            raise RuntimeError(
                "SearXNG search_web() must return a list of results.")
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            link = str(item.get("url") or "").strip()
            host = (urlparse(link).hostname or "").lower()
            allowed = any(
                host == domain.lower() or host.endswith(f".{domain.lower()}")
                for domain in domains
            )
            if not link or not allowed or link in seen_urls:
                continue
            seen_urls.add(link)
            snippet = item.get("snippet") or item.get("content") or ""
            results.append(
                {
                    "title": " ".join(str(item.get("title") or "").split()),
                    "url": link,
                    "snippet": " ".join(str(snippet).split()),
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


def select_matching_answer_result(
    *,
    question: str,
    trunk_script: str,
    source_item: dict[str, Any],
    results: list[dict[str, str]],
) -> dict[str, str] | None:
    if not results:
        return None
    result = llm_json(
        prompt=(
            "Choose one search result only when it is about the same specific topic "
            "and directly helps answer the question. Reject results that are merely "
            "keyword-related or do not contain evidence for the answer.\n\n"
            f"Source title: {source_item.get('title', '')}\n"
            f"Current trunk: {trunk_script}\n"
            f"Question: {question}\n\n"
            f"Search results: {json.dumps(results, ensure_ascii=False)}"
        ),
        system="Return valid JSON only.",
        temperature=0.0,
        max_output_tokens=120,
        json_schema={
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "selected_index": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": len(results) - 1,
                },
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


def search_web_source(seed: dict[str, Any]) -> dict[str, Any]:
    plan: dict[str, Any] = {"search_query": "", "source_groups": []}
    results: list[dict[str, str]] = []
    try:
        plan = plan_web_search(seed)
        results = search_searxng(
            plan["search_query"],
            [],
        )
        if not results:
            status = "no_search_results"
            article_text = ""
        else:
            result = select_matching_result(seed, results)
            if result is None:
                status = "irrelevant_results"
                article_text = ""
            else:
                try:
                    article_text = extract_article_text(result["url"])
                except Exception:
                    status = "page_fetch_failed"
                    article_text = ""
                else:
                    if len(article_text.split()) < MIN_ARTICLE_WORDS:
                        status = "article_too_short"
                        article_text = ""
                    else:
                        status = "found"

        save_web_search_results(seed, plan, results, status)
        if status != "found":
            return {"status": status}
        return {
            "status": "found",
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
    except Exception as exc:
        print(f"Web-source fallback failed: {exc}", flush=True)
        return {"status": "search_failed"}


def search_web_answer(
    *,
    question: str,
    trunk_script: str,
    source_item: dict[str, Any],
) -> dict[str, Any] | None:
    try:
        groups = load_source_groups()
        plan = plan_question_search(
            question=question,
            trunk_script=trunk_script,
            source_item=source_item,
            groups=groups,
        )
        results = search_searxng(
            plan["search_query"],
            allowed_domains(plan["source_groups"], groups),
        )
        search_seed = {
            "title": source_item.get("title"),
            "description": trunk_script,
            "channel": source_item.get("channel", ""),
            "url": source_item.get("url"),
        }
        save_web_search_results(search_seed, plan, results)
        result = select_matching_answer_result(
            question=question,
            trunk_script=trunk_script,
            source_item=source_item,
            results=results,
        )
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
    except (KeyError, RuntimeError, TypeError, ValueError, OSError, RequestException) as exc:
        print(f"Web-answer fallback failed: {exc}", flush=True)
        return None
