from __future__ import annotations

import os
from datetime import date
from typing import Any

from utils import OUTPUT_DIR, llm_json, read_json_file, write_json_file
from listening_history import has_listened_to_news_today, covered_topics_today, recent_primary_topics
from interest_topic_selector import select_one_topic

PODCAST_REQUEST_FILE = OUTPUT_DIR / "podcast_request.json"
USER_PREFERENCE_PROFILE_FILE = OUTPUT_DIR / "user_preference_profile.json"

CONTEXT_GUIDANCE = {
    "auto": {
        "tone": "natural and adaptive",
        "information_density": "medium",
        "explanation_depth": "medium",
        "sentence_complexity": "medium",
        "avoid_visual_references": True,
    },
    "relaxing": {
        "tone": "warm, casual, and story-driven",
        "information_density": "low",
        "explanation_depth": "light",
        "sentence_complexity": "low",
        "prefer_sources": ["youtube", "public_web", "instagram_interests"],
        "avoid_visual_references": True,
    },
    "driving": {
        "tone": "clear, conversational, and easy to follow by audio only",
        "information_density": "medium",
        "explanation_depth": "medium",
        "sentence_complexity": "low",
        "repeat_key_points": True,
        "prefer_daily_update": True,
        "avoid_visual_references": True,
    },
    "learning": {
        "tone": "educational and structured",
        "information_density": "high",
        "explanation_depth": "detailed",
        "sentence_complexity": "medium",
        "define_technical_terms": True,
        "include_examples": True,
        "prefer_educational_content": True,
        "avoid_visual_references": True,
    },
}


def _top_interest_topics(profile: dict[str, Any], limit: int = 5) -> list[str]:
    topics: list[str] = []
    for item in profile.get("top_interests", []) if isinstance(profile, dict) else []:
        topic = str(item.get("topic", "") if isinstance(item, dict) else item).strip()
        if topic and topic not in topics:
            topics.append(topic)
    return topics[:limit]



def _probe_major_news(preference_profile: dict[str, Any]) -> dict[str, Any]:
    """Lightweight pre-routing search. Failure safely falls back to no major event."""
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        return {"available": False, "importance": 0.0, "topic": "", "query": ""}

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        today = date.today().isoformat()
        interests = _top_interest_topics(preference_profile, limit=4)
        interest_text = ", ".join(interests)
        queries = [
            f"most important breaking news today {today}",
            f"major technology business and world news today {today}",
        ]
        if interest_text:
            queries.append(f"major news today related to {interest_text} {today}")

        candidates: list[dict[str, str]] = []
        for query in queries:
            response = client.search(query=query, search_depth="basic", max_results=3)
            for item in response.get("results", []):
                candidates.append({
                    "query": query,
                    "title": str(item.get("title", "")),
                    "content": str(item.get("content", ""))[:900],
                    "url": str(item.get("url", "")),
                })
        if not candidates:
            return {"available": False, "importance": 0.0, "topic": "", "query": ""}

        assessed = llm_json(f"""
Assess whether any candidate is important enough to proactively interrupt the user's normal personalized podcast choice.
Use a high threshold. Routine updates should score below 0.80. Events with broad public, economic, safety, scientific, or major industry impact may score 0.90 or higher.
User interests: {interests}
Already covered today: {covered_topics_today()}
Candidates: {candidates}
Return JSON only:
{{
  "importance": 0.0,
  "topic": "short topic",
  "query": "best search query for a podcast about this event",
  "reason": "brief reason"
}}
""", temperature=0.0, max_output_tokens=400)
        importance = max(0.0, min(float(assessed.get("importance", 0.0)), 1.0))
        return {
            "available": importance >= 0.75,
            "importance": importance,
            "topic": str(assessed.get("topic", "")).strip(),
            "query": str(assessed.get("query", "")).strip(),
            "reason": str(assessed.get("reason", "")).strip(),
        }
    except Exception as exc:
        return {"available": False, "importance": 0.0, "topic": "", "query": "", "error": repr(exc)}


def _route_prompt_first(prompt: str, preference_profile: dict[str, Any]) -> dict[str, Any]:
    try:
        result = llm_json(f"""
The user explicitly entered a podcast request. Treat it as the primary instruction.
Choose the retrieval strategy that best answers it; do not classify only by stock keywords.
Available strategies:
- news: current events, recent developments, public-company news, markets, politics, technology updates, or anything requiring fresh reporting.
- interest_digest: entertainment, lifestyle, interviews, tutorials, explainers, evergreen knowledge, hobbies, YouTube/public-web content.

User prompt: {prompt}
Optional long-term interests: {preference_profile}
Return JSON only:
{{
  "podcast_type": "news" or "interest_digest",
  "topics": ["..."],
  "content_sources": ["news", "public_web", "youtube"],
  "confidence": 0.0,
  "routing_reason": "brief reason",
  "effective_prompt": "a focused retrieval query preserving the user's intent"
}}
""", temperature=0.0, max_output_tokens=500)
        if result.get("podcast_type") not in {"news", "interest_digest"}:
            raise ValueError("invalid type")
        return result
    except Exception:
        return {
            "podcast_type": "interest_digest",
            "topics": [prompt],
            "content_sources": ["public_web", "youtube"],
            "confidence": 0.6,
            "routing_reason": "The explicit prompt is used as the episode's main topic.",
            "effective_prompt": prompt,
        }


def route_podcast_request(*, user_prompt: str = "", has_uploaded_paper: bool = False,
                          listening_context: str = "auto", duration_minutes: int = 3) -> dict[str, Any]:
    prompt = (user_prompt or "").strip()
    context = (listening_context or "auto").strip().lower()
    if context not in CONTEXT_GUIDANCE:
        context = "auto"

    preference_profile = read_json_file(USER_PREFERENCE_PROFILE_FILE) or {}
    listened_news = has_listened_to_news_today()
    major_news = {"available": False, "importance": 0.0, "topic": "", "query": ""}
    selected_interest = select_one_topic(context)
    rotated_interest = str(selected_interest.get("topic", "")).strip()
    rotated_topics = [rotated_interest] if rotated_interest else _top_interest_topics(preference_profile, limit=1)
    preference_source = str(selected_interest.get("preference_source", "fallback")).strip().lower()

    if has_uploaded_paper:
        result = {
            "selection_mode": "paper_upload",
            "podcast_type": "paper_podcast",
            "confidence": 1.0,
            "routing_reason": "An uploaded PDF has priority.",
            "topics": [prompt] if prompt else [],
            "content_sources": ["uploaded_paper"],
            "effective_prompt": prompt,
        }
    elif prompt:
        result = {"selection_mode": "prompt_first", **_route_prompt_first(prompt, preference_profile)}
    else:
        major_news = _probe_major_news(preference_profile)
        major_threshold = 0.90
        if major_news.get("importance", 0.0) >= major_threshold:
            result = {
                "selection_mode": "automatic",
                "podcast_type": "news",
                "confidence": major_news["importance"],
                "routing_reason": "A high-impact new event exceeded the proactive-news threshold.",
                "topics": [major_news.get("topic", "major news")],
                "content_sources": ["news"],
                "effective_prompt": major_news.get("query") or "most important breaking news today",
            }
        elif context == "relaxing":
            result = {
                "selection_mode": "automatic",
                "podcast_type": "interest_digest",
                "confidence": 0.9,
                "routing_reason": "Relaxing context favors recent IG/YouTube-aligned entertainment and lifestyle content.",
                "topics": rotated_topics,
                "primary_topic": rotated_interest,
                "preference_source": preference_source,
                "topic_selection": selected_interest,
                "content_sources": ["youtube", "public_web"],
                "effective_prompt": "",
            }
        elif context == "learning":
            result = {
                "selection_mode": "automatic",
                "podcast_type": "interest_digest",
                "confidence": 0.85,
                "routing_reason": "Learning context favors an educational deep dive based on the user's interests.",
                "topics": rotated_topics,
                "primary_topic": rotated_interest,
                "preference_source": preference_source,
                "topic_selection": selected_interest,
                "content_sources": ["public_web", "youtube"],
                "effective_prompt": "",
            }
        elif not listened_news:
            result = {
                "selection_mode": "automatic",
                "podcast_type": "news",
                "confidence": 0.82,
                "routing_reason": "The user has not listened to a news update today, so a personalized daily update is selected.",
                "topics": _top_interest_topics(preference_profile),
                "content_sources": ["news"],
                "effective_prompt": "",
            }
        else:
            result = {
                "selection_mode": "automatic",
                "podcast_type": "interest_digest",
                "confidence": 0.82,
                "routing_reason": "A news update was already played today, so the system selects fresh personalized interest content.",
                "topics": rotated_topics,
                "primary_topic": rotated_interest,
                "preference_source": preference_source,
                "topic_selection": selected_interest,
                "content_sources": ["youtube", "public_web"],
                "effective_prompt": "",
            }

    request = {
        **result,
        "user_prompt": prompt,
        "has_uploaded_paper": bool(has_uploaded_paper),
        "listening_context": context,
        "duration_minutes": int(duration_minutes),
        "generation_preferences": CONTEXT_GUIDANCE[context],
        "daily_state": {"listened_to_news_today": listened_news},
        "major_news_probe": major_news,
        "recent_primary_topics": recent_primary_topics(4),
    }
    write_json_file(PODCAST_REQUEST_FILE, request)
    return request


if __name__ == "__main__":
    print(route_podcast_request())
