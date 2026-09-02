from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from utils import (
    OUTPUT_DIR,
    PORTFOLIO_SNAPSHOT_FILE,
    USER_PROFILE_FILE,
    read_json_file,
    read_user_keywords,
    write_json_file,
    llm_json,
    truncate_text,
)

YOUTUBE_PROFILE_FILE = OUTPUT_DIR / "youtube_profile.json"
USER_PREFERENCE_PROFILE_FILE = OUTPUT_DIR / "user_preference_profile.json"

DEFAULT_STOCK_MARKET_TOPICS = [
    "stock market",
    "technology stocks",
    "AI stocks",
    "semiconductor stocks",
    "ETFs",
    "earnings news",
    "inflation and interest rates",
]


def as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def simple_fallback_preference(
    ig_profile: dict[str, Any],
    youtube_profile: dict[str, Any],
    portfolio_snapshot: dict[str, Any] | None,
    user_keywords: str,
) -> dict[str, Any]:
    """
    Rule-based fallback when LLM JSON parsing fails or API is unavailable.
    """
    topic_counter: Counter[str] = Counter()
    source_map: dict[str, set[str]] = {}

    def add_topic(topic: str, source: str, weight: int = 1) -> None:
        topic = " ".join(str(topic).split()).strip()
        if not topic:
            return
        topic_counter[topic] += weight
        source_map.setdefault(topic, set()).add(source)

    for key in ["interests", "preferred_topics", "podcast_focus_keywords", "recommended_news_queries"]:
        for item in as_list(ig_profile.get(key)):
            add_topic(item, "instagram", weight=2)

    for item in as_list(youtube_profile.get("top_keywords"))[:30]:
        add_topic(item, "youtube", weight=1)

    for item in as_list(youtube_profile.get("top_channels"))[:20]:
        add_topic(item, "youtube", weight=1)

    for item in as_list(youtube_profile.get("recent_search_queries"))[:20]:
        add_topic(item, "youtube", weight=2)

    for item in as_list(youtube_profile.get("recent_watch_titles"))[:20]:
        add_topic(item, "youtube", weight=1)

    if portfolio_snapshot:
        for holding in portfolio_snapshot.get("holdings", []) if isinstance(portfolio_snapshot, dict) else []:
            if isinstance(holding, dict):
                add_topic(holding.get("ticker", ""), "stock", weight=3)
                add_topic(holding.get("name", ""), "stock", weight=2)

    for item in DEFAULT_STOCK_MARKET_TOPICS:
        add_topic(item, "stock_market_fallback", weight=1)

    for line in user_keywords.splitlines():
        add_topic(line, "current_prompt", weight=5)

    ranked = topic_counter.most_common(30)

    interests = []
    for topic, score in ranked:
        interests.append(
            {
                "topic": topic,
                "score": score,
                "sources": sorted(source_map.get(topic, [])),
            }
        )

    search_queries = []
    for item in interests[:10]:
        topic = item["topic"]
        if "news" in topic.lower() or "latest" in topic.lower():
            search_queries.append(topic)
        else:
            search_queries.append(f"latest news about {topic}")

    return {
        "profile_type": "multi_source_user_preference",
        "method": "rule_based_fallback",
        "top_interests": interests,
        "personalized_news_queries": search_queries[:8],
        "preferred_news_categories": [
            "technology",
            "business",
            "entertainment",
            "lifestyle",
            "travel",
            "food",
            "finance",
        ],
        "ranking_keywords": [item["topic"] for item in interests[:30]],
        "source_summary": {
            "instagram_used": bool(ig_profile),
            "youtube_used": bool(youtube_profile),
            "portfolio_used": bool(portfolio_snapshot),
            "current_prompt_used": bool(user_keywords.strip()),
        },
        "privacy_notes": [
            "The profile uses broad topics only.",
            "Raw YouTube history and private details are not sent to the podcast script.",
            "No sensitive attributes are inferred.",
        ],
    }


def build_user_preference_profile() -> dict[str, Any]:
    ig_profile = read_json_file(USER_PROFILE_FILE) or {}
    youtube_profile = read_json_file(YOUTUBE_PROFILE_FILE) or {}
    portfolio_snapshot = read_json_file(PORTFOLIO_SNAPSHOT_FILE) or {}
    user_keywords = read_user_keywords()

    prompt = f"""
You are building a multi-source user preference profile for a personalized news podcast recommender.

Inputs:
1. Instagram-derived profile: broad interests and content preferences.
2. YouTube Takeout profile: watch/search/subscription-derived broad media interests.
3. Optional stock / portfolio snapshot. It may be empty.
4. Optional current user prompt. If present, it should strongly influence the current episode.

Important privacy rules:
- Do not include private personal data.
- Convert raw history into broad interests only.
- Do not quote private history items unless they are generic public titles/topics.

Instagram-derived profile:
{truncate_text(str(ig_profile), 8000)}

YouTube profile:
{truncate_text(str(youtube_profile), 12000)}

Stock / portfolio snapshot:
{truncate_text(str(portfolio_snapshot), 5000)}

Current user prompt:
{user_keywords if user_keywords.strip() else "No current prompt provided."}

Return ONLY valid JSON with this schema:
{{
  "profile_type": "multi_source_user_preference",
  "top_interests": [
    {{
      "topic": "interest topic",
      "score": 0.0,
      "sources": ["instagram", "youtube", "stock_market", "current_prompt"]
    }}
  ],
  "preferred_news_categories": ["category 1", "category 2"],
  "personalized_news_queries": [
    "query 1",
    "query 2",
    "query 3"
  ],
  "ranking_keywords": [
    "keyword 1",
    "keyword 2"
  ],
  "source_summary": {{
    "instagram_used": true,
    "youtube_used": true,
    "portfolio_used": false,
    "current_prompt_used": false
  }},
  "privacy_notes": ["note 1", "note 2"]
}}

Guidelines:
- top_interests should merge overlapping interests across sources.
- Higher score means stronger relevance for news recommendation.
- personalized_news_queries should be suitable for web/news search.
- If stock / portfolio snapshot is empty, still include general market topics only as weak fallback interests.
- If current prompt exists, include it as the strongest short-term interest.
"""

    try:
        preference = llm_json(prompt, temperature=0.2, max_output_tokens=3000)
    except Exception as e:
        print("LLM preference builder failed; using rule-based fallback.")
        print(f"Error: {e}")
        preference = {}

    if not isinstance(preference, dict) or not preference.get("top_interests"):
        preference = simple_fallback_preference(
            ig_profile=ig_profile,
            youtube_profile=youtube_profile,
            portfolio_snapshot=portfolio_snapshot,
            user_keywords=user_keywords,
        )
    else:
        # Add source summary protection.
        preference.setdefault(
            "source_summary",
            {
                "instagram_used": bool(ig_profile),
                "youtube_used": bool(youtube_profile),
                "portfolio_used": bool(portfolio_snapshot),
                "current_prompt_used": bool(user_keywords.strip()),
            },
        )
        preference.setdefault(
            "privacy_notes",
            [
                "The profile uses broad topics only.",
                "Sensitive attributes and private details are excluded.",
            ],
        )

    write_json_file(USER_PREFERENCE_PROFILE_FILE, preference)

    print(f"Saved user preference profile to {USER_PREFERENCE_PROFILE_FILE}")
    print("Top interests:")
    for item in preference.get("top_interests", [])[:10]:
        if isinstance(item, dict):
            print(f"- {item.get('topic', '')} ({item.get('score', '')})")
        else:
            print(f"- {item}")

    return preference


if __name__ == "__main__":
    build_user_preference_profile()
