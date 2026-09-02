from __future__ import annotations

from utils import (
    USER_IG_FILE,
    USER_PROFILE_FILE,
    read_text_file,
    read_user_keywords,
    write_json_file,
    llm_json,
    truncate_text,
    read_podcast_settings,
)


def extract_user_profile():
    ig_text = read_text_file(USER_IG_FILE)
    user_keywords = read_user_keywords()
    settings = read_podcast_settings()
    podcast_type = settings.get("podcast_type", "ig_analysis")

    if not ig_text.strip():
        raise RuntimeError(
            "data/user_ig_profile.txt is empty. "
            "Run python src\\00_parse_instagram_export.py first."
        )

    prompt = f"""
You are a user profiling assistant for a personalized podcast generation system.

You are given:
1. Instagram-exported user signals
2. Optional user-provided keywords

Your task:
- Extract broad interests and content preferences from Instagram-exported signals.
- If user-provided keywords exist, use them as the current podcast focus.
- If user-provided keywords are empty, infer podcast focus from Instagram interests only.
- Do not include private personal data.
- Do not infer sensitive attributes.

Privacy rules:
- Do NOT include email, phone number, exact location, device ID, birthday, income, religion, political view, or health condition.
- Do NOT mention private account details.
- Only use broad interests, topics, style preferences, and content preferences.

Instagram exported signals:
{truncate_text(ig_text, 20000)}

User-provided keywords:
{user_keywords if user_keywords else "No explicit user keywords were provided."}


If podcast_type is "ig_analysis":
- Focus on analyzing the user's Instagram-derived interests.
- Recommended news queries should be about lifestyle, K-pop, travel, fashion, food, campus life, or other IG-derived themes.

If podcast_type is "stock_analysis":
- Focus on stock market and investment-related analysis.
- Use the Instagram profile only as a personalization layer.
- Recommended news queries should include stock market, technology stocks, AI stocks, semiconductor stocks, ETFs, Firstrade investing, and beginner-friendly investment news.
- Do not provide financial advice. Use educational and informational tone only.
Podcast type:
{podcast_type}

Return ONLY valid JSON.
Do not wrap the JSON in markdown.
Do not use ```json.

Use this exact schema:

{{
  "interests": [
    "interest 1",
    "interest 2"
  ],
  "preferred_topics": [
    "topic 1",
    "topic 2"
  ],
  "style_preferences": [
    "style 1",
    "style 2"
  ],
  "podcast_focus_keywords": [
    "keyword 1",
    "keyword 2"
  ],
  "recommended_news_queries": [
    "query 1",
    "query 2",
    "query 3"
  ],
  "topics_to_avoid": [
    "topic 1",
    "topic 2"
  ],
  "privacy_notes": [
    "note 1",
    "note 2"
  ]
}}

Important:
- "interests" must not be empty.
- "preferred_topics" must not be empty.
- "recommended_news_queries" must not be empty.
- If user keywords are empty, create podcast_focus_keywords from Instagram interests.
"""

    profile = llm_json(
        prompt,
        temperature=0.2,
        max_output_tokens=2500,
    )

    write_json_file(USER_PROFILE_FILE, profile)

    print(f"Saved user profile to {USER_PROFILE_FILE}")
    print("Interests:", profile.get("interests", []))

    return profile


if __name__ == "__main__":
    extract_user_profile()