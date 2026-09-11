import json
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from dotenv import load_dotenv

from src.utils import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")
API_BASE_URL = "https://www.googleapis.com/youtube/v3"
TRANSLATE_API_URL = "https://translation.googleapis.com/language/translate/v2"
DATA_DIR = PROJECT_ROOT / "data" / "output"
USER_PROFILE_FILE = DATA_DIR / "user_profile.json"
YOUTUBE_PROFILE_FILE = DATA_DIR / "youtube_profile.json"
EXISTING_FEED_FILE = DATA_DIR / "personal_feed_items.json"
RAW_OUTPUT_FILE = DATA_DIR / "youtube_daily_items_raw.json"
OUTPUT_FILE = DATA_DIR / "youtube_daily_items.json"

DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_MAX_PER_CHANNEL = 3
DEFAULT_MAX_PER_QUERY = 4
DEFAULT_MAX_TOTAL_VIDEOS = 84
DEFAULT_MAX_SEARCH_QUERIES = 10


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def environment_int(name: str, default: int, minimum: int = 1) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer, not {raw_value!r}.") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def youtube_api_get(endpoint: str, params: dict[str, Any], api_key: str) -> dict[str, Any]:
    query = urlencode({**params, "key": api_key})
    request = Request(
        f"{API_BASE_URL}/{endpoint}?{query}",
        headers={"Accept": "application/json",
                 "User-Agent": "personalized-podcast-agent/1.0"},
    )
    try:
        with urlopen(request, timeout=30) as response:  # nosec B310: fixed HTTPS API base
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"YouTube API request to {endpoint} failed with HTTP {exc.code}: {details}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach the YouTube API: {exc.reason}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"YouTube API returned an invalid response for {endpoint}.")
    return payload


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def translate_daily_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_TRANSLATE_API_KEY is not set. Add it to .env before running the daily job."
        )

    fields = ("title", "description", "channel")
    texts = [
        normalize_text(item.get(field))
        for item in items
        for field in fields
        if normalize_text(item.get(field))
    ]
    if not texts:
        return items

    translated_values = []

    for start in range(0, len(texts), 128):
        batch = texts[start:start + 128]

        request = Request(
            f"{TRANSLATE_API_URL}?{urlencode({'key': api_key})}",
            data=json.dumps(
                {"q": batch, "target": "en", "format": "text"}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Google Translation API failed with HTTP {exc.code}: {details}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Could not reach the Google Translation API: {exc.reason}"
            ) from exc

        translations = payload.get("data", {}).get("translations", [])
        if len(translations) != len(batch):
            raise RuntimeError(
                "Google Translation API returned an incomplete response."
            )

        translated_values.extend(
            normalize_text(item.get("translatedText"))
            for item in translations
        )

    translated_texts = iter(translated_values)
    translated_items = []
    for item in items:
        translated_item = dict(item)
        for field in fields:
            value = normalize_text(item.get(field))
            translated_item[field] = next(translated_texts) if value else ""
        translated_item["creator"] = translated_item["channel"]
        translated_item["text"] = (
            translated_item["title"]
            if not translated_item["description"]
            else f"{translated_item['title']}\n\n{translated_item['description'][:1200]}"
        )
        translated_items.append(translated_item)

    return translated_items


def extract_channel_id(url: str) -> str | None:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if len(path_parts) >= 2 and path_parts[0] == "channel":
        return path_parts[1]
    return None


def extract_channel_handle(url: str) -> str | None:
    path_parts = [part for part in urlparse(url).path.split("/") if part]
    if path_parts and path_parts[0].startswith("@"):
        return path_parts[0][1:]
    return None


def load_subscription_channels() -> list[dict[str, str]]:
    """Read subscriptions already supplied through the user's Takeout data."""
    payload = read_json(EXISTING_FEED_FILE)
    channels: dict[str, dict[str, str]] = {}

    for item in payload.get("feed_items", []):
        if not isinstance(item, dict) or item.get("source_type") != "youtube_subscription":
            continue
        channel_name = normalize_text(item.get("creator") or item.get("text"))
        channel_url = normalize_text(item.get("url"))
        channel_id = extract_channel_id(channel_url)
        handle = extract_channel_handle(channel_url)
        key = channel_id or handle or channel_url or channel_name.casefold()
        if key:
            channels[key] = {
                "name": channel_name,
                "url": channel_url,
                "channel_id": channel_id or "",
                "handle": handle or "",
            }

    return list(channels.values())


def resolve_channel_id(channel: dict[str, str], api_key: str) -> str | None:
    existing_id = channel.get("channel_id")
    if existing_id:
        return existing_id

    handle = channel.get("handle")
    if handle:
        payload = youtube_api_get(
            "channels",
            {"part": "id", "forHandle": handle, "maxResults": 1},
            api_key,
        )
        items = payload.get("items", [])
        if items:
            return normalize_text(items[0].get("id")) or None

    # Older Takeout exports sometimes only contain a custom channel URL. A
    # name search is only a fallback, because it can be ambiguous.
    channel_name = channel.get("name")
    if channel_name:
        payload = youtube_api_get(
            "search",
            {"part": "snippet", "type": "channel",
                "q": channel_name, "maxResults": 1},
            api_key,
        )
        items = payload.get("items", [])
        if items:
            return normalize_text(items[0].get("id", {}).get("channelId")) or None

    return None


def profile_search_queries(max_queries: int) -> list[str]:
    user_profile = read_json(USER_PROFILE_FILE)
    youtube_profile = read_json(YOUTUBE_PROFILE_FILE)
    candidates: list[str] = []

    for field in ("podcast_focus_keywords", "focus_keywords", "preferred_topics"):
        values = user_profile.get(field, [])
        if isinstance(values, list):
            candidates.extend(normalize_text(value) for value in values)

    for field in ("recent_search_queries", "top_keywords", "recent_watch_titles"):
        values = youtube_profile.get(field, [])
        if isinstance(values, list):
            candidates.extend(normalize_text(value) for value in values)

    unique_queries: list[str] = []
    seen: set[str] = set()
    for query in candidates:
        query = re.sub(r"\s+", " ", query).strip()
        key = query.casefold()
        if len(query) < 3 or key in seen:
            continue
        unique_queries.append(query[:160])
        seen.add(key)
        if len(unique_queries) == max_queries:
            break
    return unique_queries


def video_to_feed_item(video: dict[str, Any], discovery: str) -> dict[str, Any] | None:
    video_id = normalize_text(video.get("id"))
    snippet = video.get("snippet", {})
    if not video_id or not isinstance(snippet, dict):
        return None

    title = normalize_text(snippet.get("title"))
    description = normalize_text(snippet.get("description"))
    channel_title = normalize_text(snippet.get("channelTitle"))
    published_at = normalize_text(snippet.get("publishedAt"))
    if not title:
        return None

    text = title if not description else f"{title}\n\n{description[:1200]}"
    return {
        "video_id": video_id,
        "source_type": "youtube_daily_video",
        "title": title,
        "description": description,
        "channel": channel_title,
        "published_at": published_at or None,
        "text": text,
        "creator": channel_title,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "occurred_at": published_at or None,
        "occurred_at_raw": published_at or None,
        "discovery": discovery,
    }


def fetch_channel_videos(
    channel_id: str,
    published_after: str,
    max_results: int,
    api_key: str,
) -> list[dict[str, Any]]:
    payload = youtube_api_get(
        "search",
        {
            "part": "snippet",
            "channelId": channel_id,
            "type": "video",
            "order": "date",
            "publishedAfter": published_after,
            "maxResults": max_results,
        },
        api_key,
    )
    videos = []
    for item in payload.get("items", []):
        video_id = normalize_text(item.get("id", {}).get("videoId"))
        if not video_id:
            continue
        videos.append({"id": video_id, "snippet": item.get("snippet", {})})
    return videos


def fetch_query_videos(
    query: str,
    published_after: str,
    max_results: int,
    api_key: str,
) -> list[dict[str, Any]]:
    payload = youtube_api_get(
        "search",
        {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": "date",
            "publishedAfter": published_after,
            "maxResults": max_results,
        },
        api_key,
    )
    videos = []
    for item in payload.get("items", []):
        video_id = normalize_text(item.get("id", {}).get("videoId"))
        if not video_id:
            continue
        videos.append({"id": video_id, "snippet": item.get("snippet", {})})
    return videos


def save_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, output_path)


def fetch_daily_youtube_items() -> dict[str, Any]:
    current_day = datetime.now(UTC).date().isoformat()
    cached_result = read_json(OUTPUT_FILE)
    cached_generated_at = str(
        cached_result.get("metadata", {}).get("generated_at", "")
    )
    force_refresh = os.getenv("YOUTUBE_FORCE_REFRESH", "").strip() == "1"

    if (
        not force_refresh
        and cached_generated_at.startswith(current_day)
        and isinstance(cached_result.get("feed_items"), list)
    ):
        print(
            f"Using today's cached YouTube items: {OUTPUT_FILE}",
            flush=True,
        )
        return cached_result

    raw_result = read_json(RAW_OUTPUT_FILE)
    raw_generated_at = str(
        raw_result.get("metadata", {}).get("generated_at", "")
    )
    if (
        not force_refresh
        and raw_generated_at.startswith(current_day)
        and isinstance(raw_result.get("feed_items"), list)
    ):
        print(
            f"Translating today's cached YouTube items: {RAW_OUTPUT_FILE}",
            flush=True,
        )
        translated_result = dict(raw_result)
        translated_result["feed_items"] = translate_daily_items(
            raw_result["feed_items"]
        )
        save_json_atomic(translated_result, OUTPUT_FILE)
        return translated_result

    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY is not set. Add it to your environment before running the daily job."
        )

    lookback_days = environment_int(
        "YOUTUBE_LOOKBACK_DAYS", DEFAULT_LOOKBACK_DAYS)
    max_per_channel = environment_int(
        "YOUTUBE_MAX_PER_CHANNEL", DEFAULT_MAX_PER_CHANNEL)
    max_per_query = environment_int(
        "YOUTUBE_MAX_PER_QUERY", DEFAULT_MAX_PER_QUERY)
    max_total = environment_int(
        "YOUTUBE_MAX_TOTAL_VIDEOS", DEFAULT_MAX_TOTAL_VIDEOS)
    max_queries = environment_int(
        "YOUTUBE_MAX_SEARCH_QUERIES", DEFAULT_MAX_SEARCH_QUERIES)

    now = datetime.now(UTC)
    published_after = (now - timedelta(days=lookback_days)
                       ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    collected: dict[str, dict[str, Any]] = {}
    resolved_channels = 0

    for channel in load_subscription_channels():
        if len(collected) >= max_total:
            break
        channel_id = resolve_channel_id(channel, api_key)
        if not channel_id:
            continue
        resolved_channels += 1
        for video in fetch_channel_videos(channel_id, published_after, max_per_channel, api_key):
            item = video_to_feed_item(video, discovery="subscription")
            if item:
                collected.setdefault(item["video_id"], item)
            if len(collected) >= max_total:
                break

    queries = profile_search_queries(max_queries)
    for query in queries:
        if len(collected) >= max_total:
            break
        for video in fetch_query_videos(query, published_after, max_per_query, api_key):
            item = video_to_feed_item(
                video, discovery=f"profile_query:{query}")
            if item:
                collected.setdefault(item["video_id"], item)
            if len(collected) >= max_total:
                break

    feed_items = sorted(
        collected.values(),
        key=lambda item: str(item.get("occurred_at") or ""),
        reverse=True,
    )
    raw_result = {
        "metadata": {
            "schema_version": 1,
            "generated_at": now.isoformat(),
            "lookback_days": lookback_days,
            "published_after": published_after,
            "subscription_channels_resolved": resolved_channels,
            "profile_queries_used": queries,
            "total_feed_items": len(feed_items),
            "privacy": "Public video metadata only; private watch history is not collected.",
        },
        "feed_items": feed_items,
    }
    save_json_atomic(raw_result, RAW_OUTPUT_FILE)

    translated_result = dict(raw_result)
    translated_result["feed_items"] = translate_daily_items(feed_items)
    return translated_result


def main() -> None:
    result = fetch_daily_youtube_items()
    save_json_atomic(result, OUTPUT_FILE)
    print(
        f"Saved {len(result['feed_items'])} daily YouTube videos to {OUTPUT_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
