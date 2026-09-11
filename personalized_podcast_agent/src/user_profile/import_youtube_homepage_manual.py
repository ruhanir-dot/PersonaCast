import json
import os
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

from src.user_profile.fetch_youtube_daily import (
    DATA_DIR,
    OUTPUT_FILE,
    RAW_OUTPUT_FILE,
    save_json_atomic,
    translate_daily_items,
    video_to_feed_item,
    youtube_api_get,
)

MANUAL_INPUT_FILE = DATA_DIR / "youtube_homepage_manual.json"
VIDEOS_PER_REQUEST = 50
MIN_TRANSCRIPT_CHARACTERS = 1200


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def video_id_from_url(url: str) -> str:
    parsed = urlparse(normalize_text(url))
    return normalize_text(parse_qs(parsed.query).get("v", [""])[0])


def load_manual_video_ids() -> list[str]:
    if not MANUAL_INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Manual homepage file was not found: {MANUAL_INPUT_FILE}"
        )

    with MANUAL_INPUT_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON array in {MANUAL_INPUT_FILE}.")

    video_ids: list[str] = []
    seen: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        video_id = video_id_from_url(item.get("url", ""))
        if video_id and video_id not in seen:
            video_ids.append(video_id)
            seen.add(video_id)

    if not video_ids:
        raise ValueError(
            f"No valid YouTube video URLs were found in {MANUAL_INPUT_FILE}.")
    return video_ids


def fetch_video_metadata(video_ids: list[str], api_key: str) -> dict[str, dict[str, Any]]:
    videos: dict[str, dict[str, Any]] = {}
    for start in range(0, len(video_ids), VIDEOS_PER_REQUEST):
        batch = video_ids[start:start + VIDEOS_PER_REQUEST]
        payload = youtube_api_get(
            "videos",
            {"part": "snippet", "id": ",".join(batch)},
            api_key,
        )
        for video in payload.get("items", []):
            if not isinstance(video, dict):
                continue
            video_id = normalize_text(video.get("id"))
            snippet = video.get("snippet")
            if video_id and isinstance(snippet, dict):
                videos[video_id] = video
    return videos


def fetch_transcript(video_id: str) -> dict[str, Any] | None:
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        try:
            transcript = transcript_list.find_transcript(
                ["en", "en-US", "zh-TW", "zh", "ko", "ja"]
            )
        except Exception:
            transcript = next(iter(transcript_list))
        fetched_transcript = transcript.fetch()
    except Exception as exc:
        print(
            f"No transcript for {video_id}; storing null ({exc})", flush=True)
        return None

    transcript_text = normalize_text(
        " ".join(
            str(snippet.text).strip()
            for snippet in fetched_transcript
            if str(snippet.text).strip()
        )
    )
    spoken_text = re.sub(r"\[[^\]]*\]|\([^)]*\)|[♪♫]", " ", transcript_text)
    spoken_text = normalize_text(spoken_text)
    if len(spoken_text) < MIN_TRANSCRIPT_CHARACTERS:
        print(f"Transcript too short for {video_id}; storing null", flush=True)
        return None

    return {
        "transcript_text": transcript_text,
        "transcript_character_count": len(spoken_text),
        "transcript_language": str(fetched_transcript.language),
        "transcript_language_code": str(fetched_transcript.language_code),
        "transcript_is_generated": bool(fetched_transcript.is_generated),
    }


def import_manual_homepage_items() -> dict[str, Any]:
    api_key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "YOUTUBE_API_KEY is not set. Add it to your environment before running the importer."
        )

    video_ids = load_manual_video_ids()
    metadata_by_id = fetch_video_metadata(video_ids, api_key)

    feed_items: list[dict[str, Any]] = []
    for homepage_rank, video_id in enumerate(video_ids, start=1):
        video = metadata_by_id.get(video_id)
        if not video:
            continue
        item = video_to_feed_item(video, discovery="youtube_homepage_manual")
        if not item:
            continue
        transcript = fetch_transcript(video_id)
        item["homepage_rank"] = homepage_rank
        if transcript is None:
            item.update(
                {
                    "text": None,
                    "transcript_text": None,
                    "transcript_character_count": None,
                    "transcript_language": None,
                    "transcript_language_code": None,
                    "transcript_is_generated": None,
                }
            )
        else:
            item.update(transcript)
        feed_items.append(item)

    if not feed_items:
        raise RuntimeError(
            "YouTube did not return metadata for the manual homepage videos.")

    raw_result = {
        "metadata": {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "manual_video_count": len(video_ids),
            "total_feed_items": len(feed_items),
            "privacy": "Public metadata and available transcripts for YouTube videos manually selected from the homepage.",
        },
        "feed_items": feed_items,
    }
    save_json_atomic(raw_result, RAW_OUTPUT_FILE)

    translated_result = dict(raw_result)
    translated_result["feed_items"] = translate_daily_items(feed_items)
    return translated_result


def main() -> None:
    result = import_manual_homepage_items()
    save_json_atomic(result, OUTPUT_FILE)
    print(
        f"Saved {len(result['feed_items'])} manual YouTube homepage videos to {OUTPUT_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
