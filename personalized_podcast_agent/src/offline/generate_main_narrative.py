

import json
import os
import re
import time
from typing import Any

from src.offline.search_web_sources import search_web_source
from src.utils import PROJECT_ROOT


TOPIC_COUNT = 10
TRUNKS_PER_TOPIC = 5
DATA_DIR = PROJECT_ROOT / "data" / "output"
MAIN_NARRATIVES_FILE = DATA_DIR / "main_narratives.json"
DAILY_YOUTUBE_FILE = DATA_DIR / "youtube_daily_items.json"


def save_main_narratives(result: dict[str, Any]) -> None:
    MAIN_NARRATIVES_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = MAIN_NARRATIVES_FILE.with_name(
        f"{MAIN_NARRATIVES_FILE.stem}.{os.getpid()}.tmp"
    )
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    for attempt in range(5):
        try:
            os.replace(temporary_file, MAIN_NARRATIVES_FILE)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def load_saved_transcript_sources() -> dict[str, dict[str, Any]]:
    if not DAILY_YOUTUBE_FILE.exists():
        raise FileNotFoundError(
            "Saved YouTube videos are missing. Run the YouTube importer first."
        )

    with DAILY_YOUTUBE_FILE.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    sources: dict[str, dict[str, Any]] = {}
    for item in payload.get("feed_items", []):
        if not isinstance(item, dict):
            continue

        url = str(item.get("url") or "").strip()
        transcript_text = str(item.get("transcript_text") or "").strip()
        if not url or not transcript_text:
            continue

        sources[url] = {
            "source_type": "youtube_transcript",
            "title": str(item.get("title") or "").strip(),
            "url": url,
            "published_at": item.get("occurred_at"),
            "transcript_text": transcript_text,
            "article_text": transcript_text,
            "transcript_word_count": len(transcript_text.split()),
            "article_word_count": len(transcript_text.split()),
            "language": str(item.get("transcript_language") or ""),
            "language_code": str(item.get("transcript_language_code") or ""),
            "is_generated": bool(item.get("transcript_is_generated")),
        }

    return sources


def split_source_text(source_text: str) -> list[str]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", source_text)
        if sentence.strip()
    ]
    if len(sentences) < TRUNKS_PER_TOPIC:
        words = source_text.split()
        sentences = [
            " ".join(
                words[
                    round(order * len(words) / TRUNKS_PER_TOPIC):
                    round((order + 1) * len(words) / TRUNKS_PER_TOPIC)
                ]
            )
            for order in range(TRUNKS_PER_TOPIC)
        ]

    parts: list[str] = []
    for order in range(TRUNKS_PER_TOPIC):
        start = round(order * len(sentences) / TRUNKS_PER_TOPIC)
        end = round((order + 1) * len(sentences) / TRUNKS_PER_TOPIC)
        parts.append(" ".join(sentences[start:end]).strip())

    if any(not part for part in parts):
        raise ValueError("The source text could not be split into five parts.")
    return parts


def source_key(source_item: dict[str, Any]) -> str:
    return str(source_item.get("url") or source_item.get("title") or "").casefold()


def build_main_narrative(
    *,
    selected_seed: dict[str, Any],
    transcript_sources: dict[str, dict[str, Any]],
    used_source_keys: set[str],
) -> dict[str, Any] | None:
    source_item = transcript_sources.get(
        str(selected_seed.get("url") or "").strip())
    if source_item is None:
        source_item = search_web_source(selected_seed)
    if source_item is None or source_key(source_item) in used_source_keys:
        return None

    story_title = str(source_item.get("title") or "").strip()
    source_parts = split_source_text(str(source_item["article_text"]))
    used_source_keys.add(source_key(source_item))
    return {
        "story_title": story_title,
        "selected_seed": selected_seed,
        "steps": [
            {
                "chunk_order": order,
                "focus": f"Source segment {order}",
                "source_segment": source_part,
                "search_query": source_item.get("search_query"),
                "source_item": source_item,
            }
            for order, source_part in enumerate(source_parts, start=1)
        ],
    }


def generate_main_narratives(
    story_seed_pool: dict[str, Any],
) -> dict[str, Any]:
    stories: list[dict[str, Any]] = []
    used_source_keys: set[str] = set()
    candidates = story_seed_pool.get("candidates", [])
    transcript_sources = load_saved_transcript_sources()

    for candidate_number, candidate_data in enumerate(candidates, start=1):
        print(
            f"[{candidate_number}/{len(candidates)}] Building source story.",
            flush=True,
        )
        try:
            narrative = build_main_narrative(
                selected_seed=candidate_data["selected_seed"],
                transcript_sources=transcript_sources,
                used_source_keys=used_source_keys,
            )
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Skipping candidate: {exc}", flush=True)
            continue

        if narrative is None:
            continue
        narrative["story_id"] = f"story_{len(stories) + 1:02d}"
        stories.append(narrative)
        print(
            f"[{len(stories)}/{len(candidates)}] Found story: "
            f"{narrative['story_title']}",
            flush=True,
        )
        if len(stories) >= TOPIC_COUNT:
            break

    return {
        "configuration": {
            "story_count": len(stories),
            "trunks_per_story": TRUNKS_PER_TOPIC,
        },
        "stories": stories,
    }


def main() -> None:
    from src.user_profile.select_feed_seeds import (
        save_feed_seeds,
        select_story_seeds,
    )

    story_seed_pool = select_story_seeds()
    save_feed_seeds(story_seed_pool)
    result = generate_main_narratives(story_seed_pool)
    if not result["stories"]:
        raise ValueError(
            "Could not build a story from the selected seeds.")
    save_main_narratives(result)
    print(
        f"Saved {len(result['stories'])} main narratives to: "
        f"{MAIN_NARRATIVES_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
