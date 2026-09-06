from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from src.utils import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data" / "output"
DAILY_YOUTUBE_FILE = DATA_DIR / "youtube_daily_items.json"
OUTPUT_FILE = DATA_DIR / "personal_feed_items.json"


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def make_feed_id(video_id: str, url: str) -> str:
    digest = hashlib.sha256(
        f"{video_id}|{url}".encode("utf-8")).hexdigest()[:16]
    return f"youtube_daily_video_{digest}"


def build_feed_items() -> dict[str, Any]:
    with DAILY_YOUTUBE_FILE.open("r", encoding="utf-8") as file:
        daily_data = json.load(file)

    source_items = daily_data.get("feed_items", [])
    if not isinstance(source_items, list):
        raise ValueError(
            "youtube_daily_items.json must contain a feed_items array.")

    feed_items: list[dict[str, Any]] = []
    for fallback_rank, source_item in enumerate(source_items, start=1):
        if not isinstance(source_item, dict):
            continue

        video_id = clean_text(source_item.get("video_id"))
        url = clean_text(source_item.get("url"))
        title = clean_text(source_item.get("title"))
        description = clean_text(source_item.get("description"))
        channel = clean_text(source_item.get("channel")
                             or source_item.get("creator"))
        if not title or not (video_id or url):
            continue

        try:
            homepage_rank = int(source_item.get(
                "homepage_rank", fallback_rank))
        except (TypeError, ValueError):
            homepage_rank = fallback_rank

        text = title if not description else f"{title}\n\n{description}"
        feed_items.append(
            {
                "feed_id": make_feed_id(video_id, url),
                "source_type": "youtube_daily_video",
                "title": title,
                "description": description,
                "channel": channel or None,
                "creator": channel or None,
                "text": text,
                "url": url or None,
                "occurred_at": source_item.get("occurred_at"),
                "occurred_at_raw": source_item.get("occurred_at_raw"),
                "homepage_rank": homepage_rank,
                "interaction_count": 1,
            }
        )

    feed_items.sort(key=lambda item: int(item["homepage_rank"]))
    return {
        "metadata": {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "total_feed_items": len(feed_items),
            "source_counts": {"youtube_daily_video": len(feed_items)},
        },
        "feed_items": feed_items,
    }


def main() -> None:
    result = build_feed_items()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(
        f"Saved {len(result['feed_items'])} YouTube homepage videos to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
