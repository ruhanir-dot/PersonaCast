import json
import math
import os
import re
import time
from typing import Any

from src.utils import PROJECT_ROOT, llm_json


TOPIC_COUNT = 10
WORDS_PER_TRUNK = 40
MAX_TRUNKS_PER_TOPIC = 15
DATA_DIR = PROJECT_ROOT / "data" / "output"
MAIN_NARRATIVES_FILE = DATA_DIR / "main_narratives.json"
FEED_ITEMS_FILE = DATA_DIR / "personal_feed_items.json"


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


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_feed_items() -> list[dict[str, Any]]:
    with FEED_ITEMS_FILE.open("r", encoding="utf-8") as file:
        feed_data = json.load(file)
    return feed_data["feed_items"]


def feed_to_source_item(item: dict[str, Any]) -> dict[str, Any] | None:
    source_type = str(item.get("source_type") or "")
    title = normalize_text(
        item.get("title")
        or item.get("caption")
        or item.get("query")
    )
    raw_text = str(item.get("text") or "").strip()

    if source_type == "google_news_article":
        text = "\n\n".join(
            normalize_text(paragraph)
            for paragraph in re.split(r"\n\s*\n|\r?\n", raw_text)
            if paragraph.strip()
        )
    else:
        text = normalize_text(raw_text)

    if not text:
        text = normalize_text(
            " ".join(
                value for value in [
                    item.get("description"),
                    item.get("transcript"),
                    item.get("caption"),
                    item.get("query"),
                ]
                if value
            )
        )

    if not title and not text:
        return None

    return {
        "feed_id": item.get("feed_id"),
        "source_type": source_type,
        "title": title,
        "url": item.get("url"),
        "creator": item.get("creator") or item.get("channel"),
        "search_query": item.get("query"),
        "article_text": text or title,
    }


def generate_story_title(
    *,
    selected_seed: dict[str, Any],
    source_items: list[dict[str, Any]],
) -> str | None:
    if not source_items:
        return None

    source_content = "\n\n".join(
        f"{item.get('title', '')}\n{item.get('article_text', '')[:600]}"
        for item in source_items
    )

    prompt = f"""
Create one fluent English podcast story title based on the related sources below.

Make the title clear and related to the source content.
Return JSON only in this format:
{{"story_title": "..."}}.

Selected topic:
{selected_seed.get('title', '')}

Related sources:
{source_content[:5000]}
""".strip()

    result = llm_json(
        prompt,
        system="You create clear and fluent podcast story titles.",
    )

    title = str(result.get("story_title") or "").strip()
    title = re.sub(r"\s+", " ", title)
    return title or normalize_text(selected_seed.get("title")) or None


def split_source_text(source_text: str, source_type: str) -> list[str]:
    source_text = source_text.strip()

    if source_type in {"instagram_post", "instagram_reel"}:
        return [normalize_text(source_text)]

    if source_type == "google_news_article":
        paragraphs = [
            normalize_text(paragraph)
            for paragraph in re.split(r"\n\s*\n|\r?\n", source_text)
            if paragraph.strip()
        ]

        if len(paragraphs) <= 1:
            paragraphs = [
                sentence.strip()
                for sentence in re.split(
                    r"(?<=[.!?。！？])\s*",
                    source_text,
                )
                if sentence.strip()
            ]

        return paragraphs or [normalize_text(source_text)]

    words = source_text.split()

    if len(words) <= 150:
        return [source_text]

    trunk_count = max(1, math.ceil(len(words) / WORDS_PER_TRUNK))
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?。！？])\s*", source_text)
        if sentence.strip()
    ]

    if len(sentences) < trunk_count:
        sentences = [
            " ".join(
                words[
                    round(order * len(words) / trunk_count):
                    round((order + 1) * len(words) / trunk_count)
                ]
            )
            for order in range(trunk_count)
        ]

    parts: list[str] = []
    for order in range(trunk_count):
        start = round(order * len(sentences) / trunk_count)
        end = round((order + 1) * len(sentences) / trunk_count)
        parts.append(" ".join(sentences[start:end]).strip())

    if any(not part for part in parts):
        raise ValueError("The source text could not be split into trunks.")
    return parts


def allocate_trunk_budgets(part_counts: list[int]) -> list[int]:
    budgets = [1 for _ in part_counts]
    remaining = MAX_TRUNKS_PER_TOPIC - len(budgets)

    while remaining > 0:
        available = [
            index
            for index, count in enumerate(part_counts)
            if budgets[index] < count
        ]
        if not available:
            break

        index = max(
            available,
            key=lambda value: part_counts[value] / budgets[value],
        )
        budgets[index] += 1
        remaining -= 1

    return budgets


def merge_parts(parts: list[str], count: int) -> list[str]:
    if len(parts) <= count:
        return parts

    return [
        " ".join(
            parts[
                round(order * len(parts) / count):
                round((order + 1) * len(parts) / count)
            ]
        ).strip()
        for order in range(count)
    ]


def find_related_sources(
    feed_ids: list[str],
    feed_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    feed_ids = {str(feed_id) for feed_id in feed_ids}
    related_sources: list[dict[str, Any]] = []

    for item in feed_items:
        if str(item.get("feed_id")) not in feed_ids:
            continue

        source_item = feed_to_source_item(item)
        if source_item is None:
            continue

        related_sources.append(source_item)

    return related_sources


def build_main_narrative(
    *,
    selected_seed: dict[str, Any],
    feed_ids: list[str],
    feed_items: list[dict[str, Any]],
) -> dict[str, Any] | None:
    source_items = find_related_sources(
        feed_ids=feed_ids,
        feed_items=feed_items,
    )
    if not source_items:
        return None

    story_title = generate_story_title(
        selected_seed=selected_seed,
        source_items=source_items,
    )
    if story_title is None:
        return None

    source_parts_by_item = [
        split_source_text(
            str(source_item["article_text"]),
            str(source_item.get("source_type") or ""),
        )
        for source_item in source_items
    ]
    budgets = allocate_trunk_budgets(
        [len(parts) for parts in source_parts_by_item]
    )

    steps: list[dict[str, Any]] = []
    for source_item, source_parts, budget in zip(
        source_items,
        source_parts_by_item,
        budgets,
    ):
        source_parts = merge_parts(source_parts, budget)
        for part_number, source_part in enumerate(source_parts, start=1):
            steps.append(
                {
                    "chunk_order": len(steps) + 1,
                    "focus": (
                        f"{source_item['source_type']} "
                        f"trunk {part_number}"
                    ),
                    "source_segment": source_part,
                    "search_query": source_item.get("search_query"),
                    "source_item": source_item,
                }
            )

    return {
        "story_title": story_title,
        "selected_seed": selected_seed,
        "steps": steps,
    }


def generate_main_narratives(
    story_seed_pool: dict[str, Any],
) -> dict[str, Any]:
    stories: list[dict[str, Any]] = []
    candidates = story_seed_pool.get("candidates", [])
    feed_items = load_feed_items()

    for candidate_number, candidate_data in enumerate(candidates, start=1):
        print(
            f"[{candidate_number}/{len(candidates)}] Building source story.",
            flush=True,
        )
        try:
            narrative = build_main_narrative(
                selected_seed=candidate_data["selected_seed"],
                feed_ids=candidate_data["feed_ids"],
                feed_items=feed_items,
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
            "trunks_per_story": "dynamic",
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
            "Could not build a story from the selected seeds."
        )

    save_main_narratives(result)

    print(
        f"Saved {len(result['stories'])} main narratives to: "
        f"{MAIN_NARRATIVES_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
