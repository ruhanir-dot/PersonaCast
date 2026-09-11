from __future__ import annotations

import json
import os
import time
from typing import Any

import numpy as np

from src.utils import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data" / "output"
FEED_ITEMS_FILE = DATA_DIR / "personal_feed_items.json"
FEED_EMBEDDINGS_FILE = DATA_DIR / "personal_feed_embeddings.npy"
FEED_IDS_FILE = DATA_DIR / "personal_feed_embedding_ids.json"
OUTPUT_FILE = DATA_DIR / "topic_feed_seeds.json"

STORY_CANDIDATE_COUNT = 10
MMR_LAMBDA = 0.60


def load_embedding_data() -> tuple[list[dict[str, Any]], np.ndarray, str]:
    with FEED_ITEMS_FILE.open("r", encoding="utf-8") as file:
        feed_data = json.load(file)
    with FEED_IDS_FILE.open("r", encoding="utf-8") as file:
        embedding_index = json.load(file)

    embeddings = np.load(FEED_EMBEDDINGS_FILE, allow_pickle=False)
    feed_ids = embedding_index["feed_ids"]
    if len(feed_ids) != len(embeddings):
        raise ValueError(
            "The number of feed IDs does not match the embeddings.")

    items_by_id = {
        str(item["feed_id"]): item for item in feed_data["feed_items"]
    }
    ordered_items = []
    for feed_id in feed_ids:
        item = items_by_id.get(str(feed_id))
        if item is None:
            raise ValueError(
                f"Feed ID {feed_id} is missing from the feed items.")
        ordered_items.append(item)

    return (
        ordered_items,
        embeddings,
        str(embedding_index["metadata"]["model_name"]),
    )


def select_with_mmr(
    relevance_scores: np.ndarray,
    feed_embeddings: np.ndarray,
    candidate_indices: list[int],
    selection_count: int,
) -> list[tuple[int, float]]:
    selected: list[int] = []
    ranked: list[tuple[int, float]] = []

    while len(selected) < selection_count:
        best_index: int | None = None
        best_score = float("-inf")

        for candidate_index in candidate_indices:
            if candidate_index in selected:
                continue

            redundancy = 0.0
            if selected:
                redundancy = float(
                    np.max(
                        feed_embeddings[selected]
                        @ feed_embeddings[candidate_index]
                    )
                )

            score = (
                MMR_LAMBDA * float(relevance_scores[candidate_index])
                - (1.0 - MMR_LAMBDA) * redundancy
            )
            if score > best_score:
                best_index = candidate_index
                best_score = score

        if best_index is None:
            break

        selected.append(best_index)
        ranked.append((best_index, best_score))

    return ranked


def select_story_seeds() -> dict[str, Any]:
    """Use homepage order and MMR to choose ten diverse YouTube videos."""
    feed_items, feed_embeddings, model_name = load_embedding_data()
    daily_indices = sorted(
        [
            index
            for index, item in enumerate(feed_items)
            if item.get("source_type") == "youtube_daily_video"
        ],
        key=lambda index: int(feed_items[index].get("homepage_rank") or 10**9),
    )
    if not daily_indices:
        raise ValueError("No daily YouTube videos are available.")

    relevance_scores = np.zeros(len(feed_embeddings), dtype=np.float32)
    homepage_scores = np.linspace(1.0, 0.5, len(daily_indices))
    relevance_scores[daily_indices] = homepage_scores
    selected = select_with_mmr(
        relevance_scores=relevance_scores,
        feed_embeddings=feed_embeddings,
        candidate_indices=daily_indices,
        selection_count=min(STORY_CANDIDATE_COUNT, len(daily_indices)),
    )

    candidates = []
    for rank, (feed_index, mmr_score) in enumerate(selected, start=1):
        item = feed_items[feed_index]
        candidates.append(
            {
                "candidate_rank": rank,
                "selected_seed": {
                    "feed_id": item["feed_id"],
                    "source_type": item["source_type"],
                    "title": " ".join(str(item.get("title") or "").split()),
                    "description": " ".join(
                        str(item.get("description") or "").split()
                    ),
                    "channel": " ".join(
                        str(item.get("channel") or item.get(
                            "creator") or "").split()
                    ),
                    "url": item.get("url"),
                    "occurred_at": item.get("occurred_at"),
                    "homepage_rank": item.get("homepage_rank"),
                    "mmr_score": round(float(mmr_score), 6),
                },
            }
        )

    return {
        "configuration": {
            "candidate_seed_count": len(candidates),
            "mmr_lambda": MMR_LAMBDA,
            "embedding_model": model_name,
        },
        "candidates": candidates,
    }


def save_feed_seeds(result: dict[str, Any]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.{os.getpid()}.tmp"
    )
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    for attempt in range(5):
        try:
            os.replace(temporary_file, OUTPUT_FILE)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)
