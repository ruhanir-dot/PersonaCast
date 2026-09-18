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

FINE_COSINE_THRESHOLD = 0.60
TOPIC_COSINE_THRESHOLD = 0.60
MAX_TOPIC_SIZE = 15
MAX_NEIGHBORS = 5
MIN_FEEDS = 3
MAX_FEEDS = 15


def load_embedding_data() -> tuple[list[dict[str, Any]], np.ndarray, str]:
    with FEED_ITEMS_FILE.open("r", encoding="utf-8") as file:
        feed_data = json.load(file)
    with FEED_IDS_FILE.open("r", encoding="utf-8") as file:
        embedding_index = json.load(file)

    embeddings = np.load(FEED_EMBEDDINGS_FILE, allow_pickle=False)
    feed_ids = embedding_index["feed_ids"]

    if len(feed_ids) != len(embeddings):
        raise ValueError("Feed IDs and embeddings have different lengths.")

    items_by_id = {
        str(item["feed_id"]): item
        for item in feed_data["feed_items"]
    }

    items = []
    for feed_id in feed_ids:
        item = items_by_id.get(str(feed_id))
        if item is None:
            raise ValueError(f"Feed ID {feed_id} is missing from feed items.")
        items.append(item)

    return (
        items,
        embeddings,
        str(embedding_index["metadata"]["model_name"]),
    )


def build_groups(
    embeddings: np.ndarray,
    cosine_threshold: float,
) -> list[list[int]]:
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -1.0)
    graph = [set() for _ in range(len(embeddings))]

    for index in range(len(embeddings)):
        candidates = np.where(
            similarity[index] >= cosine_threshold
        )[0]
        candidates = candidates[
            np.argsort(similarity[index, candidates])[::-1]
        ][:MAX_NEIGHBORS]

        for neighbor in candidates:
            graph[index].add(int(neighbor))
            graph[int(neighbor)].add(index)

    groups = []
    visited = set()

    for start in range(len(graph)):
        if start in visited:
            continue

        stack = [start]
        visited.add(start)
        group = []

        while stack:
            current = stack.pop()
            group.append(current)

            for neighbor in graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        if group:
            groups.append(group)

    return groups


def group_centroid(
    group: list[int],
    embeddings: np.ndarray,
) -> np.ndarray:
    centroid = embeddings[group].mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm else centroid


def merge_groups(
    groups: list[list[int]],
    embeddings: np.ndarray,
) -> list[list[int]]:
    merged_groups = [list(group) for group in groups]

    while True:
        best_pair = None
        best_score = TOPIC_COSINE_THRESHOLD

        for left in range(len(merged_groups)):
            for right in range(left + 1, len(merged_groups)):
                left_group = merged_groups[left]
                right_group = merged_groups[right]

                if len(left_group) + len(right_group) > MAX_TOPIC_SIZE:
                    continue

                cross_similarity = (
                    embeddings[left_group] @ embeddings[right_group].T
                )

                score = float(cross_similarity.mean())

                if score > best_score:
                    best_score = score
                    best_pair = (left, right)

        if best_pair is None:
            break

        left, right = best_pair
        merged_groups[left].extend(merged_groups[right])
        del merged_groups[right]

    return merged_groups


def feed_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "feed_id": item["feed_id"],
        "source_type": item.get("source_type"),
        "title": " ".join(
            str(
                item.get("title")
                or item.get("caption")
                or item.get("query")
                or ""
            ).split()
        ),
        "description": " ".join(
            str(
                item.get("description")
                or item.get("text")
                or ""
            ).split()
        ),
        "channel": " ".join(
            str(
                item.get("channel")
                or item.get("creator")
                or ""
            ).split()
        ),
        "url": item.get("url"),
    }


def select_story_seeds() -> dict[str, Any]:
    feed_items, embeddings, model_name = load_embedding_data()
    groups = build_groups(embeddings, FINE_COSINE_THRESHOLD)
    groups = merge_groups(groups, embeddings)
    groups = [
        group
        for group in groups
        if len(group) >= MIN_FEEDS
    ]
    candidates = []

    for rank, group in enumerate(groups, start=1):
        center = max(
            group,
            key=lambda index: float(
                (embeddings[index] @ embeddings[group].T).mean()
            ),
        )

        selected = sorted(
            group,
            key=lambda index: float(
                embeddings[center] @ embeddings[index]
            ),
            reverse=True,
        )[:MAX_FEEDS]

        candidates.append(
            {
                "candidate_rank": rank,
                "feed_ids": [feed_items[index]["feed_id"] for index in selected],
                "feed_count": len(selected),
                "selected_seed": feed_summary(feed_items[center]),
                "feeds": [feed_summary(feed_items[index]) for index in selected],
            }
        )

    return {
        "configuration": {
            "group_count": len(candidates),
            "min_feeds": MIN_FEEDS,
            "max_feeds": MAX_FEEDS,
            "cosine_threshold": FINE_COSINE_THRESHOLD,
            "topic_cosine_threshold": TOPIC_COSINE_THRESHOLD,
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


if __name__ == "__main__":
    save_feed_seeds(select_story_seeds())
