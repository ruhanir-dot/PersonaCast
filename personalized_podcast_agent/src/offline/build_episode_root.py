"""Add one episode-level root above the topic hierarchy.

Run after ``merge_topic_hierarchy.py``:

    python -m src.offline.build_episode_root

The episode root keeps story-cluster and topic-cluster roots as children,
excluding unmerged singleton feed roots. Its short spoken overview highlights
four to six topic roots; other included clusters remain available for later
selection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import llm_json  # noqa: E402

DATA_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_INPUT = DATA_DIR / "topic_hierarchy.json"
DEFAULT_OUTPUT = DATA_DIR / "episode_hierarchy.json"
MIN_OVERVIEW_WORDS = 100
MAX_OVERVIEW_WORDS = 150
MIN_SELECTED_TOPICS = 4
MAX_SELECTED_TOPICS = 6


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary_path, path)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[A-Za-z0-9]+(?:[’'-][A-Za-z0-9]+)*\b", text))


def index_nodes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = normalize_text(node.get("trunk_id"))
        if not node_id:
            raise ValueError("Every hierarchy node must have a trunk_id.")
        if node_id in nodes:
            raise ValueError(f"Duplicate hierarchy node id: {node_id}")
        nodes[node_id] = node
    if not nodes:
        raise ValueError("The topic hierarchy contains no nodes.")
    return nodes


def get_active_roots(
    payload: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> list[str]:
    roots = [normalize_text(root) for root in payload.get("roots", [])]
    if not roots:
        raise ValueError("The topic hierarchy contains no active roots.")
    missing = [root for root in roots if root not in nodes]
    if missing:
        raise ValueError(f"Active roots missing from nodes: {missing[:5]}")
    return roots


def get_topic_roots(
    roots: list[str], nodes: dict[str, dict[str, Any]]
) -> list[str]:
    topic_roots = [
        root
        for root in roots
        if nodes[root].get("node_type") == "topic_merge_parent_trunk"
    ]
    if not topic_roots:
        raise ValueError("No topic merge roots were found in the active roots.")
    return topic_roots


def get_episode_child_roots(
    roots: list[str], nodes: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    """Keep story and topic clusters; exclude singleton feed roots."""
    included: list[str] = []
    excluded_singletons: list[str] = []
    for root in roots:
        node_type = nodes[root].get("node_type")
        if node_type == "feed_leaf_trunk":
            excluded_singletons.append(root)
        elif node_type in {"merge_parent_trunk", "topic_merge_parent_trunk"}:
            included.append(root)
        else:
            raise ValueError(
                f"Unsupported active root type for {root}: {node_type!r}"
            )
    if not included:
        raise ValueError("No story or topic cluster roots remain for the episode.")
    return included, excluded_singletons


def reachable_node_ids(
    start_ids: list[str], nodes: dict[str, dict[str, Any]]
) -> set[str]:
    """Return descendants of selected roots, excluding unrelated components."""
    reachable: set[str] = set()
    pending = list(start_ids)
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        if node_id not in nodes:
            raise ValueError(f"Hierarchy child is missing from nodes: {node_id}")
        reachable.add(node_id)
        children = nodes[node_id].get("children") or []
        if not isinstance(children, list):
            raise ValueError(f"Node children must be a list: {node_id}")
        pending.extend(str(child) for child in children)
    return reachable


def topic_candidate(node_id: str, node: dict[str, Any]) -> dict[str, str]:
    return {
        "topic_id": node_id,
        "topic": normalize_text(node.get("topic_label") or node.get("title"))
        or node_id,
        "summary": normalize_text(node.get("script")),
    }


def choose_topics_and_write_overview(
    topic_roots: list[str], nodes: dict[str, dict[str, Any]]
) -> tuple[list[str], str]:
    candidates = [topic_candidate(topic_id, nodes[topic_id]) for topic_id in topic_roots]
    if any(not candidate["summary"] for candidate in candidates):
        raise ValueError("Every topic root must have a non-empty script.")

    selected_count_min = min(MIN_SELECTED_TOPICS, len(candidates))
    selected_count_max = min(MAX_SELECTED_TOPICS, len(candidates))
    candidate_text = "\n\n".join(
        f"ID: {candidate['topic_id']}\n"
        f"Topic: {candidate['topic']}\n"
        f"Summary: {candidate['summary'][:600]}"
        for candidate in candidates
    )

    schema = {
        "type": "object",
        "properties": {
            "selected_topic_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": selected_count_min,
                "maxItems": selected_count_max,
                "uniqueItems": True,
            },
            "script": {"type": "string"},
        },
        "required": ["selected_topic_ids", "script"],
        "additionalProperties": False,
    }
    prompt = f"""Write a concise spoken English opening for today's personalized podcast.

Select {selected_count_min} to {selected_count_max} topic roots that make the most
useful and varied overview. Prefer specific, informative topics over generic
labels. Treat each selected topic as a separate update; do not imply unrelated
topics are one story. Use only facts in the supplied summaries. Return the IDs
of the selected topic roots and a natural transition-led overview of
{MIN_OVERVIEW_WORDS}-{MAX_OVERVIEW_WORDS} words. Do not mention IDs, counts,
model selection, or unselected topics in the spoken script.

TOPIC ROOTS
{candidate_text}
"""
    result = llm_json(
        prompt=prompt,
        system="Create an accurate, concise podcast episode overview. Return valid JSON only.",
        temperature=0.1,
        max_output_tokens=500,
        json_schema=schema,
    )
    if not isinstance(result, dict):
        raise ValueError("Episode-root generation returned a non-object result.")

    selected_ids = result.get("selected_topic_ids")
    script = normalize_text(result.get("script"))
    valid_ids = set(topic_roots)
    if not isinstance(selected_ids, list):
        raise ValueError("Episode overview must return selected_topic_ids as a list.")
    selected_ids = [normalize_text(item) for item in selected_ids]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Episode overview returned duplicate selected topic IDs.")
    if not selected_count_min <= len(selected_ids) <= selected_count_max:
        raise ValueError(
            f"Episode overview must select {selected_count_min}-"
            f"{selected_count_max} topic roots; got {len(selected_ids)}."
        )
    unknown = [topic_id for topic_id in selected_ids if topic_id not in valid_ids]
    if unknown:
        raise ValueError(f"Episode overview returned unknown topic IDs: {unknown}")
    count = word_count(script)
    if not MIN_OVERVIEW_WORDS <= count <= MAX_OVERVIEW_WORDS:
        raise ValueError(
            f"Episode overview must contain {MIN_OVERVIEW_WORDS}-"
            f"{MAX_OVERVIEW_WORDS} words; got {count}."
        )
    return selected_ids, script


def build_episode_hierarchy(
    topic_hierarchy: dict[str, Any],
    *,
    episode_date: str | None = None,
) -> dict[str, Any]:
    nodes = index_nodes(topic_hierarchy)
    source_active_roots = get_active_roots(topic_hierarchy, nodes)
    episode_children, excluded_singletons = get_episode_child_roots(
        source_active_roots, nodes
    )
    topic_roots = get_topic_roots(episode_children, nodes)
    selected_topic_ids, script = choose_topics_and_write_overview(topic_roots, nodes)

    episode_date = episode_date or date.today().isoformat()
    episode_id = f"episode_root_{episode_date.replace('-', '')}"
    if episode_id in nodes:
        raise ValueError(f"Episode root already exists in input: {episode_id}")

    source_feed_ids = sorted(
        {
            str(feed_id)
            for root in episode_children
            for feed_id in nodes[root].get("source_feed_ids", [])
            if feed_id is not None
        }
    )
    selected_set = set(selected_topic_ids)
    unmentioned_root_ids = [
        root for root in episode_children if root not in selected_set
    ]

    for root in episode_children:
        nodes[root]["parent_id"] = episode_id

    episode_node = {
        "trunk_id": episode_id,
        "node_type": "episode_root",
        "title": "Today's Personalized Podcast",
        "topic_label": "Daily Episode Overview",
        "script": script,
        "word_count": word_count(script),
        "children": episode_children,
        "source_feed_ids": source_feed_ids,
        "selected_topic_root_ids": selected_topic_ids,
        "unmentioned_active_root_ids": unmentioned_root_ids,
        "excluded_singleton_root_count": len(excluded_singletons),
        "generation_method": "llm_overview_of_selected_topic_roots",
        "parent_id": None,
    }

    reachable = reachable_node_ids(episode_children, nodes)
    output_nodes = [node for node_id, node in nodes.items() if node_id in reachable]
    output_nodes.append(episode_node)
    configuration = dict(topic_hierarchy.get("configuration") or {})
    configuration.update(
        {
            "algorithm": "episode_root_over_topic_hierarchy",
            "episode_root_added": True,
            "source_active_root_count": len(source_active_roots),
            "episode_child_root_count": len(episode_children),
            "excluded_singleton_root_count": len(excluded_singletons),
            "topic_root_count": len(topic_roots),
            "narrated_topic_root_count": len(selected_topic_ids),
            "unmentioned_active_root_count": len(unmentioned_root_ids),
            "root_count": 1,
        }
    )
    return {
        "configuration": configuration,
        "roots": [episode_id],
        "nodes": output_nodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Topic hierarchy not found: {args.input}")
    source = read_json(args.input)
    result = build_episode_hierarchy(source)
    atomic_write_json(args.output, result)
    root = result["nodes"][-1]
    print(
        f"Added {root['trunk_id']} above {len(root['children'])} story/topic roots; "
        f"excluded {root['excluded_singleton_root_count']} singleton roots. "
        f"The overview narrates {len(root['selected_topic_root_ids'])} topic roots "
        f"({root['word_count']} words). Saved to {args.output}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
