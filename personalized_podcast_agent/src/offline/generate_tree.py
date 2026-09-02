from __future__ import annotations

import json
import heapq
import uuid
import os
import threading
import time
from itertools import count
from pathlib import Path
from typing import Any

from src.audio.tts import generate_tts
from src.offline.generate_main_narrative import (
    generate_segments,
    retrieve_recommended_contents,
)
from src.offline.predict_user_actions import predict_user_questions
from src.offline.question_index import add_nodes
from src.utils import PROJECT_ROOT, llm_json

DATA_DIR = PROJECT_ROOT / "data" / "output"
USER_PROFILE_FILE = DATA_DIR / "user_profile.json"
OUTPUT_FILE = DATA_DIR / "candidate_tree.json"


CANDIDATE_POOL_SIZE = 200
TOPIC_COUNT = 10
_SAVE_LOCK = threading.Lock()


def load_user_profile() -> dict[str, Any]:
    with USER_PROFILE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def create_node(
    *,
    depth: int,
    question: str,
    question_type: str,
    script: str,
    topic: str,
    history: list[dict[str, Any]],
    probability: float,
) -> dict[str, Any]:
    return {
        "node_id": str(uuid.uuid4()),
        "depth": depth,
        "user_question": question,
        "question_type": question_type,
        "topic": topic,
        "script": script,
        "path_history": history,
        "children": [],
        "probability": probability,
        "audio_file": None,
        "audio_status": "not_generated",
    }


def save_tree(root: dict[str, Any]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary_file = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.{os.getpid()}.tmp"
    )

    with _SAVE_LOCK:
        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(root, file, ensure_ascii=False, indent=2)

        for attempt in range(5):
            try:
                os.replace(temporary_file, OUTPUT_FILE)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.1)


def predict_topics(user_profile: dict[str, Any]) -> list[str]:
    topics = [
        str(topic).strip()
        for topic in user_profile.get("podcast_focus_keywords", [])
        if str(topic).strip()
    ]

    if len(topics) < TOPIC_COUNT:
        raise ValueError(
            f"Expected at least {TOPIC_COUNT} podcast focus keywords, "
            f"but received {len(topics)}."
        )

    return topics[:TOPIC_COUNT]


def limit_words(text: str, maximum: int = 30) -> str:
    words = text.strip().split()

    if len(words) <= maximum:
        return text.strip()

    return " ".join(words[:maximum]).rstrip(",:;-") + "."


def generate_initial_script(topic: str, source_items: list[dict[str, Any]]) -> str:
    result = llm_json(
        prompt=(
            f"Topic:\n{topic}\n\n"
            "Retrieved sources:\n"
            f"{json.dumps(source_items, ensure_ascii=False, indent=2)}\n\n"
            "Generate a concise spoken podcast introduction.\n"
            "Use only information supported by the retrieved sources.\n"
            "Do not invent facts, dates, events, or quotations."
            "Write between 45 to 70 words"
        ),
        temperature=0.0,
        max_output_tokens=300,
        json_schema={
            "type": "object",
            "properties": {
                "script": {
                    "type": "string",
                    "minLength": 120,
                    "maxLength": 500,

                }
            },
            "required": ["script"],
            "additionalProperties": False,
        },
    )

    return str(result["script"]).strip()


def print_node(node: dict[str, Any]) -> None:
    separator = "#" if node["depth"] == 1 else "="
    line = separator * 70

    if node["depth"] == 1:
        question_label = "Topic / Initial Request"
    else:
        question_label = "User Question"

    print(
        f"\n{line}\n"
        f"Depth {node['depth']}\n"
        f"Path_Probability: {node['path_probability']}\n"
        f"{question_label}: {node['user_question']}\n"
        f"Script: {node['script']}\n"
        f"{line}"
    )


def expand_tree(node: dict[str, Any], user_profile: dict[str, Any], root: dict[str, Any]) -> list[dict[str, Any]]:

    node.setdefault("children", [])

    # Prevent the same node from being expanded twice.
    if node["children"]:
        return []

    prediction_context = {
        "topic": node["topic"],
        "current_depth": node["depth"],
        "current_script": node["script"],
    }

    branch_profile = {
        "selected_topic": node["topic"],
        "style_preferences": user_profile.get(
            "style_preferences",
            [],
        ),
    }

    predictions = predict_user_questions(
        narrative=prediction_context,
        user_profile=branch_profile,
        path_history=node["path_history"],
    )

    for predicted in predictions:
        question = str(predicted["question"]).strip()
        query = f"{node['topic']} {question}".strip()
        predicted["source_items"] = retrieve_recommended_contents(query=query)

    scripts = generate_segments(
        current_script=node["script"],
        predicted_questions=predictions,
    )

    if len(predictions) != len(scripts):
        raise ValueError(
            "The number of predictions and scripts does not match: "
            f"{len(predictions)} predictions, "
            f"{len(scripts)} scripts."
        )

    new_children: list[dict[str, Any]] = []
    parent_path_probability = float(
        node.get("path_probability")
        or node.get("probability")
        or 0.0
    )

    for predicted, script in zip(predictions, scripts):
        question = str(predicted["question"]).strip()
        probability = float(predicted["probability"])
        history_item = {
            "question": question,
            "question_type": predicted["question_type"],
            "information_target": predicted.get(
                "information_target",
                "",
            ),
            "probability": predicted["probability"],
            "script": script,
        }

        child = create_node(
            depth=node["depth"] + 1,
            question=question,
            question_type=str(
                predicted.get(
                    "question_type",
                    "ask_question",
                )
            ),
            script=script,
            topic=node["topic"],
            history=[
                *node["path_history"],
                history_item,
            ],
            probability=probability,
        )

        child["parent_id"] = node["node_id"]
        child["children"] = child.get("children", [])
        child["conditional_probability"] = probability
        child["path_probability"] = (
            parent_path_probability * probability
        )
        child["expanded"] = False

        node["children"].append(child)
        new_children.append(child)

        print(
            f"{child['node_id']} script finish",
            flush=True,
        )
        print_node(child)

    add_nodes(new_children)
    save_tree(root)

    return new_children


def materialize_child_audio(node: dict[str, Any], root: dict[str, Any]) -> None:
    """Generate audio only for the three direct children of this node."""
    children = node.get("children", [])

    print(
        f"{node['node_id']} has {len(children)} children",
        flush=True,
    )

    for child in children:
        audio_value = child.get("audio_file")
        audio_path = Path(audio_value) if audio_value else None

        if audio_path and audio_path.exists():
            print(
                f"{child['node_id']} audio already exists",
                flush=True,
            )
            continue

        print(
            f"{child['node_id']} audio start",
            flush=True,
        )

        audio_file = generate_tts(
            text=child["script"],
            node_id=child["node_id"],
        )

        child["audio_file"] = str(audio_file)
        child["audio_status"] = "ready"

        print(
            f"{child['node_id']} audio finish",
            flush=True,
        )

    save_tree(root)


def main() -> None:
    user_profile = load_user_profile()

    root: dict[str, Any] = {
        "node_id": "root",
        "depth": 0,
        "script": "What do you want to listen to?",
        "children": [],
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    audio_file = generate_tts(text=root["script"], node_id=root["node_id"])
    root["audio_file"] = str(audio_file)
    root["audio_status"] = "ready"

    save_tree(root)

    topics = predict_topics(user_profile)
    topic_scores = [
        max(0.0, float(topic.get("probability") or 0.0))
        for topic in topics
    ]
    topic_total = sum(topic_scores)

    if topic_total <= 0:
        topic_probabilities = [
            1.0 / len(topics)
            for _ in topics
        ]
    else:
        topic_probabilities = [
            score / topic_total
            for score in topic_scores
        ]

    candidate_pool: list[dict[str, Any]] = []
    frontier: list[tuple[float, int, dict[str, Any]]] = []
    heap_counter = count()

    for topic_data, probability in zip(topics, topic_probabilities):
        topic = str(topic_data["topic"]).strip()

        source_items = retrieve_recommended_contents(
            query=topic,
        )
        script = generate_initial_script(topic, source_items)

        node = create_node(
            depth=1,
            question=topic,
            question_type="topic_selection",
            script=script,
            topic=topic,
            history=[
                {
                    "question": topic,
                    "script": script,
                    "probability": probability,
                }
            ],
            probability=probability,
        )

        node["probability"] = probability
        node["conditional_probability"] = probability
        node["path_probability"] = probability
        node["path_score"] = probability
        node["expanded"] = False

        print(f"{node['node_id']} script finish")

        root["children"].append(node)
        candidate_pool.append(node)

        heapq.heappush(
            frontier,
            (
                -probability,
                next(heap_counter),
                node,
            ),
        )

        # add_nodes() 接收的是 list
        add_nodes([node])

        audio_file = generate_tts(
            text=node["script"],
            node_id=node["node_id"],
        )

        node["audio_file"] = str(audio_file)
        node["audio_status"] = "ready"

        print_node(node)
        save_tree(root)

    while (len(candidate_pool) < CANDIDATE_POOL_SIZE and frontier):
        if CANDIDATE_POOL_SIZE - len(candidate_pool) < 3:
            break

        _, _, parent = heapq.heappop(frontier)

        if parent.get("expanded", False):
            continue

        parent["expanded"] = True

        children = expand_tree(
            parent,
            user_profile=user_profile,
            root=root,
        )

        for child in children:
            child["path_score"] = float(child["path_probability"])
            child["expanded"] = False

            candidate_pool.append(child)

            heapq.heappush(
                frontier,
                (
                    -child["path_probability"],
                    next(heap_counter),
                    child,
                ),
            )

            # If all audio must be generated offline
            if child.get("audio_status") != "ready":
                audio_file = generate_tts(
                    text=child["script"],
                    node_id=child["node_id"],
                )
                child["audio_file"] = str(audio_file)
                child["audio_status"] = "ready"

        save_tree(root)
        add_nodes(children)

        print(
            f"Candidate pool: "
            f"{len(candidate_pool)}/{CANDIDATE_POOL_SIZE}"
        )

    save_tree(root)

    print(
        f"Offline candidate tree completed: "
        f"{len(candidate_pool)} nodes"
    )


if __name__ == "__main__":
    main()
