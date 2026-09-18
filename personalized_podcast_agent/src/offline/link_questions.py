from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.offline.search_web_sources import search_web_answer
from src.utils import PROJECT_ROOT, llm_json


DATA_DIR = PROJECT_ROOT / "data" / "output"
INPUT_FILE = DATA_DIR / "candidate_trunks.json"
OUTPUT_FILE = DATA_DIR / "candidate_trunks_with_next.json"
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
TOP_CANDIDATES = 5


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def load_pool() -> dict[str, Any]:
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def collect_trunks(pool: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for topic in pool.get("topics", []):
        for trunk in topic.get("trunks", []):
            records.append(
                {
                    "topic_id": topic.get("topic_id"),
                    "trunk_id": trunk["trunk_id"],
                    "script": clean(trunk.get("script")),
                    "trunk": trunk,
                }
            )
    return records


def choose_trunk(
    question: str,
    current_trunk_id: str,
    candidates: list[dict[str, Any]],
) -> str | None:
    if not candidates:
        return None

    candidate_text = "\n\n".join(
        f"TRUNK_ID: {item['trunk_id']}\nCONTENT: {item['script']}"
        for item in candidates
    )
    result = llm_json(
        prompt=(
            "Decide whether one candidate trunk contains enough information "
            "to answer the question. Choose only one candidate ID when it does. "
            "Return NONE when none of them answers the question. Never use the "
            "current trunk.\n\n"
            f"Current trunk: {current_trunk_id}\n"
            f"Question: {question}\n\n"
            f"Candidates:\n{candidate_text}"
        ),
        system="Return valid JSON only.",
        temperature=0.0,
        max_output_tokens=120,
        json_schema={
            "type": "object",
            "properties": {
                "answerable": {"type": "boolean"},
                "target_trunk_id": {"type": "string"},
            },
            "required": ["answerable", "target_trunk_id"],
            "additionalProperties": False,
        },
    )

    target = clean(result.get("target_trunk_id"))
    candidate_ids = {item["trunk_id"] for item in candidates}
    if result.get("answerable") and target in candidate_ids:
        return target
    return None


def answer_from_web(question: str, trunk: dict[str, Any]) -> str:
    web_source = search_web_answer(
        question=question,
        trunk_script=clean(trunk.get("script")),
        source_item=trunk.get("source_item") or {},
    )
    if not web_source:
        return ""

    result = llm_json(
        prompt=(
            "Answer the question using only the web source below. "
            "If there is not enough evidence, return an empty answer. "
            "Do not guess.\n\n"
            f"Question:\n{question}\n\n"
            f"Title:\n{clean(web_source.get('title'))}\n\n"
            f"Content:\n{clean(web_source.get('article_text'))}"
        ),
        system="Return valid JSON only.",
        temperature=0.0,
        max_output_tokens=300,
        json_schema={
            "type": "object",
            "properties": {"answer": {"type": "string", "maxLength": 700}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    return clean(result.get("answer"))


def main() -> None:
    pool = load_pool()
    records = collect_trunks(pool)
    if not records:
        raise ValueError("No trunks found in candidate_trunks.json.")

    print("Loading embedding model...", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    embeddings = model.encode(
        [item["script"] for item in records],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    index_by_id = {
        item["trunk_id"]: index for index, item in enumerate(records)
    }

    output_pool = copy.deepcopy(pool)
    output_by_id = {
        trunk["trunk_id"]: trunk
        for topic in output_pool.get("topics", [])
        for trunk in topic.get("trunks", [])
    }

    total_questions = sum(
        len(trunk.get("question_answers", []))
        for topic in output_pool.get("topics", [])
        for trunk in topic.get("trunks", [])
    )
    processed = 0

    for record in records:
        current_id = record["trunk_id"]
        current_index = index_by_id[current_id]
        similarities = embeddings @ embeddings[current_index]
        candidate_indices = [
            index
            for index in np.argsort(-similarities)
            if records[index]["trunk_id"] != current_id
        ][:TOP_CANDIDATES]
        candidates = [records[index] for index in candidate_indices]
        trunk = output_by_id[current_id]

        for question_answer in trunk.get("question_answers", []):
            question = clean(question_answer.get("question"))
            try:
                target_id = choose_trunk(question, current_id, candidates)
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                print(f"Warning: trunk matching failed: {exc}", flush=True)
                target_id = None

            question_answer["answer"] = ""
            question_answer["next_trunks"] = []
            question_answer["audio_file"] = None
            question_answer["audio_status"] = "not_generated"

            if target_id:
                question_answer["next_trunks"] = [target_id]
            else:
                try:
                    answer = answer_from_web(question, trunk)
                except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                    print(f"Warning: web search failed: {exc}", flush=True)
                    answer = ""
                question_answer["answer"] = (
                    answer
                    or "The available sources do not provide enough information to answer this question."
                )
                if candidates:
                    question_answer["next_trunks"] = [
                        candidates[0]["trunk_id"]
                    ]

            processed += 1
            print(f"Linked question {processed}/{total_questions}", flush=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output_pool, file, ensure_ascii=False, indent=2)

    print(f"Saved to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
