from __future__ import annotations

import json
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import PROJECT_ROOT, llm_json


DATA_DIR = PROJECT_ROOT / "data" / "output"
INPUT_FILE = DATA_DIR / "candidate_trunks.json"
OUTPUT_FILE = DATA_DIR / "candidate_trunks_with_next.json"

EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
TOP_K = 5


def load_candidate_pool() -> dict[str, Any]:
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    candidate_pool = load_candidate_pool()
    trunks = [
        trunk
        for topic in candidate_pool.get("topics", [])
        for trunk in topic.get("trunks", [])
    ]

    if len(trunks) < 2:
        raise ValueError("At least two trunks are required.")

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    trunk_texts = [
        f"{trunk.get('title', '')}\n{trunk.get('script', '')}"
        for trunk in trunks
    ]
    trunk_embeddings = model.encode(
        trunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    questions: list[tuple[dict[str, Any], dict[str, Any], str]] = []
    for trunk in trunks:
        for question_answer in trunk.get("question_answers", []):
            question_answer["next_trunks"] = []
            question = str(
                question_answer.get("question") or ""
            ).strip()
            if question:
                questions.append((trunk, question_answer, question))

    question_embeddings = model.encode(
        [question for _, _, question in questions],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    trunk_ids = [str(trunk["trunk_id"]) for trunk in trunks]
    trunk_by_id = {
        str(trunk["trunk_id"]): trunk
        for trunk in trunks
    }

    for number, ((current_trunk, question_answer, question), question_embedding) in enumerate(
        zip(questions, question_embeddings, strict=True),
        start=1,
    ):
        current_trunk_id = str(current_trunk["trunk_id"])

        similarities = trunk_embeddings @ question_embedding
        candidate_indices = [
            int(index)
            for index in np.argsort(-similarities)
            if trunk_ids[int(index)] != current_trunk_id
        ][:TOP_K]

        candidate_text = "\n\n".join(
            f"Trunk ID: {trunk_ids[index]}\n"
            f"Content: {trunks[index].get('script', '')}"
            for index in candidate_indices
        )

        result = llm_json(
            prompt=(
                "Choose the one candidate trunk that best answers or continues "
                "the user's question. Choose only from the candidate trunk IDs. "
                "Return NONE if none is relevant.\n\n"
                f"User question:\n{question}\n\n"
                f"Candidate trunks:\n{candidate_text}"
            ),
            system="Return valid JSON only.",
            temperature=0.0,
            max_output_tokens=100,
            json_schema={
                "type": "object",
                "properties": {
                    "target_trunk_id": {"type": "string"}
                },
                "required": ["target_trunk_id"],
                "additionalProperties": False,
            },
        )

        target_id = str(
            result.get("target_trunk_id") or ""
        ).strip()
        if target_id in trunk_by_id and target_id != current_trunk_id:
            question_answer["next_trunks"] = [target_id]

        print(f"[{number}/{len(questions)}] Linked {current_trunk_id}")

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(candidate_pool, file, ensure_ascii=False, indent=2)

    print(f"Saved updated candidate trunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
