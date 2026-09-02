from __future__ import annotations

import json
import math
from typing import Any

from src.utils import llm_json


QUESTION_TYPES = (
    "clarification",
    "deeper_explanation",
    "causal_reasoning",
    "comparison",
    "practical_application",
    "related_topic",
)

CANDIDATE_COUNT = 6
TOP_K = 3
SOFTMAX_TEMPERATURE = 1.0


def generate_candidate_questions(
    topic: str,
    trunk_script: str,
    selected_feed_seed: dict[str, Any],
    user_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Stage 1:
    Generate six diverse candidate questions without probabilities.
    """

    result = llm_json(
        prompt=(
            f"Generate exactly {CANDIDATE_COUNT} candidate questions "
            "that this specific user is most likely to ask after hearing "
            "the current podcast trunk.\n\n"

            "Requirements:\n"
            "- The podcast topic is fixed.\n"
            "- Every question must naturally extend the trunk script.\n"
            "- Use the selected feed seed to understand why this trunk "
            "matches the user's interests.\n"
            "- Use the user profile only to identify likely long-term "
            "interests and preferences.\n"
            "- Do not introduce an unrelated profile interest.\n"
            "- Choose six meaningfully different information directions within the current topic.\n"
            "- Ask about information not already answered by the trunk.\n"
            "- Do not merely rewrite a sentence from the trunk as a question.\n"
            "- Each question must be one complete English sentence.\n"
            "- Keep every question under 20 words.\n"
            "- Do not answer the questions.\n"
            "- Do not assign probabilities or scores.\n\n"

            "Podcast topic:\n"
            f"{topic}\n\n"

            "Selected personal feed seed:\n"
            f"{json.dumps(selected_feed_seed, ensure_ascii=False, indent=2)}\n\n"

            "User profile:\n"
            f"{json.dumps(user_profile, ensure_ascii=False, indent=2)}\n\n"

            "Trunk script:\n"
            f"{trunk_script}"
        ),
        system=(
            "You generate candidate user interactions for an offline "
            "personalized podcast. Stay within the current podcast "
            "topic. Return valid JSON only."
        ),
        temperature=0.5,
        max_output_tokens=700,
        json_schema={
            "type": "object",
            "properties": {
                "candidate_questions": {
                    "type": "array",
                    "minItems": CANDIDATE_COUNT,
                    "maxItems": CANDIDATE_COUNT,
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "maxLength": 250,
                            },
                            "question_type": {
                                "type": "string",
                                "enum": list(QUESTION_TYPES),
                            },
                            "information_target": {
                                "type": "string",
                                "maxLength": 180,
                            },
                        },
                        "required": [
                            "question",
                            "question_type",
                            "information_target",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["candidate_questions"],
            "additionalProperties": False,
        },
    )

    candidates = result["candidate_questions"]

    for candidate in candidates:
        candidate["question"] = str(
            candidate["question"]
        ).strip()

    return candidates


def rerank_candidate_questions(
    candidates: list[dict[str, Any]],
    topic: str,
    trunk_script: str,
    selected_feed_seed: dict[str, Any],
    user_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Stage 2:
    Score all six candidates and return the Top-3.
    """

    candidates_with_ids = [
        {
            "candidate_id": f"q{index}",
            **candidate,
        }
        for index, candidate in enumerate(candidates)
    ]

    candidate_ids = [
        item["candidate_id"]
        for item in candidates_with_ids
    ]

    result = llm_json(
        prompt=(
            "Rank the candidate questions according to how likely this "
            "specific user is to ask each question next.\n\n"

            "Evaluate each candidate using:\n"
            "1. relevance to the trunk script,\n"
            "2. consistency with the podcast topic,\n"
            "3. relevance to the selected personal feed seed,\n"
            "4. relevance to the user's long-term profile,\n"
            "5. novelty beyond information already stated in the trunk.\n\n"

            "Scoring requirements:\n"
            "- Assign every candidate a score from 0 to 10.\n"
            "- A higher score means the user is more likely to ask it.\n"
            "- Score each candidate exactly once.\n"
            "- Do not rewrite the questions.\n"
            "- Do not introduce new candidates.\n\n"

            "Podcast topic:\n"
            f"{topic}\n\n"

            "Selected personal feed seed:\n"
            f"{json.dumps(selected_feed_seed, ensure_ascii=False, indent=2)}\n\n"

            "User profile:\n"
            f"{json.dumps(user_profile, ensure_ascii=False, indent=2)}\n\n"

            "Trunk script:\n"
            f"{trunk_script}\n\n"

            "Candidate questions:\n"
            f"{json.dumps(
                candidates_with_ids,
                ensure_ascii=False,
                indent=2,
            )}"
        ),
        system=(
            "You are a personalized user-action ranking model. "
            "Return candidate IDs and scores as valid JSON only."
        ),
        temperature=0.0,
        max_output_tokens=500,
        json_schema={
            "type": "object",
            "properties": {
                "ranked_candidates": {
                    "type": "array",
                    "minItems": CANDIDATE_COUNT,
                    "maxItems": CANDIDATE_COUNT,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "enum": candidate_ids,
                            },
                            "score": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 10,
                            },
                        },
                        "required": [
                            "candidate_id",
                            "score",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["ranked_candidates"],
            "additionalProperties": False,
        },
    )

    score_by_id: dict[str, float] = {}

    for item in result["ranked_candidates"]:
        candidate_id = str(item["candidate_id"])

        if candidate_id in score_by_id:
            raise ValueError(
                f"Duplicate candidate ID returned: {candidate_id}"
            )

        score_by_id[candidate_id] = float(item["score"])

    missing_ids = [
        candidate_id
        for candidate_id in candidate_ids
        if candidate_id not in score_by_id
    ]

    if missing_ids:
        raise ValueError(
            f"Reranker did not score candidates: {missing_ids}"
        )

    ranked_ids = sorted(
        candidate_ids,
        key=lambda candidate_id: score_by_id[candidate_id],
        reverse=True,
    )

    selected_ids = ranked_ids[:TOP_K]
    selected_scores = [
        score_by_id[candidate_id]
        for candidate_id in selected_ids
    ]

    probabilities = softmax(
        selected_scores,
        temperature=SOFTMAX_TEMPERATURE,
    )

    candidate_by_id = {
        item["candidate_id"]: item
        for item in candidates_with_ids
    }

    selected_candidates: list[dict[str, Any]] = []

    for candidate_id, probability in zip(
        selected_ids,
        probabilities,
    ):
        original = candidate_by_id[candidate_id]

        selected_candidates.append({
            "question": original["question"],
            "question_type": original["question_type"],
            "information_target": original["information_target"],
            "probability": round(probability, 6),
        })

    return selected_candidates


def softmax(
    scores: list[float],
    *,
    temperature: float,
) -> list[float]:
    if not scores:
        return []

    if temperature <= 0:
        raise ValueError("Softmax temperature must be greater than zero.")

    # Subtracting the maximum prevents numerical overflow.
    maximum_score = max(scores)

    exponentials = [
        math.exp((score - maximum_score) / temperature)
        for score in scores
    ]

    total = sum(exponentials)

    return [
        value / total
        for value in exponentials
    ]


def predict_user_questions(
    topic: str,
    trunk_script: str,
    selected_feed_seed: dict[str, Any],
    user_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Stage 1: generate six candidates.
    Stage 2: rerank and return the Top-3.
    """

    candidates = generate_candidate_questions(
        topic=topic,
        trunk_script=trunk_script,
        selected_feed_seed=selected_feed_seed,
        user_profile=user_profile,
    )

    return rerank_candidate_questions(
        candidates=candidates,
        topic=topic,
        trunk_script=trunk_script,
        selected_feed_seed=selected_feed_seed,
        user_profile=user_profile,
    )
