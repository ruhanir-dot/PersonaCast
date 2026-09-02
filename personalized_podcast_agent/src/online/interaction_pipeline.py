from typing import Any

from src.offline.question_index import (
    encode_user_question,
    search_by_embedding,
)
from src.online.adapt_candidate import adapt_tree_script
from src.online.generate_live_fallback import generate_tree_fallback


MATCH_THRESHOLD = 0.80


def run_tree_interaction(
    *,
    current_node: dict[str, Any],
    user_question: str,
) -> dict[str, Any]:
    question = user_question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    embedding = encode_user_question(question).reshape(-1)
    matches = search_by_embedding(embedding, top_k=3)
    best_match = matches[0] if matches else None
    similarity = (
        float(best_match["similarity"])
        if best_match
        else 0.0
    )

    print("Similarity matches:")
    for match in matches:
        print(
            match["similarity"],
            match["user_question"],
        )

    if best_match and similarity >= MATCH_THRESHOLD:
        mode = "adapt"
        matched_node_id = best_match.get("node_id")
        script = adapt_tree_script(
            current_script=str(current_node.get("script", "")),
            user_question=question,
            matched_question=str(best_match.get("user_question", "")),
            matched_script=str(best_match.get("script", "")),
        )
    else:
        mode = "generate"
        matched_node_id = None
        script = generate_tree_fallback(
            current_node=current_node,
            user_question=question,
        )

    return {
        "mode": mode,
        "user_question": question,
        "matched_node_id": matched_node_id,
        "similarity": similarity,
        "script": script,
        "embedding_dimension": int(embedding.shape[0]),
        "embedding_norm": float(
            (embedding @ embedding) ** 0.5
        ),
    }
