from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

indexed_nodes: list[dict[str, Any]] = []
indexed_node_ids: set[str] = set()

embeddings = np.empty(
    (0, 384),
    dtype=np.float32,
)


def encode_user_question(question: str) -> np.ndarray:
    question = question.strip()
    embedding = model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True)
    return embedding.astype(np.float32)


def reset_index() -> None:
    global embeddings

    indexed_nodes.clear()
    indexed_node_ids.clear()
    embeddings = np.empty(
        (0, 384),
        dtype=np.float32,
    )


# add preditcted node to embedding
def add_nodes(nodes: list[dict[str, Any]]) -> None:
    global embeddings

    valid_nodes = [
        node
        for node in nodes
        if node.get("node_id")
        and node.get("user_question")
        and str(node["node_id"]) not in indexed_node_ids
    ]

    if not valid_nodes:
        return

    questions = [
        str(node["user_question"]).strip()
        for node in valid_nodes
    ]

    new_embeddings = model.encode(
        questions,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    embeddings = np.vstack([
        embeddings,
        new_embeddings,
    ])

    for node in valid_nodes:
        node_id = str(node["node_id"])

        indexed_nodes.append({
            "node_id": node_id,
            "user_question": node["user_question"],
            "depth": node.get("depth"),
            "topic": node.get("topic"),
            "script": node.get("script"),
        })
        indexed_node_ids.add(node_id)


def search_by_embedding(
    query_embedding: np.ndarray,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    if not indexed_nodes or embeddings.size == 0:
        return []

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    if embeddings.shape[1] != query_embedding.shape[0]:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"nodes={embeddings.shape[1]}, "
            f"query={query_embedding.shape[0]}"
        )

    # 两边都已经 normalize，所以 dot product 就是 cosine similarity。
    scores = embeddings @ query_embedding

    top_k = min(top_k, len(indexed_nodes))
    indices = np.argsort(scores)[::-1][:top_k]

    return [
        {
            **indexed_nodes[int(index)],
            "similarity": float(scores[index]),
        }
        for index in indices
    ]
