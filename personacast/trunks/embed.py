"""
sentence embedding for online question matching
we embed a listeners spoken interruption aainst offline question bank 

we take listeners raw utterance and fire the embedding based on the regex lookslikeaquestion method in parallel with interpret call 

makes sure that the model we ate using is the same one pool is build with and keep same 384d embedding 

e embed raw utterance into 384d vector and compute cosine similarity bia dot product against the 3 row slice of question matrix of the 3 predicted questions of the given trink 
then take best scoring hit

score then gets compared to 0.45 threshold
"""


from __future__ import annotations
import threading
import numpy as np
from .. import config

DEFAULT_DIM = 384

_model = None
_lock = threading.Lock()


def _choose_device() -> str | None:
    if config.TRUNK_EMBED_DEVICE:
        return config.TRUNK_EMBED_DEVICE

    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


def get_model():
    """
    load once per process
    """
    global _model

    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                device = _choose_device()
                kwargs = {"device": device} if device else {}
                _model = SentenceTransformer(config.TRUNK_EMBED_MODEL, **kwargs)

    return _model


def warm() -> None:
    """
    keep model loading at sesison start to make sure import time isnt felt at interruption
    """
    try:
        encode("warm")
    except Exception:
        pass


def encode(texts: str | list[str]) -> np.ndarray:
    """
    turning given text in
    """
    single = isinstance(texts, str)
    batch = [texts] if single else list(texts)

    if not batch:
        return np.zeros((0, DEFAULT_DIM), dtype=np.float32)

    model = get_model()
    with _lock:
        vectors = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

    return vectors[0] if single else vectors


def search(vector: np.ndarray, matrix: np.ndarray, top_k: int = 5) -> list[tuple[int, float]]:
    """
    search utility to get cosine similarity via dot product of vector and matrix 
    """
    if matrix.size == 0 or vector.size == 0:
        return []

    scores = matrix @ vector
    top_k = max(1, min(top_k, int(scores.shape[0])))
    top = np.argpartition(scores, -top_k)[-top_k:]
    ranked = sorted(top, key=lambda i: float(scores[i]), reverse=True)
    return [(int(index), float(scores[index])) for index in ranked]


def best_match(vector: np.ndarray, matrix: np.ndarray) -> tuple[int, float] | None:
    """
    get best matched question to utterance
    """
    hits = search(vector, matrix, top_k=1)
    return hits[0] if hits else None
