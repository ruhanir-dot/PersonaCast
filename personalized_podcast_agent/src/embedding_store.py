"""Shared sentence embeddings: one encoder per model and a content-addressed disk cache.

Vectors are keyed by sha256(text), so an unchanged script is never re-encoded,
regardless of which pipeline step or the API asks for it.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
from pathlib import Path

import numpy as np

from src.utils import OUTPUT_DIR

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
CACHE_DIR = OUTPUT_DIR / "embedding_cache"

_ENCODERS: dict[str, object] = {}
_CACHES: dict[str, dict[str, np.ndarray]] = {}
# One lock per process: encoder.encode is not guaranteed thread-safe on GPU.
_LOCK = threading.RLock()


def get_encoder(model_name: str = DEFAULT_MODEL):
    with _LOCK:
        if model_name not in _ENCODERS:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[embeddings] loading {model_name} on {device}", flush=True)
            _ENCODERS[model_name] = SentenceTransformer(model_name, device=device)
        return _ENCODERS[model_name]


def _cache_path(model_name: str, cache_dir: Path) -> Path:
    return cache_dir / (re.sub(r"[^A-Za-z0-9._-]+", "_", model_name) + ".npz")


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(model_name: str, cache_dir: Path) -> dict[str, np.ndarray]:
    key = f"{cache_dir}|{model_name}"
    if key not in _CACHES:
        cache: dict[str, np.ndarray] = {}
        path = _cache_path(model_name, cache_dir)
        if path.exists():
            try:
                with np.load(path) as data:
                    cache = dict(zip(data["keys"].tolist(), data["vectors"]))
            except (OSError, ValueError, KeyError) as exc:
                print(f"[embeddings] ignoring unreadable cache {path}: {exc}", flush=True)
        _CACHES[key] = cache
    return _CACHES[key]


def _save_cache(model_name: str, cache_dir: Path, cache: dict[str, np.ndarray]) -> None:
    path = _cache_path(model_name, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez(temporary, keys=np.array(list(cache), dtype="U64"),
             vectors=np.stack(list(cache.values())).astype(np.float32))
    os.replace(temporary, path)


def encode_texts(texts: list[str], model_name: str = DEFAULT_MODEL, *, cache: bool = True,
                 batch_size: int = 16, show_progress: bool = False,
                 cache_dir: Path | None = None) -> np.ndarray:
    """Return normalized float32 vectors (len(texts), dim), encoding only uncached texts."""
    cache_dir = cache_dir or CACHE_DIR
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    with _LOCK:
        if not cache:
            return np.asarray(get_encoder(model_name).encode(
                list(texts), normalize_embeddings=True, batch_size=batch_size,
                show_progress_bar=show_progress), dtype=np.float32)
        store = _load_cache(model_name, cache_dir)
        keys = [_text_key(text) for text in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in store))
        if missing:
            by_key = dict(zip(keys, texts))
            print(f"[embeddings] encoding {len(missing)}/{len(texts)} uncached texts", flush=True)
            vectors = np.asarray(get_encoder(model_name).encode(
                [by_key[k] for k in missing], normalize_embeddings=True, batch_size=batch_size,
                show_progress_bar=show_progress), dtype=np.float32)
            store.update(zip(missing, vectors))
            _save_cache(model_name, cache_dir, store)
        return np.stack([store[k] for k in keys]).astype(np.float32)
