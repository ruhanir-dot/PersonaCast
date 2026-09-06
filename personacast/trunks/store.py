from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from .schema import EmbeddingIndex, TrunkPool

POOL_FILE = "candidate_trunks.json"
TRUNK_EMBEDDINGS_FILE = "candidate_trunk_embeddings.npy"
QUESTION_EMBEDDINGS_FILE = "candidate_question_embeddings.npy"
EMBEDDING_IDS_FILE = "candidate_embedding_ids.json"


class TrunkStoreError(RuntimeError):
    """
    error handling if pool is missing or inconsistent with embeddings 
    """
    pass


@dataclass
class LoadedTrunks:
    """
    validated trunk pool for all 10 topics 
    """
    pool: TrunkPool
    index: EmbeddingIndex
    trunk_embeddings: np.ndarray
    question_embeddings: np.ndarray
    source_dir: Path

    def question_row(self, qa_id: str) -> int | None:
        try:
            return self.index.question_ids.index(qa_id)
        except ValueError:
            return None

    def trunk_row(self, trunk_id: str) -> int | None:
        try:
            return self.index.trunk_ids.index(trunk_id)
        except ValueError:
            return None

    def audio_path(self, relative: str | None) -> Path | None:
        if not relative:
            return None
        candidate = self.source_dir / relative
        return candidate if candidate.exists() else None


def pool_dir() -> Path:
    return Path(config.TRUNK_POOL_DIR)


def pool_dir_for(persona_id: str | None) -> Path:
    """
    pool directory for one listener at their trunk_pool_root/persona_id
    """
    if persona_id:
        candidate = Path(config.TRUNK_POOL_ROOT) / persona_id
        if (candidate / POOL_FILE).exists():
            return candidate
    return pool_dir()


def available_personas() -> list[str]:
    """
    listeners with available pool on disk so we show on UI
    """
    root = Path(config.TRUNK_POOL_ROOT)
    if not root.is_dir():
        return []
    return sorted(
        d.name for d in root.iterdir()
        if d.is_dir() and (d / POOL_FILE).exists()
    )


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TrunkStoreError(
            f"trunk pool artifact missing: {path}\n"
            "build it first with:\n"
            "  cd personalized_podcast_agent\n"
            '  export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")\n'
            "  python -m src.offline.generate_trunks"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TrunkStoreError(f"trunk pool artifact is not valid JSON: {path}") from exc


def _load_matrix(path: Path, expected_rows: int, expected_dim: int, label: str) -> np.ndarray:
    if not path.exists():
        raise TrunkStoreError(f"{label} embeddings missing: {path}")

    matrix = np.load(path, allow_pickle=False)

    if matrix.ndim != 2:
        raise TrunkStoreError(f"{label} embeddings must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] != expected_rows:
        raise TrunkStoreError(
            f"{label} embeddings have {matrix.shape[0]} rows but the id index "
            f"lists {expected_rows}. the pool and its embeddings are out of sync -- "
            "rerun the offline build."
        )
    if matrix.shape[1] != expected_dim:
        raise TrunkStoreError(
            f"{label} embeddings are {matrix.shape[1]}-d, expected {expected_dim}"
        )

    return matrix.astype(np.float32, copy=False)


def load(source_dir: Path | str | None = None, *, persona_id: str | None = None) -> LoadedTrunks:
    """
    read and validate the pool raises error on any inconsistency
    """
    directory = Path(source_dir) if source_dir is not None else pool_dir_for(persona_id)

    pool = TrunkPool.model_validate(_read_json(directory / POOL_FILE))
    index = EmbeddingIndex.model_validate(_read_json(directory / EMBEDDING_IDS_FILE))

    if index.metadata.model_name != config.TRUNK_EMBED_MODEL:
        raise TrunkStoreError(
            "embedding model mismatch -- online queries would not be comparable "
            "to the offline vectors.\n"
            f"  pool was built with: {index.metadata.model_name}\n"
            f"  config expects:      {config.TRUNK_EMBED_MODEL}\n"
            "set PERSONACAST_TRUNK_EMBED_MODEL to match, or rebuild the pool."
        )

    dim = index.metadata.embedding_dimension
    trunk_embeddings = _load_matrix(
        directory / TRUNK_EMBEDDINGS_FILE, len(index.trunk_ids), dim, "trunk"
    )
    question_embeddings = _load_matrix(
        directory / QUESTION_EMBEDDINGS_FILE, len(index.question_ids), dim, "question"
    )

    _check_ids_match_pool(pool, index)

    return LoadedTrunks(
        pool=pool,
        index=index,
        trunk_embeddings=trunk_embeddings,
        question_embeddings=question_embeddings,
        source_dir=directory,
    )


def _check_ids_match_pool(pool: TrunkPool, index: EmbeddingIndex) -> None:
    trunk_ids = {trunk.trunk_id for trunk in pool.all_trunks()}
    missing = [t for t in index.trunk_ids if t not in trunk_ids]
    if missing:
        raise TrunkStoreError(
            f"{len(missing)} embedded trunk id(s) absent from the pool, "
            f"first: {missing[0]}"
        )

    qa_ids = {qa.qa_id for trunk in pool.all_trunks() for qa in trunk.question_answers}
    missing_qa = [q for q in index.question_ids if q not in qa_ids]
    if missing_qa:
        raise TrunkStoreError(
            f"{len(missing_qa)} embedded question id(s) absent from the pool, "
            f"first: {missing_qa[0]}"
        )
