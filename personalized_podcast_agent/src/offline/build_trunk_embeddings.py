"""Create normalized embeddings for generated podcast trunks.

Run from the project root:
    python -m src.offline.build_trunk_embeddings

For a different input or output location:
    python -m src.offline.build_trunk_embeddings \
        --input data/output/candidate_trunks.json \
        --embeddings data/output/trunk_embeddings.npy \
        --embedding-ids data/output/trunk_embedding_ids.json
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from src.embedding_store import encode_texts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "output"
INPUT_FILE = DATA_DIR / "candidate_trunks.json"
EMBEDDINGS_FILE = DATA_DIR / "trunk_embeddings.npy"
EMBEDDING_IDS_FILE = DATA_DIR / "trunk_embedding_ids.json"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def load_trunks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        pool = json.load(file)

    trunks = pool.get("trunks") if isinstance(pool, dict) else None
    if not isinstance(trunks, list) or not trunks:
        raise ValueError(f"No trunks found in {path}.")
    if any(not isinstance(trunk, dict) for trunk in trunks):
        raise ValueError("Every trunk must be an object.")

    trunk_ids = [str(trunk.get("trunk_id") or "").strip() for trunk in trunks]
    if any(not trunk_id for trunk_id in trunk_ids):
        raise ValueError("Every trunk must have a non-empty trunk_id.")
    if len(trunk_ids) != len(set(trunk_ids)):
        raise ValueError(
            "candidate_trunks contains duplicate trunk_id values.")
    if any(not str(trunk.get("script") or "").strip() for trunk in trunks):
        raise ValueError("Every trunk must have a non-empty script.")
    return trunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--embedding-ids", type=Path,
                        default=EMBEDDING_IDS_FILE)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"candidate_trunks file not found: {args.input}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    trunks = load_trunks(args.input)
    trunk_ids = [str(trunk["trunk_id"]) for trunk in trunks]
    scripts = [str(trunk["script"]).strip() for trunk in trunks]

    embeddings = encode_texts(scripts, EMBEDDING_MODEL,
                              batch_size=args.batch_size, show_progress=True)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(trunk_ids):
        raise ValueError(
            f"Unexpected embedding shape {embeddings.shape} for "
            f"{len(trunk_ids)} trunks."
        )

    args.embeddings.parent.mkdir(parents=True, exist_ok=True)
    args.embedding_ids.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings, embeddings)
    with args.embedding_ids.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "trunk_ids": trunk_ids,
                "metadata": {
                    "model_name": EMBEDDING_MODEL,
                    "input_signature": hashlib.sha256(json.dumps(
                        list(zip(trunk_ids, scripts)), ensure_ascii=False
                    ).encode("utf-8")).hexdigest(),
                    "normalized": True,
                    "count": len(trunk_ids),
                    "dimension": int(embeddings.shape[1]),
                },
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved {len(trunk_ids)} embeddings to {args.embeddings}")
    print(f"Saved trunk ID mapping to {args.embedding_ids}")


if __name__ == "__main__":
    main()
