from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


try:
    from src.utils import PROJECT_ROOT
except ImportError:
    # This fallback works when the file is placed in src/user_profile/.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_INPUT = (
    PROJECT_ROOT / "data" / "output" / "personal_feed_items.json"
)
DEFAULT_EMBEDDINGS_OUTPUT = (
    PROJECT_ROOT / "data" / "output" / "personal_feed_embeddings.npy"
)
DEFAULT_INDEX_OUTPUT = (
    PROJECT_ROOT / "data" / "output" / "personal_feed_embedding_ids.json"
)

# This multilingual model supports the English, Chinese, and Korean text that
# may appear in Instagram and YouTube histories. Its embeddings have 384
# dimensions and work with cosine similarity.
DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def build_embedding_text(item: dict[str, Any]) -> str:
    """Build semantic text without adding topic, URL, or timestamp fields."""
    text = clean_text(item.get("text"))
    creator = clean_text(item.get("creator"))

    if creator and creator.casefold() not in text.casefold():
        return f"{text} — {creator}"
    return text


def load_feed_items(
    input_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Feed item file does not exist: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("The feed item file must contain a JSON object.")

    items = payload.get("feed_items")
    if not isinstance(items, list):
        raise ValueError(
            "The feed item file must contain a 'feed_items' JSON array."
        )

    valid_items: list[dict[str, Any]] = []
    seen_feed_ids: set[str] = set()
    skipped_empty = 0

    for row_number, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"feed_items[{row_number}] must be a JSON object."
            )

        feed_id = clean_text(item.get("feed_id"))
        if not feed_id:
            raise ValueError(
                f"feed_items[{row_number}] does not contain a feed_id."
            )
        if feed_id in seen_feed_ids:
            raise ValueError(f"Duplicate feed_id found: {feed_id}")

        embedding_text = build_embedding_text(item)
        if not embedding_text:
            skipped_empty += 1
            continue

        copied_item = dict(item)
        copied_item["feed_id"] = feed_id
        copied_item["_embedding_text"] = embedding_text
        valid_items.append(copied_item)
        seen_feed_ids.add(feed_id)

    input_metadata = payload.get("metadata")
    if not isinstance(input_metadata, dict):
        input_metadata = {}

    load_metadata = {
        "input_total_feed_items": len(items),
        "embedded_feed_items": len(valid_items),
        "skipped_empty_text": skipped_empty,
        "input_generated_at": input_metadata.get("generated_at"),
    }
    return valid_items, load_metadata


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_sentence_transformer(model_name: str, device: str | None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it with: "
            "pip install sentence-transformers"
        ) from exc

    model_arguments: dict[str, Any] = {}
    if device:
        model_arguments["device"] = device

    print(f"Loading embedding model: {model_name}", flush=True)
    return SentenceTransformer(model_name, **model_arguments)


def generate_embeddings(
    model: Any,
    texts: list[str],
    batch_size: int,
) -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is not installed. Install it with: pip install numpy"
        ) from exc

    print(
        f"Generating embeddings for {len(texts):,} feed items...",
        flush=True,
    )
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional embedding matrix, got "
            f"shape {embeddings.shape}."
        )
    if embeddings.shape[0] != len(texts):
        raise ValueError(
            "The number of generated embeddings does not match the number "
            "of input feed items."
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("The embedding matrix contains NaN or infinity.")

    return embeddings


def save_npy_atomic(embeddings: Any, output_path: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is not installed. Install it with: pip install numpy"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("wb") as file:
        np.save(file, embeddings, allow_pickle=False)
    temporary_path.replace(output_path)


def save_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def build_index_payload(
    *,
    items: list[dict[str, Any]],
    embeddings: Any,
    model_name: str,
    input_path: Path,
    load_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "metadata": {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
            "embedding_dimension": int(embeddings.shape[1]),
            "embedding_count": int(embeddings.shape[0]),
            "normalized_embeddings": True,
            "similarity_metric": "cosine_similarity_via_dot_product",
            "embedding_text_fields": ["text", "creator"],
            "topic_included_in_feed_embedding": False,
            "input_file": str(input_path.resolve()),
            "input_sha256": calculate_sha256(input_path),
            **load_metadata,
        },
        # Row i in this list corresponds exactly to row i in the .npy matrix.
        "feed_ids": [item["feed_id"] for item in items],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate normalized semantic embeddings for personal feed items."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input feed JSON path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--embeddings-output",
        type=Path,
        default=DEFAULT_EMBEDDINGS_OUTPUT,
        help=(
            "Output NumPy embedding matrix path "
            f"(default: {DEFAULT_EMBEDDINGS_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=DEFAULT_INDEX_OUTPUT,
        help=(
            "Output feed ID index path "
            f"(default: {DEFAULT_INDEX_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size (default: 128)",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default=None,
        help="Optional inference device; otherwise selected automatically.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    items, load_metadata = load_feed_items(args.input)
    if not items:
        raise ValueError("No valid feed items were found in the input file.")

    texts = [item["_embedding_text"] for item in items]
    model = load_sentence_transformer(args.model, args.device)
    embeddings = generate_embeddings(model, texts, args.batch_size)

    index_payload = build_index_payload(
        items=items,
        embeddings=embeddings,
        model_name=args.model,
        input_path=args.input,
        load_metadata=load_metadata,
    )

    save_npy_atomic(embeddings, args.embeddings_output)
    save_json_atomic(index_payload, args.index_output)

    print(
        "Completed feed embeddings:\n"
        f"  Items: {embeddings.shape[0]:,}\n"
        f"  Dimensions: {embeddings.shape[1]}\n"
        f"  Embeddings: {args.embeddings_output.resolve()}\n"
        f"  Feed ID index: {args.index_output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
