from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.utils import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data" / "output"
FEED_ITEMS_FILE = DATA_DIR / "personal_feed_items.json"
FEED_EMBEDDINGS_FILE = DATA_DIR / "personal_feed_embeddings.npy"
FEED_IDS_FILE = DATA_DIR / "personal_feed_embedding_ids.json"
OUTPUT_FILE = DATA_DIR / "topic_feed_seeds.json"

TOP_CANDIDATES = 50
SEEDS_PER_TOPIC = 8
MMR_LAMBDA = 0.60
MAX_PAIR_SIMILARITY = 0.80
TRANSLATION_MODEL = "facebook/nllb-200-distilled-600M"

HAN_PATTERN = re.compile(r"[\u3400-\u9fff]")
HIRAGANA_KATAKANA_PATTERN = re.compile(r"[\u3040-\u30ff]")
HANGUL_PATTERN = re.compile(r"[\uac00-\ud7af]")


def load_embedding_data() -> tuple[
    list[dict[str, Any]],
    np.ndarray,
    str,
]:
    with FEED_ITEMS_FILE.open("r", encoding="utf-8") as file:
        feed_data = json.load(file)

    with FEED_IDS_FILE.open("r", encoding="utf-8") as file:
        embedding_index = json.load(file)

    embeddings = np.load(FEED_EMBEDDINGS_FILE, allow_pickle=False)
    feed_ids = embedding_index["feed_ids"]
    model_name = embedding_index["metadata"]["model_name"]

    if len(feed_ids) != len(embeddings):
        raise ValueError(
            "The number of feed IDs does not match the number of embeddings."
        )

    items_by_id = {
        str(item["feed_id"]): item
        for item in feed_data["feed_items"]
    }

    ordered_items = []
    for feed_id in feed_ids:
        item = items_by_id.get(str(feed_id))
        if item is None:
            raise ValueError(
                f"Feed ID {feed_id} is missing from personal_feed_items.json."
            )
        ordered_items.append(item)

    return ordered_items, embeddings, model_name


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_source_language(text: str) -> str | None:
    if HANGUL_PATTERN.search(text):
        return "kor_Hang"
    if HIRAGANA_KATAKANA_PATTERN.search(text):
        return "jpn_Jpan"
    if HAN_PATTERN.search(text):
        return "zho_Hant"
    return None


def load_translation_model(device: str) -> tuple[Any, Any]:
    print(
        f"Loading translation model on {device}: {TRANSLATION_MODEL}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL)
    model.to(device)
    model.eval()
    return tokenizer, model


def translate_feed_text_to_english(
    text: Any,
    feed_id: str,
    tokenizer: Any,
    translation_model: Any,
    device: str,
) -> str:
    source_text = " ".join(str(text or "").split())
    if not source_text:
        return ""

    source_language = detect_source_language(source_text)

    if source_language is None:
        return source_text

    tokenizer.src_lang = source_language
    encoded = tokenizer(
        source_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {name: value.to(device) for name, value in encoded.items()}

    with torch.inference_mode():
        generated_tokens = translation_model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_new_tokens=128,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )

    english_text = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0].strip()

    if not english_text:
        raise ValueError(
            f"Empty English text generated for feed {feed_id}."
        )
    return english_text


def select_with_mmr(
    topic_similarities: np.ndarray,
    feed_embeddings: np.ndarray,
    candidate_indices: list[int],
) -> list[tuple[int, float]]:
    selected: list[int] = []
    selected_with_scores: list[tuple[int, float]] = []

    while len(selected) < SEEDS_PER_TOPIC:
        best_index: int | None = None
        best_score = float("-inf")

        for candidate_index in candidate_indices:
            if candidate_index in selected:
                continue

            relevance = float(topic_similarities[candidate_index])
            redundancy = 0.0

            if selected:
                similarities_to_selected = (
                    feed_embeddings[selected]
                    @ feed_embeddings[candidate_index]
                )
                redundancy = float(np.max(similarities_to_selected))

                if redundancy >= MAX_PAIR_SIMILARITY:
                    continue

            mmr_score = (
                MMR_LAMBDA * relevance
                - (1.0 - MMR_LAMBDA) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = candidate_index

        if best_index is None:
            break

        selected.append(best_index)
        selected_with_scores.append((best_index, best_score))

    if len(selected_with_scores) != SEEDS_PER_TOPIC:
        raise ValueError(
            f"Could only select {len(selected_with_scores)} sufficiently "
            f"different feed seeds. Expected {SEEDS_PER_TOPIC}."
        )

    return selected_with_scores


def select_feed_seeds(topics: list[str]) -> dict[str, Any]:
    feed_items, feed_embeddings, model_name = load_embedding_data()

    device = choose_device()
    print(
        f"Loading topic embedding model on {device}: {model_name}", flush=True)
    model = SentenceTransformer(model_name, device=device)
    translation_tokenizer, translation_model = load_translation_model(device)
    topic_embeddings = model.encode(
        topics,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    topic_results = []

    for topic, topic_embedding in zip(topics, topic_embeddings):
        topic_similarities = feed_embeddings @ topic_embedding
        candidate_count = min(TOP_CANDIDATES, len(feed_items))
        candidate_indices = np.argsort(topic_similarities)[::-1][
            :candidate_count
        ].tolist()

        selected = select_with_mmr(
            topic_similarities=topic_similarities,
            feed_embeddings=feed_embeddings,
            candidate_indices=candidate_indices,
        )

        selected_seeds = []
        for feed_index, mmr_score in selected:
            item = feed_items[feed_index]
            english_text = translate_feed_text_to_english(
                item["text"],
                str(item["feed_id"]),
                translation_tokenizer,
                translation_model,
                device,
            )
            english_creator = translate_feed_text_to_english(
                item.get("creator"),
                str(item["feed_id"]),
                translation_tokenizer,
                translation_model,
                device,
            )
            selected_seeds.append(
                {
                    "feed_id": item["feed_id"],
                    "source_type": item["source_type"],
                    "text": english_text,
                    "original_text": item["text"],
                    "creator": english_creator or None,
                    "url": item.get("url"),
                    "occurred_at": item.get("occurred_at"),
                    "interaction_count": item.get("interaction_count", 1),
                    "topic_similarity": round(
                        float(topic_similarities[feed_index]),
                        6,
                    ),
                    "mmr_score": round(float(mmr_score), 6),
                }
            )

        topic_results.append(
            {
                "topic": topic,
                "selected_seeds": selected_seeds,
            }
        )

    return {
        "configuration": {
            "top_candidates": TOP_CANDIDATES,
            "seeds_per_topic": SEEDS_PER_TOPIC,
            "mmr_lambda": MMR_LAMBDA,
            "max_pair_similarity": MAX_PAIR_SIMILARITY,
            "embedding_model": model_name,
        },
        "topics": topic_results,
    }


def save_feed_seeds(result: dict[str, Any]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.{os.getpid()}.tmp"
    )

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    for attempt in range(5):
        try:
            os.replace(temporary_file, OUTPUT_FILE)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)
