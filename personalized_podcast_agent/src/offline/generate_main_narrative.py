from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote_plus, urlparse

import feedparser
import numpy as np
import requests
import trafilatura
from googlenewsdecoder import gnewsdecoder
from sentence_transformers import SentenceTransformer

from src.utils import llm_json


ARTICLE_MIN_WORDS = 100
ARTICLE_MAX_CHARACTERS = 6000
SOURCE_RELEVANCE_TEXT_CHARACTERS = 2000
SOURCE_RELEVANCE_THRESHOLD = 0.30
SOURCE_RELEVANCE_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    )
}

_source_relevance_model: SentenceTransformer | None = None


def get_source_relevance_model() -> SentenceTransformer:
    global _source_relevance_model

    if _source_relevance_model is None:
        print(
            "Loading source relevance model on cpu: "
            f"{SOURCE_RELEVANCE_MODEL}",
            flush=True,
        )
        _source_relevance_model = SentenceTransformer(
            SOURCE_RELEVANCE_MODEL,
            device="cpu",
        )

    return _source_relevance_model


def build_relevance_reference(query: str) -> str:
    """Keep the topic and seed while removing search-only boilerplate."""
    reference = re.sub(
        r"\b(?:latest|recent|news|updates?)\b",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    return " ".join(reference.split()) or query.strip()


def source_relevance_score(
    reference_embedding: np.ndarray,
    source_item: dict[str, Any],
) -> float:
    title = str(source_item.get("title") or "").strip()
    article_text = str(source_item.get("article_text") or "").strip()
    if not title or not article_text:
        return 0.0

    title_and_content = (
        f"{title}. "
        f"{article_text[:SOURCE_RELEVANCE_TEXT_CHARACTERS]}"
    )
    model = get_source_relevance_model()
    source_embeddings = model.encode(
        [title, title_and_content],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    title_similarity = float(source_embeddings[0] @ reference_embedding)
    content_similarity = float(source_embeddings[1] @ reference_embedding)
    return 0.65 * title_similarity + 0.35 * content_similarity


def generate_segments(
    *,
    current_script: str,
    predicted_questions: list[dict[str, Any]],
) -> list[str]:
    scripts: list[str] = []

    for predicted in predicted_questions:
        question = str(predicted["question"]).strip()

        result = llm_json(
            prompt=(
                "Current podcast content:\n"
                f"{current_script}\n\n"
                "User question:\n"
                f"{question}\n\n"
                "Write the next podcast segment that directly answers the "
                "user question.\n\n"
                "Requirements:\n"
                "- Answer the question immediately in the first sentence.\n"
                "- Write 45 to 60 words in exactly one paragraph.\n"
                "- Introduce new information that was not already explained "
                "in the current podcast content.\n"
                "- Do not copy, repeat, summarize, or closely paraphrase the "
                "current podcast content.\n"
                "- Do not repeat or paraphrase the user question.\n"
                "- Do not output another question.\n"
                "- Do not use phrases such as 'In this podcast', "
                "'In today's episode', 'Join us', or 'Stay tuned'.\n"
                "- Do not include introductions, conclusions, headings, "
                "hashtags, emojis, or source descriptions.\n"
                "- Use informative declarative sentences.\n"
                "- Return JSON only."
            ),
            system=(
                "You answer a user question with the next podcast segment. "
                "The script must be an answer, not a question."
            ),
            temperature=0.1,
            max_output_tokens=180,
            json_schema={
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "maxLength": 700,
                    }
                },
                "required": ["script"],
                "additionalProperties": False,
            },
        )

        scripts.append(str(result["script"]).strip())

    return scripts


def decode_article_url(url: str) -> str | None:
    domain = urlparse(url).netloc.casefold()
    if "news.google.com" not in domain:
        return url

    try:
        decoded_result = gnewsdecoder(url, interval=0.5)
    except Exception as error:
        print(f"Could not decode Google News URL: {error}", flush=True)
        return None

    if not decoded_result.get("status"):
        print(
            "Could not decode Google News URL: "
            f"{decoded_result.get('message', 'unknown error')}",
            flush=True,
        )
        return None

    decoded_url = str(decoded_result.get("decoded_url") or "").strip()
    return decoded_url or None


def extract_article_content(
    source_item: dict[str, Any],
) -> dict[str, Any] | None:
    google_news_url = str(source_item.get("url") or "").strip()
    if not google_news_url:
        return None

    article_url = decode_article_url(google_news_url)
    if not article_url:
        return None

    try:
        response = requests.get(
            article_url,
            headers=REQUEST_HEADERS,
            timeout=20,
            allow_redirects=True,
        )
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"Could not download article: {error}", flush=True)
        return None

    final_url = response.url
    final_domain = urlparse(final_url).netloc.casefold()
    if "news.google.com" in final_domain:
        return None

    article_text = trafilatura.extract(
        response.text,
        url=final_url,
        include_comments=False,
        include_tables=False,
        favor_precision=True,
    )
    if not article_text:
        return None

    article_text = " ".join(article_text.split())
    article_word_count = len(article_text.split())
    if article_word_count < ARTICLE_MIN_WORDS:
        return None

    enriched_source = dict(source_item)
    enriched_source["google_news_url"] = google_news_url
    enriched_source["url"] = final_url
    enriched_source["article_text"] = article_text[
        :ARTICLE_MAX_CHARACTERS
    ]
    enriched_source["article_word_count"] = article_word_count
    return enriched_source


def retrieve_recommended_contents(
    query: str,
    max_news: int = 5,
    max_youtube: int = 2,
) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []

    try:
        relevance_reference = build_relevance_reference(query)
        relevance_model = get_source_relevance_model()
        reference_embedding = relevance_model.encode(
            [relevance_reference],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        encoded_query = quote_plus(query)
        news_url = (
            "https://news.google.com/rss/search"
            f"?q={encoded_query}"
            "&hl=en-US"
            "&gl=US"
            "&ceid=US:en"
        )

        feed = feedparser.parse(news_url)

        for entry in feed.entries:
            source_item = {
                "source_type": "news",
                "title": str(entry.get("title", "")).strip(),
                "summary": str(entry.get("summary", "")).strip(),
                "url": str(entry.get("link", "")).strip(),
                "published_at": str(entry.get("published", "")).strip(),
            }

            enriched_source = extract_article_content(source_item)
            if enriched_source is None:
                continue

            relevance_score = source_relevance_score(
                reference_embedding,
                enriched_source,
            )
            if relevance_score < SOURCE_RELEVANCE_THRESHOLD:
                print(
                    "Skipped irrelevant article "
                    f"(score={relevance_score:.3f}): "
                    f"{enriched_source.get('title', '')}",
                    flush=True,
                )
                continue

            enriched_source["relevance_score"] = round(
                relevance_score,
                6,
            )

            contents.append(enriched_source)
            if len(contents) >= max_news:
                break

    except Exception as exc:
        print(f"Google News retrieval failed: {exc}", flush=True)

    print(
        f"Retrieved {len(contents)} contents for query: {query}",
        flush=True,
    )

    return contents
