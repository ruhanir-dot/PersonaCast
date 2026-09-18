import html
import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen, HTTPError

import requests
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "daily"

INPUT_FILE = DATA_DIR / "googe_search_manual.json"
RAW_OUTPUT_FILE = DATA_DIR / "google_search_profile_raw.json"
OUTPUT_FILE = DATA_DIR / "google_search_profile.json"

TRANSLATE_API_URL = (
    "https://translation.googleapis.com/language/translate/v2"
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(value):
    if value is None:
        return None

    value = str(value).strip()
    return value if value else None


def translate_batch(texts):
    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("GOOGLE_TRANSLATE_API_KEY is not set.")

    translated = [""] * len(texts)
    batch = []
    batch_chars = 0

    def request_batch(items):
        values = [text for _, text in items]

        request = Request(
            f"{TRANSLATE_API_URL}?{urlencode({'key': api_key})}",
            data=json.dumps({
                "q": values,
                "target": "en",
                "format": "text"
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urlopen(request, timeout=120) as response:
                payload = json.loads(
                    response.read().decode("utf-8")
                )

            return [
                html.unescape(item.get("translatedText", ""))
                for item in payload["data"]["translations"]
            ]

        except HTTPError as error:
            if len(items) > 1:
                middle = len(items) // 2
                return (
                    request_batch(items[:middle]) +
                    request_batch(items[middle:])
                )

            print(
                f"Translation failed: {error.code}",
                flush=True
            )
            return [values[0]]

    def flush():
        nonlocal batch, batch_chars

        if not batch:
            return

        values = request_batch(batch)

        for (index, _), value in zip(batch, values):
            translated[index] += value + " "

        batch = []
        batch_chars = 0

    for index, value in enumerate(texts):
        text = clean_text(value) or ""

        if not text:
            continue

        parts = [
            text[i:i + 3000]
            for i in range(0, len(text), 3000)
        ]

        for part in parts:
            if (
                batch and
                (
                    len(batch) >= 20 or
                    batch_chars + len(part) > 8000
                )
            ):
                flush()

            batch.append((index, part))
            batch_chars += len(part)

    flush()

    return [
        value.strip()
        for value in translated
    ]


def search_news(query):
    api_key = os.getenv("TAVILY_API_KEY", "").strip()

    if not api_key or not query or len(query.strip()) < 2:
        return []

    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "query": query,
            "topic": "news",
            "time_range": "day",
            "max_results": 3,
            "include_raw_content": True
        },
        timeout=30
    )

    if not response.ok:
        print(
            f"Tavily failed ({response.status_code}): "
            f"{response.text}",
            flush=True
        )
        return []

    return [
        {
            "title": item.get("title"),
            "text": item.get("raw_content") or item.get("content"),
            "source_url": item.get("url")
        }
        for item in response.json().get("results", [])
    ]


def main():
    source_items = load_json(INPUT_FILE)
    raw_items = []

    for index, source in enumerate(source_items, start=1):
        query = clean_text(source.get("query"))

        print(
            f"[Google Search {index}/{len(source_items)}] "
            f"Searching: {query}",
            flush=True
        )

        articles = search_news(query)

        raw_items.append({
            "item_id": f"google_search_{index:03d}",
            "source": "google_search",
            "text": " ".join(
                item.get("text", "")
                for item in articles
                if item.get("text")
            ) or query,
            "url": None,
            "creator": None,
            "query": query,
            "news_articles": articles
        })

    with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)

    targets = []

    for item in raw_items:
        if item.get("query"):
            targets.append((item, "query"))

        for article in item.get("news_articles", []):
            if article.get("title"):
                targets.append((article, "title"))

            if article.get("text"):
                targets.append((article, "text"))

    translated = translate_batch([
        container[field]
        for container, field in targets
    ])

    for (container, field), value in zip(targets, translated):
        container[field] = value

    for item in raw_items:
        article_text = " ".join(
            article.get("text", "")
            for article in item.get("news_articles", [])
            if article.get("text")
        )

        item["text"] = article_text or item.get("query")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)

    print(f"Saved raw data to {RAW_OUTPUT_FILE}", flush=True)
    print(f"Saved translated data to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
