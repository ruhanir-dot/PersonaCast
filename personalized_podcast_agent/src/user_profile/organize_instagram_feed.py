import html
import json
import os
import re
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data" / "daily"

INPUT_FILE = DATA_DIR / "instagram_feed_manual_json"
RAW_OUTPUT_FILE = DATA_DIR / "instagram_profile_raw.json"
OUTPUT_FILE = DATA_DIR / "instagram_profile.json"

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


def clean_caption(value):
    value = clean_text(value)
    if not value:
        return None

    value = re.sub(
        r"^\s*[\d,.]+(?:[KMB])?\s+likes?,\s*"
        r"[\d,.]+(?:[KMB])?\s+comments?\s*-\s*"
        r"[^:\n]+?\s+on\s+[^:]+:\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = value.strip()
    if value.startswith('"'):
        value = value[1:]
        if value.endswith('".'):
            value = value[:-2] + "."
        elif value.endswith('"'):
            value = value[:-1]

    return clean_text(value)


def build_text(caption, hashtags):
    parts = [clean_text(caption)]
    parts.extend(
        clean_text(tag)
        for tag in hashtags
        if clean_text(tag)
    )
    return " ".join(part for part in parts if part)


def translate_batch(texts):
    if not texts:
        return []

    api_key = os.getenv("GOOGLE_TRANSLATE_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("GOOGLE_TRANSLATE_API_KEY is not set.")

    translated = []

    for start in range(0, len(texts), 128):
        batch = texts[start:start + 128]

        print(
            f"[Instagram translation] "
            f"{start + 1}-{start + len(batch)} / {len(texts)}",
            flush=True
        )

        request = Request(
            f"{TRANSLATE_API_URL}?{urlencode({'key': api_key})}",
            data=json.dumps({
                "q": batch,
                "target": "en",
                "format": "text"
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))

        translated.extend(
            html.unescape(
                item.get("translatedText", "")
            )
            for item in payload["data"]["translations"]
        )

    return translated


def get_instagram_content(url):
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        meta = soup.find(
            "meta",
            attrs={"property": "og:description"}
        )

        caption = meta.get("content") if meta else None
        username = None

        for script in soup.select('script[type="application/ld+json"]'):
            try:
                data = json.loads(script.get_text())

                if isinstance(data, dict):
                    data = data.get("@graph", [data])

                if not isinstance(data, list):
                    continue

                for item in data:
                    if not isinstance(item, dict):
                        continue

                    caption = (
                        item.get("caption")
                        or item.get("description")
                        or caption
                    )

                    author = item.get("author")

                    if isinstance(author, dict):
                        username = author.get("name")
                    elif isinstance(author, str):
                        username = author

            except Exception:
                continue

        caption = clean_caption(caption)

        hashtags = list(dict.fromkeys(
            re.findall(r"#[\w]+", caption or "", re.UNICODE)
        ))

        return {
            "caption": clean_text(caption),
            "hashtags": hashtags,
            "username": clean_text(username)
        }

    except Exception as error:
        print(f"Instagram failed: {error}", flush=True)

        return {
            "caption": None,
            "hashtags": [],
            "username": None
        }


def main():
    source_items = load_json(INPUT_FILE)
    raw_items = []

    for index, source in enumerate(source_items, start=1):
        print(
            f"[Instagram {index}/{len(source_items)}] Fetching...",
            flush=True
        )

        url = source.get("url")

        if not url or url.rstrip("/") == "https://www.instagram.com/reels":
            continue

        data = get_instagram_content(url)

        caption = clean_caption(
            data.get("caption")
            or source.get("text")
            or source.get("title")
        )
        hashtags = data.get("hashtags", [])
        text = build_text(caption, hashtags)

        if not text:
            print(
                f"[Instagram {index}/{len(source_items)}] Skipped: no text", flush=True)
            continue

        raw_items.append({
            "item_id": f"instagram_{index:03d}",
            "source": "instagram",
            "text": text,
            "url": url,
            "creator": clean_text(data.get("username")),
            "caption": clean_text(caption),
            "hashtags": hashtags
        })

    with open(RAW_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)

    captions = [
        item["caption"]
        for item in raw_items
        if item.get("caption")
    ]

    translated = translate_batch(captions)
    translated_index = 0

    for item in raw_items:
        if item.get("caption"):
            item["caption"] = translated[translated_index]
            translated_index += 1

        item["text"] = build_text(
            item.get("caption"),
            item.get("hashtags", [])
        )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(raw_items, f, ensure_ascii=False, indent=2)

    print(f"Saved raw data to {RAW_OUTPUT_FILE}", flush=True)
    print(f"Saved translated data to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
