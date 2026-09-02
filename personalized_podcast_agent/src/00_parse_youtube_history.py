from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from html import unescape

from utils import DATA_DIR, OUTPUT_DIR, write_json_file


YOUTUBE_HISTORY_DIR = DATA_DIR / "youtube_history"
YOUTUBE_WATCH_DIR = YOUTUBE_HISTORY_DIR / "watch_history"
YOUTUBE_SUBSCRIPTIONS_DIR = YOUTUBE_HISTORY_DIR / "subscriptions"
YOUTUBE_PROFILE_FILE = OUTPUT_DIR / "youtube_profile.json"


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "from",
    "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "official", "video", "videos", "youtube", "watch", "watched", "shorts", "short",
    "episode", "part", "full", "new", "latest", "live", "stream", "clip", "clips",
    "reaction", "review", "how", "what", "why", "when", "where", "who", "can",
    "will", "would", "should", "your",
}

def clean_text(text: str) -> str:
    text = unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def extract_links_from_block(block: str) -> list[dict[str, str]]:
    pattern = r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
    links = []

    for url, text in re.findall(pattern, block, flags=re.IGNORECASE | re.DOTALL):
        links.append(
            {
                "url": unescape(url),
                "text": clean_text(text),
            }
        )

    return links


def iter_takeout_blocks_fast(path: Path, max_items: int = 50):
    """
    Fast parser for Google Takeout HTML.

    It does not parse the whole HTML DOM.
    It scans text and yields activity blocks one by one.
    """
    if not path or not path.exists():
        return

    buffer = ""
    count = 0
    inside = False

    print(f"Reading HTML file: {path}")
    print(f"File size: {path.stat().st_size / (1024 * 1024):.2f} MB")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no % 50000 == 0:
                print(f"Scanned {line_no} lines, collected {count} items...")

            if "outer-cell" in line or "content-cell" in line:
                inside = True
                buffer = line
            elif inside:
                buffer += line

            if inside and ("</div>" in line):
                links = extract_links_from_block(buffer)
                raw_text = clean_text(buffer)

                if links or raw_text:
                    yield {
                        "links": links,
                        "raw_text": raw_text,
                    }

                    count += 1

                    if count >= max_items:
                        print(f"Reached max_items={max_items}. Stop scanning.")
                        return

                inside = False
                buffer = ""


def extract_watch_history_from_html(path: Path | None, max_items: int = 50) -> list[dict[str, str]]:
    records = []

    if not path or not path.exists():
        print("Watch history file not found.")
        return records

    for item in iter_takeout_blocks_fast(path, max_items=max_items):
        links = item.get("links", [])
        raw_text = item.get("raw_text", "")

        title = ""
        url = ""
        channel = ""

        if links:
            title = links[0].get("text", "")
            url = links[0].get("url", "")

        if len(links) >= 2:
            channel = links[1].get("text", "")

        if not title:
            title = raw_text
            title = re.sub(r"^Watched\s+", "", title, flags=re.IGNORECASE)
            title = re.sub(r"^觀看了\s*", "", title)
            title = clean_text(title)

        if title:
            records.append(
                {
                    "title": title,
                    "channel": channel,
                    "url": url,
                    "raw_text": raw_text,
                    "source": "youtube_watch_history",
                }
            )

        if len(records) >= max_items:
            break

    return records


def extract_search_history_from_html(path: Path | None, max_items: int = 50) -> list[dict[str, str]]:
    records = []

    if not path or not path.exists():
        print("Search history file not found.")
        return records

    for item in iter_takeout_blocks_fast(path, max_items=max_items):
        links = item.get("links", [])
        raw_text = item.get("raw_text", "")

        query = ""
        url = ""

        if links:
            query = links[0].get("text", "")
            url = links[0].get("url", "")

        if not query:
            query = raw_text
            query = re.sub(r"^Searched for\s+", "", query, flags=re.IGNORECASE)
            query = re.sub(r"^搜尋了\s*", "", query)
            query = re.sub(r"^已搜尋\s*", "", query)
            query = clean_text(query)

        if query:
            records.append(
                {
                    "query": query,
                    "url": url,
                    "raw_text": raw_text,
                    "source": "youtube_search_history",
                }
            )

        if len(records) >= max_items:
            break

    return records


def extract_subscriptions_from_csv(max_items: int = 30) -> list[dict[str, str]]:
    candidate_files = [
        YOUTUBE_SUBSCRIPTIONS_DIR / "subscriptions.csv",
        YOUTUBE_SUBSCRIPTIONS_DIR / "subscriptions.csv.csv",
        YOUTUBE_HISTORY_DIR / "subscriptions.csv",
        YOUTUBE_HISTORY_DIR / "subscriptions.csv.csv",
    ]

    path = find_first_existing(candidate_files)

    if path is None:
        print("Subscription file not found.")
        return []

    print(f"Reading subscriptions file: {path}")

    records = []

    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        for row in reader:
            channel_name = ""
            channel_url = ""
            channel_id = ""

            for key in fieldnames:
                lower_key = key.lower()
                value = clean_text(row.get(key, ""))

                if not value:
                    continue

                if "title" in lower_key or "channel" in lower_key or "name" in lower_key:
                    if not channel_name:
                        channel_name = value
                elif "url" in lower_key:
                    channel_url = value
                elif "id" in lower_key:
                    channel_id = value

            if not channel_name:
                for value in row.values():
                    value = clean_text(value or "")
                    if value and not value.startswith("http"):
                        channel_name = value
                        break

            if channel_name:
                records.append(
                    {
                        "channel": channel_name,
                        "url": channel_url,
                        "channel_id": channel_id,
                        "source": "youtube_subscriptions",
                    }
                )

            if len(records) >= max_items:
                break

    return records


def keywordize(text: str) -> list[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\s]", " ", text)
    words = [w.strip() for w in text.split() if w.strip()]

    keywords = []

    for word in words:
        if len(word) <= 2:
            continue

        if word in STOPWORDS:
            continue

        keywords.append(word)

    return keywords


def build_youtube_profile(max_items: int = 50) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    watch_path = find_first_existing(
        [
            YOUTUBE_WATCH_DIR / "watch-history.html",
            YOUTUBE_WATCH_DIR / "watch_history.html",
            YOUTUBE_HISTORY_DIR / "watch-history.html",
            YOUTUBE_HISTORY_DIR / "watch_history.html",
        ]
    )

    search_path = find_first_existing(
        [
            YOUTUBE_WATCH_DIR / "search-history.html",
            YOUTUBE_WATCH_DIR / "search_history.html",
            YOUTUBE_HISTORY_DIR / "search-history.html",
            YOUTUBE_HISTORY_DIR / "search_history.html",
        ]
    )

    print("YouTube parser started.")
    print(f"Watch path: {watch_path}")
    print(f"Search path: {search_path}")

    print("Extracting watch history...")
    watch_records = extract_watch_history_from_html(watch_path, max_items=max_items)

    print("Extracting search history...")
    search_records = extract_search_history_from_html(search_path, max_items=max_items)

    print("Extracting subscriptions...")
    subscription_records = extract_subscriptions_from_csv(max_items=30)

    keyword_counter = Counter()
    channel_counter = Counter()

    for record in watch_records:
        keyword_counter.update(keywordize(record.get("title", "")))

        channel = record.get("channel", "")

        if channel:
            channel_counter[channel] += 1

    for record in search_records:
        keyword_counter.update(keywordize(record.get("query", "")))

    for record in subscription_records:
        channel = record.get("channel", "")

        if channel:
            channel_counter[channel] += 1
            keyword_counter.update(keywordize(channel))

    profile = {
        "source": "youtube_takeout",
        "watch_history_file_found": str(watch_path) if watch_path else None,
        "search_history_file_found": str(search_path) if search_path else None,
        "watch_history_count": len(watch_records),
        "search_history_count": len(search_records),
        "subscription_count": len(subscription_records),
        "top_keywords": [word for word, _ in keyword_counter.most_common(50)],
        "top_channels": [channel for channel, _ in channel_counter.most_common(30)],
        "recent_watch_titles": [item.get("title", "") for item in watch_records[:50]],
        "recent_search_queries": [item.get("query", "") for item in search_records[:50]],
        "subscriptions": [item.get("channel", "") for item in subscription_records[:50]],
    }

    write_json_file(YOUTUBE_PROFILE_FILE, profile)

    print(f"Saved YouTube profile to {YOUTUBE_PROFILE_FILE}")
    print(f"Watch history count: {profile['watch_history_count']}")
    print(f"Search history count: {profile['search_history_count']}")
    print(f"Subscription count: {profile['subscription_count']}")
    print("Top YouTube keywords:", json.dumps(profile["top_keywords"][:10], ensure_ascii=True))

    return profile


if __name__ == "__main__":
    build_youtube_profile(max_items=50)