from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, TextIO


try:
    from src.utils import PROJECT_ROOT
except ImportError:
    # This fallback works when the file is placed in src/user_profile/.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


# The Instagram and YouTube exports are already extracted under data/.
# No ZIP file is required.
DEFAULT_INPUT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "output" / "personal_feed_items.json"

INSTAGRAM_LIKES = ("instagram_export/likes/liked_posts.json",)
YOUTUBE_WATCH = ("youtube_history/watch_history/watch-history.html",)
YOUTUBE_SEARCH = ("youtube_history/watch_history/search-history.html",)
YOUTUBE_SUBSCRIPTIONS = (
    "youtube_history/subscriptions/subscriptions.csv",
    "youtube_history/subscriptions/subscriptions.csv.csv",
)

SPACE_RE = re.compile(r"\s+")
YOUTUBE_DATE_RE = re.compile(
    r"(\d{4})年(\d{1,2})月(\d{1,2})日\s+"
    r"(上午|下午|晚上|凌晨)(\d{1,2}):(\d{2}):(\d{2})\s+([A-Z]{2,5})"
)
DELETED_TITLES = {
    "deleted video",
    "private video",
    "已刪除的影片",
    "私人影片",
}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", html.unescape(str(value))).strip()


def repair_mojibake(value: str) -> str:
    """Repair UTF-8 byte sequences that Instagram stored as Latin-1 text."""
    # Do not normalize whitespace before repairing. Some mojibake continuation
    # bytes (for example U+00A0) are classified as whitespace by Python.
    value = html.unescape(str(value or ""))
    suspicious = ("Ã", "Â", "â", "å", "æ", "ç", "ä", "é")
    if not any(mark in value for mark in suspicious):
        return clean_text(value)

    repaired_parts: list[str] = []
    index = 0

    while index < len(value):
        byte = ord(value[index])
        if byte > 255 or byte < 128:
            repaired_parts.append(value[index])
            index += 1
            continue

        if 0xC2 <= byte <= 0xDF:
            sequence_length = 2
        elif 0xE0 <= byte <= 0xEF:
            sequence_length = 3
        elif 0xF0 <= byte <= 0xF4:
            sequence_length = 4
        else:
            repaired_parts.append(value[index])
            index += 1
            continue

        candidate = value[index: index + sequence_length]
        candidate_bytes = [ord(character) for character in candidate]
        valid_continuations = (
            len(candidate_bytes) == sequence_length
            and all(number <= 255 for number in candidate_bytes)
            and all(0x80 <= number <= 0xBF for number in candidate_bytes[1:])
        )

        if not valid_continuations:
            repaired_parts.append(value[index])
            index += 1
            continue

        try:
            repaired_parts.append(bytes(candidate_bytes).decode("utf-8"))
            index += sequence_length
        except UnicodeDecodeError:
            repaired_parts.append(value[index])
            index += 1

    repaired = "".join(repaired_parts)

    # Only byte sequences that form valid UTF-8 are converted, so mixed text
    # containing English, Korean, emoji, and mojibake can be repaired safely.
    return clean_text(repaired)


def make_feed_id(source_type: str, stable_key: str) -> str:
    digest = hashlib.sha256(
        f"{source_type}|{stable_key}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{source_type}_{digest}"


def normalize_key(value: str) -> str:
    return clean_text(value).casefold()


def unix_timestamp_to_iso(value: Any) -> str | None:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def parse_youtube_datetime(text: str) -> str | None:
    match = YOUTUBE_DATE_RE.search(text)
    if not match:
        return None

    year, month, day = (int(match.group(i)) for i in range(1, 4))
    period = match.group(4)
    hour = int(match.group(5))
    minute = int(match.group(6))
    second = int(match.group(7))
    zone_name = match.group(8)

    if period in {"下午", "晚上"} and hour < 12:
        hour += 12
    elif period in {"上午", "凌晨"} and hour == 12:
        hour = 0

    zone_offsets = {
        "PDT": -7,
        "PST": -8,
        "UTC": 0,
        "GMT": 0,
    }
    offset_hours = zone_offsets.get(zone_name)
    if offset_hours is None:
        return None

    tz = timezone(timedelta(hours=offset_hours))
    return datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
        tzinfo=tz,
    ).isoformat()


class InputBundle:
    """Find the required export files inside an extracted data directory."""

    def __init__(self, input_path: Path) -> None:
        self.input_path = input_path.resolve()
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: {self.input_path}"
            )
        if not self.input_path.is_dir():
            raise NotADirectoryError(
                "Input must be an extracted directory, not a ZIP file: "
                f"{self.input_path}"
            )

    def _find_local_file(self, suffixes: tuple[str, ...]) -> Path:
        for suffix in suffixes:
            normalized_suffix = suffix.replace("\\", "/")
            basename = Path(normalized_suffix).name
            for candidate in self.input_path.rglob(basename):
                normalized_path = candidate.as_posix()
                if normalized_path.endswith(normalized_suffix):
                    return candidate
        raise FileNotFoundError(
            f"Could not find any of {suffixes} under {self.input_path}"
        )

    def open_text(self, suffixes: tuple[str, ...]) -> TextIO:
        path = self._find_local_file(suffixes)
        return path.open(
            "r",
            encoding="utf-8-sig",
            errors="replace",
            newline="",
        )


class FeedStore:
    def __init__(self) -> None:
        self._items: dict[tuple[str, str], dict[str, Any]] = {}
        self.skipped: Counter[str] = Counter()

    def add(
        self,
        *,
        source_type: str,
        stable_key: str,
        text: str,
        creator: str = "",
        url: str = "",
        occurred_at: str | None = None,
        occurred_at_raw: str = "",
    ) -> None:
        text = clean_text(text)
        creator = clean_text(creator)
        url = clean_text(url)
        stable_key = clean_text(stable_key)

        if not text:
            self.skipped["empty_text"] += 1
            return
        if normalize_key(text) in DELETED_TITLES:
            self.skipped["deleted_or_private"] += 1
            return
        if not stable_key:
            stable_key = f"{text}|{creator}"

        key = (source_type, normalize_key(stable_key))
        existing = self._items.get(key)
        if existing:
            existing["interaction_count"] += 1
            self.skipped["duplicate_interaction_merged"] += 1
            return

        self._items[key] = {
            "feed_id": make_feed_id(source_type, stable_key),
            "source_type": source_type,
            "text": text,
            "creator": creator or None,
            "url": url or None,
            "occurred_at": occurred_at,
            "occurred_at_raw": clean_text(occurred_at_raw) or None,
            "interaction_count": 1,
        }

    def items(self) -> list[dict[str, Any]]:
        items = list(self._items.values())
        items.sort(
            key=lambda item: item.get("occurred_at") or "",
            reverse=True,
        )
        return items


def find_nested_label(value: Any, label_name: str) -> str:
    if isinstance(value, dict):
        if value.get("label") == label_name:
            # Return the raw value. Instagram captions may contain mojibake
            # bytes that must be repaired before whitespace normalization.
            return str(value.get("value") or "")
        for child in value.values():
            found = find_nested_label(child, label_name)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_nested_label(child, label_name)
            if found:
                return found
    return ""


def parse_instagram_likes(bundle: InputBundle, store: FeedStore) -> None:
    with bundle.open_text(INSTAGRAM_LIKES) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Instagram liked_posts.json must contain a JSON array.")

    for record in records:
        label_values = record.get("label_values") or []
        caption = repair_mojibake(find_nested_label(label_values, "Caption"))
        title = repair_mojibake(find_nested_label(label_values, "Title"))
        url = find_nested_label(label_values, "URL")
        username = find_nested_label(label_values, "Username")
        owner_name = repair_mojibake(find_nested_label(label_values, "Name"))
        creator = username or owner_name
        text = caption or title

        store.add(
            source_type="instagram_like",
            stable_key=url or str(record.get("fbid") or ""),
            text=text,
            creator=creator,
            url=url,
            occurred_at=unix_timestamp_to_iso(record.get("timestamp")),
        )


class YouTubeTakeoutParser(HTMLParser):
    """Stream Google Takeout HTML and emit one item per outer activity cell."""

    def __init__(self, mode: str, store: FeedStore) -> None:
        super().__init__(convert_charrefs=True)
        if mode not in {"watch", "search"}:
            raise ValueError(f"Unsupported YouTube parser mode: {mode}")
        self.mode = mode
        self.store = store
        self.capture_depth = 0
        self.text_parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self.current_link: dict[str, Any] | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attrs_dict = dict(attrs)

        if tag == "div":
            classes = set((attrs_dict.get("class") or "").split())
            if self.capture_depth == 0 and "outer-cell" in classes:
                self.capture_depth = 1
                self.text_parts = []
                self.links = []
                self.current_link = None
                return
            if self.capture_depth > 0:
                self.capture_depth += 1

        if self.capture_depth > 0 and tag == "a":
            self.current_link = {
                "href": attrs_dict.get("href") or "",
                "text_parts": [],
            }

    def handle_data(self, data: str) -> None:
        if self.capture_depth == 0:
            return
        self.text_parts.append(data)
        if self.current_link is not None:
            self.current_link["text_parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.capture_depth == 0:
            return

        if tag == "a" and self.current_link is not None:
            self.links.append(
                {
                    "href": clean_text(self.current_link["href"]),
                    "text": clean_text(" ".join(self.current_link["text_parts"])),
                }
            )
            self.current_link = None

        if tag == "div":
            self.capture_depth -= 1
            if self.capture_depth == 0:
                self._finish_activity()

    def _finish_activity(self) -> None:
        block_text = clean_text(" ".join(self.text_parts))

        if self.mode == "watch":
            if "From Google Ads" in block_text:
                self.store.skipped["youtube_google_ad"] += 1
                return

            video_link = next(
                (
                    link
                    for link in self.links
                    if "youtube.com/watch" in link["href"]
                    or "youtu.be/" in link["href"]
                ),
                None,
            )
            if not video_link:
                self.store.skipped["youtube_watch_without_video"] += 1
                return

            channel_link = next(
                (
                    link
                    for link in self.links
                    if "youtube.com/channel/" in link["href"]
                    or "youtube.com/@" in link["href"]
                ),
                None,
            )
            occurred_at = parse_youtube_datetime(block_text)
            self.store.add(
                source_type="youtube_watch",
                stable_key=video_link["href"],
                text=video_link["text"],
                creator=channel_link["text"] if channel_link else "",
                url=video_link["href"],
                occurred_at=occurred_at,
                occurred_at_raw=self._raw_date(block_text),
            )
            return

        search_link = next(
            (
                link
                for link in self.links
                if "youtube.com/results?search_query=" in link["href"]
            ),
            None,
        )
        if not search_link:
            self.store.skipped["youtube_search_without_query"] += 1
            return

        occurred_at = parse_youtube_datetime(block_text)
        self.store.add(
            source_type="youtube_search",
            stable_key=search_link["text"],
            text=search_link["text"],
            url=search_link["href"],
            occurred_at=occurred_at,
            occurred_at_raw=self._raw_date(block_text),
        )

    @staticmethod
    def _raw_date(block_text: str) -> str:
        match = YOUTUBE_DATE_RE.search(block_text)
        return match.group(0) if match else ""


def parse_youtube_html(
    bundle: InputBundle,
    suffixes: tuple[str, ...],
    mode: str,
    store: FeedStore,
) -> None:
    parser = YouTubeTakeoutParser(mode=mode, store=store)
    with bundle.open_text(suffixes) as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            parser.feed(chunk)
    parser.close()


def parse_youtube_subscriptions(
    bundle: InputBundle,
    store: FeedStore,
) -> None:
    with bundle.open_text(YOUTUBE_SUBSCRIPTIONS) as file:
        reader = csv.DictReader(file)
        for row in reader:
            values = [clean_text(value) for value in row.values()]
            channel_id = values[0] if len(values) > 0 else ""
            channel_url = values[1] if len(values) > 1 else ""
            channel_name = values[2] if len(values) > 2 else ""
            store.add(
                source_type="youtube_subscription",
                stable_key=channel_id or channel_url or channel_name,
                text=channel_name,
                creator=channel_name,
                url=channel_url,
            )


def build_feed_items(input_path: Path) -> dict[str, Any]:
    bundle = InputBundle(input_path)
    store = FeedStore()

    print("Parsing Instagram liked posts...", flush=True)
    parse_instagram_likes(bundle, store)

    print("Parsing YouTube watch history...", flush=True)
    parse_youtube_html(bundle, YOUTUBE_WATCH, "watch", store)

    print("Parsing YouTube search history...", flush=True)
    parse_youtube_html(bundle, YOUTUBE_SEARCH, "search", store)

    print("Parsing YouTube subscriptions...", flush=True)
    parse_youtube_subscriptions(bundle, store)

    items = store.items()
    source_counts = Counter(item["source_type"] for item in items)

    return {
        "metadata": {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_name": input_path.name,
            "total_feed_items": len(items),
            "source_counts": dict(sorted(source_counts.items())),
            "skipped_or_merged": dict(sorted(store.skipped.items())),
            "privacy": {
                "included": [
                    "Instagram liked-post captions, owners, URLs, and timestamps",
                    "YouTube watch titles, channels, URLs, and timestamps",
                    "YouTube search queries and timestamps",
                    "YouTube subscriptions",
                ],
                "excluded": [
                    "Google Ads watch events",
                    "emails",
                    "exact locations",
                    "device information",
                    "profile information",
                    "private comments and messages",
                ],
            },
        },
        "feed_items": items,
    }


def save_json(result: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert raw Instagram and YouTube exports into unified feed items."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Path to the extracted Instagram/YouTube data directory "
            f"(default: {DEFAULT_INPUT})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_feed_items(args.input)
    save_json(result, args.output)

    metadata = result["metadata"]
    print(
        "Completed: "
        f"{metadata['total_feed_items']} unified feed items.\n"
        f"Source counts: {metadata['source_counts']}\n"
        f"Saved to: {args.output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
