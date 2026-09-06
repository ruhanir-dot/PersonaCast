from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from src.utils import PROJECT_ROOT


DATA_DIR = PROJECT_ROOT / "data" / "output"
DAILY_YOUTUBE_FILE = DATA_DIR / "youtube_daily_items.json"
OUTPUT_FILE = DATA_DIR / "personal_feed_items.json"

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

### English date handling! 

YOUTUBE_DATE_EN_RE = re.compile(
    r"([A-Z][a-z]{2})\s+(\d{1,2}),\s+(\d{4}),\s+"
    r"(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)\s+([A-Z]{2,5})"
)
ENGLISH_MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}
### more zone offsets for more users if needed, or ever travelling
ZONE_OFFSETS = {
    "UTC": 0, "GMT": 0,
    "EDT": -4, "EST": -5,
    "CDT": -5, "CST": -6,
    "MDT": -6, "MST": -7,
    "PDT": -7, "PST": -8,
}
DELETED_TITLES = {
    "deleted video",
    "private video",
    "已刪除的影片",
    "私人影片",
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def make_feed_id(video_id: str, url: str) -> str:
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


def match_youtube_date(text: str) -> tuple[re.Match[str], str] | None:
    """
    matching export timestamp to chinese or english region for proper parsing downstream
    """
    # depending on youtube export timestamp determine if chinese or english handling
    match = YOUTUBE_DATE_RE.search(text)
    if match:
        return match, "zh"
    match = YOUTUBE_DATE_EN_RE.search(text)
    if match:
        return match, "en"
    return None


def parse_youtube_datetime(text: str) -> str | None:
    """
    parsing youtube datetime for handling chinese and english exports
    """
    matched = match_youtube_date(text) # check using utility func

    if not matched:
        return None
    match, locale = matched

    if locale == "zh":
        year, month, day = (int(match.group(i)) for i in range(1, 4))
        period = match.group(4)
        hour = int(match.group(5))
        minute = int(match.group(6))
        second = int(match.group(7))
        if period in {"下午", "晚上"} and hour < 12:
            hour += 12
        elif period in {"上午", "凌晨"} and hour == 12:
            hour = 0
    
    else:
        month = ENGLISH_MONTHS.get(match.group(1), 0)
        if not month:
            return None
        day = int(match.group(2))
        year = int(match.group(3))
        hour = int(match.group(4))
        minute = int(match.group(5))
        second = int(match.group(6))
        period = match.group(7)
        if period == "PM" and hour < 12:
            hour += 12
        elif period == "AM" and hour == 12:
            hour = 0

    zone_name = match.group(8)

    offset_hours = ZONE_OFFSETS.get(zone_name)
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


def build_feed_items() -> dict[str, Any]:
    with DAILY_YOUTUBE_FILE.open("r", encoding="utf-8") as file:
        daily_data = json.load(file)

    source_items = daily_data.get("feed_items", [])
    if not isinstance(source_items, list):
        raise ValueError(
            "youtube_daily_items.json must contain a feed_items array.")

    feed_items: list[dict[str, Any]] = []
    for fallback_rank, source_item in enumerate(source_items, start=1):
        if not isinstance(source_item, dict):
            continue

        video_id = clean_text(source_item.get("video_id"))
        url = clean_text(source_item.get("url"))
        title = clean_text(source_item.get("title"))
        description = clean_text(source_item.get("description"))
        channel = clean_text(source_item.get("channel")
                             or source_item.get("creator"))
        if not title or not (video_id or url):
            continue

        try:
            homepage_rank = int(source_item.get(
                "homepage_rank", fallback_rank))
        except (TypeError, ValueError):
            homepage_rank = fallback_rank

        text = title if not description else f"{title}\n\n{description}"
        feed_items.append(
            {
                "feed_id": make_feed_id(video_id, url),
                "source_type": "youtube_daily_video",
                "title": title,
                "description": description,
                "channel": channel or None,
                "creator": channel or None,
                "text": text,
                "url": url or None,
                "occurred_at": source_item.get("occurred_at"),
                "occurred_at_raw": source_item.get("occurred_at_raw"),
                "homepage_rank": homepage_rank,
                "interaction_count": 1,
            }
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
        matched = match_youtube_date(block_text)
        return matched[0].group(0) if matched else ""


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

    feed_items.sort(key=lambda item: int(item["homepage_rank"]))
    return {
        "metadata": {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "total_feed_items": len(feed_items),
            "source_counts": {"youtube_daily_video": len(feed_items)},
        },
        "feed_items": feed_items,
    }


def main() -> None:
    result = build_feed_items()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(
        f"Saved {len(result['feed_items'])} YouTube homepage videos to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
