from pathlib import Path
import json
import re
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "data" / "instagram_export"
OUTPUT_FILE = ROOT / "data" / "user_ig_profile.txt"


ALLOW_FILES = {
    "liked_posts.json",
    "post_comments_1.json",
    "reels_comments.json",
    "hype.json",
    "posts.json",
    "posts_1.json",
    "reposts.json",
    "stories.json",
    "other_content.json",
}


SKIP_FILES = {
    "possible_emails.json",
    "locations_of_interest.json",
    "profile_based_in.json",
    "instagram_profile_information.json",
    "camera_information.json",
    "instagram_friend_map.json",
    "personal_information.json",
    "profile_changes.json",
    "profile_photos.json",
}


def fix_instagram_encoding(text: str) -> str:
  
    if not isinstance(text, str):
        return ""

    text = text.strip()

    if not text:
        return ""

    try:
        fixed = text.encode("latin1").decode("utf-8")
        if any("\u4e00" <= ch <= "\u9fff" for ch in fixed) or "ð" not in fixed:
            return fixed.strip()
    except Exception:
        pass

    return text.strip()


def clean_text(text: str) -> str:
    text = fix_instagram_encoding(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    # 去掉太短或太像純符號的東西
    if len(text) < 3:
        return ""

    if re.fullmatch(r"[\W_]+", text):
        return ""

    return text


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin1") as f:
            return json.load(f)


def extract_label_values(obj, source_file, results):
    """
    針對 Instagram export 常見格式：
    label_values: [
        {"label": "Caption", "value": "..."},
        {"label": "URL", "value": "..."},
        {"title": "Hashtags", "dict": [...]}
    ]

    string_map_data: {
        "Comment": {"value": "..."},
        "Media Owner": {"value": "..."}
    }
    """
    if isinstance(obj, dict):
        # timestamp
        timestamp = obj.get("timestamp")
        if timestamp is None:
            timestamp = obj.get("creation_timestamp")

        # label_values 格式
        if "label_values" in obj and isinstance(obj["label_values"], list):
            for item in obj["label_values"]:
                if not isinstance(item, dict):
                    continue

                label = item.get("label", "")
                value = item.get("value", "")

                if label in ["Caption", "Title", "Text", "Name", "Username"]:
                    text = clean_text(value)
                    if text:
                        results.append({
                            "source": source_file,
                            "type": label,
                            "text": text,
                            "timestamp": timestamp
                        })

                # hashtags
                if item.get("title") == "Hashtags":
                    hashtags = extract_hashtags(item)
                    for tag in hashtags:
                        results.append({
                            "source": source_file,
                            "type": "Hashtag",
                            "text": tag,
                            "timestamp": timestamp
                        })

                # owner / author
                if item.get("title") in ["Owner", "Author"]:
                    owners = extract_owner_names(item)
                    for owner in owners:
                        results.append({
                            "source": source_file,
                            "type": "Owner",
                            "text": owner,
                            "timestamp": timestamp
                        })

        # string_map_data 格式
        if "string_map_data" in obj and isinstance(obj["string_map_data"], dict):
            smd = obj["string_map_data"]

            for key in ["Comment", "Media Owner", "Title", "Caption"]:
                if key in smd and isinstance(smd[key], dict):
                    value = smd[key].get("value", "")
                    text = clean_text(value)
                    if text:
                        results.append({
                            "source": source_file,
                            "type": key,
                            "text": text,
                            "timestamp": timestamp
                        })

        # media 裡面有時候 title 會藏 caption
        if "media" in obj and isinstance(obj["media"], list):
            for media_item in obj["media"]:
                if isinstance(media_item, dict):
                    title = clean_text(media_item.get("title", ""))
                    if title:
                        results.append({
                            "source": source_file,
                            "type": "MediaTitle",
                            "text": title,
                            "timestamp": media_item.get("creation_timestamp", timestamp)
                        })

                    metadata = media_item.get("media_metadata", {})
                    video_metadata = metadata.get("video_metadata", {})
                    music_genre = clean_text(video_metadata.get("music_genre", ""))
                    if music_genre:
                        results.append({
                            "source": source_file,
                            "type": "MusicGenre",
                            "text": music_genre,
                            "timestamp": media_item.get("creation_timestamp", timestamp)
                        })

        # 遞迴繼續找
        for value in obj.values():
            extract_label_values(value, source_file, results)

    elif isinstance(obj, list):
        for item in obj:
            extract_label_values(item, source_file, results)


def extract_hashtags(item):
    tags = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("label") == "Name" and x.get("value"):
                tag = clean_text(x["value"])
                if tag:
                    tags.append("#" + tag.lstrip("#"))
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(item)
    return tags


def extract_owner_names(item):
    owners = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("label") in ["Username", "Name"] and x.get("value"):
                val = clean_text(x["value"])
                if val:
                    owners.append(val)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(item)
    return owners


def deduplicate(items):
    seen = set()
    output = []

    for item in items:
        key = (item["type"], item["text"])
        if key in seen:
            continue
        seen.add(key)
        output.append(item)

    return output


def timestamp_to_date(ts):
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
    except Exception:
        return ""


def main():
    print("Parsing Instagram export...")

    results = []

    json_files = list(EXPORT_DIR.rglob("*.json"))

    if not json_files:
        print(f"No JSON files found under: {EXPORT_DIR}")
        return

    for path in json_files:
        filename = path.name

        if filename in SKIP_FILES:
            print(f"Skip sensitive file: {filename}")
            continue

        if filename not in ALLOW_FILES:
            print(f"Skip unlisted file: {filename}")
            continue

        try:
            data = load_json(path)
            extract_label_values(data, filename, results)
            print(f"Parsed: {filename}")
        except Exception as e:
            print(f"Failed to parse {filename}: {e}")

    results = deduplicate(results)

    # 最近資料排前面
    results.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("Instagram Exported User Data for Personalized Podcast Generation\n")
        f.write("=" * 70 + "\n\n")
        f.write("Privacy note:\n")
        f.write("- This file excludes email, phone number, exact location, device information, and friend map data.\n")
        f.write("- It only keeps captions, comments, hashtags, post owners, story titles, and music genres.\n\n")

        f.write("Extracted Instagram Signals:\n")
        f.write("-" * 70 + "\n")

        for i, item in enumerate(results[:500], start=1):
            date = timestamp_to_date(item.get("timestamp"))
            date_str = f"[{date}] " if date else ""
            f.write(
                f"{i}. {date_str}"
                f"({item['source']} / {item['type']}) "
                f"{item['text']}\n"
            )

    print()
    print(f"Saved parsed IG text to: {OUTPUT_FILE}")
    print(f"Number of extracted items: {len(results)}")
    print()
    print("Next step:")
    print("python src\\main.py")


if __name__ == "__main__":
    main()