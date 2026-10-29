"""Read the player's interaction log (data/interactions/YYYYMMDD.jsonl, written by trunks_api).

Records become history rows in the same shape load_history() returns, oldest first, so the
predictor and the profile updater learn from what the listener actually did in the new player.
Automatic 'continue' events carry no reaction and are skipped.
"""
import hashlib
import json
from pathlib import Path

from src import paths

INTERACTIONS_DIR = paths.DATA_ROOT / "interactions"
SKIP_EVENTS = {"skip_topic"}


def read_interactions(user_id, folder=None):
    folder = Path(folder or INTERACTIONS_DIR)
    if not folder.is_dir():
        return []
    rows = []
    for file in sorted(folder.glob("*.jsonl")):
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("event") == "continue" or record.get("user_id") not in (None, user_id):
                continue
            skipped = record.get("event") in SKIP_EVENTS or record.get("action_type") == "not_interested"
            text = " ".join(str(record.get("text") or "").split()) or (
                "Not interested in this topic; skip it." if skipped else "")
            if not text:
                continue
            rows.append({
                "event_id": hashlib.sha256(json.dumps(
                    [record.get("timestamp"), record.get("session_id"), record.get("trunk_id"), text]
                ).encode()).hexdigest()[:24],
                "timestamp": record.get("timestamp"),
                "run_id": record.get("session_id") or file.stem,
                "current_trunk_id": record.get("trunk_id"),
                "played_script": record.get("played_script") or "",
                "topic": None,
                "user_text": text,
                "recorded_intent": record.get("action_type") or record.get("event"),
                "recorded_type": "skip" if skipped else "question"})
    rows.sort(key=lambda row: row["timestamp"] or "")
    for number, row in enumerate(rows, 1):
        row["iteration"] = number
    return rows
