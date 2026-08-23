
from __future__ import annotations

import json
import time
from pathlib import Path

from .. import config
from ..models import CuratedItem, Persona


def cache_dir() -> Path:
    return Path(config.POOL_CACHE_DIR)


def cache_path(persona: Persona) -> Path:
    return cache_dir() / f"{persona.persona_id}.json"


def _topic_key(persona: Persona) -> list[str]:
    return sorted(interest.topic for interest in persona.interests)


def load(persona: Persona) -> dict[str, list[CuratedItem]] | None:
    if not config.POOL_CACHE:
        return None

    path = cache_path(persona)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    if raw.get("topics") != _topic_key(persona):
        return None

    age_hours = (time.time() - raw.get("built_at", 0)) / 3600
    if age_hours > config.POOL_CACHE_TTL_HOURS:
        return None

    try:
        return {
            topic: [CuratedItem.model_validate(item) for item in items]
            for topic, items in raw.get("pool", {}).items()
        }
    except Exception:
        return None


def save(persona: Persona, pool: dict[str, list[CuratedItem]]) -> Path | None:
    if not config.POOL_CACHE:
        return None

    path = cache_path(persona)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "persona_id": persona.persona_id,
        "topics": _topic_key(persona),
        "built_at": time.time(),
        "pool": {topic: [item.model_dump() for item in items] for topic, items in pool.items()},
    }, indent=2))
    return path


def age_hours(persona: Persona) -> float | None:
    try:
        raw = json.loads(cache_path(persona).read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return (time.time() - raw.get("built_at", 0)) / 3600
