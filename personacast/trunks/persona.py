"""
build persona out of trunk pool artifacts, user_profile.json
"""


from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from ..models import Expertise, Interest, Persona

PROFILE_FILE = "user_profile.json"


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tone_from(style_preferences: list[str]) -> str:
    """
    style preferances cleaned as a sentence and then attached to a default knowledgable friend vibe
    """
    cleaned = [str(s).strip().lower() for s in style_preferences if str(s).strip()]
    if not cleaned:
        return Persona.model_fields["tone"].default
    return (
        f"{', '.join(cleaned)} like a knowledgeable friend talking through what "
        "they just read, not a presenter reading a bulletin"
    )


def derive_persona( persona_id: str = "user", source_dir: Path | str | None = None, llm=None, *, memory=None, on_stage=None,) -> Persona | None:

    """
    building persona object from the user_profile.json 
    """
    from . import store as trunk_store

    directory = Path(source_dir) if source_dir is not None else trunk_store.pool_dir_for(persona_id)

    ## read the json
    profile = _read(directory / PROFILE_FILE)
    if not isinstance(profile, dict):
        if on_stage:
            on_stage(f"No {PROFILE_FILE} — keeping the existing persona")
        return None

    ## pull the ten topics
    topics = [str(t).strip() for t in profile.get("podcast_focus_keywords", []) if str(t).strip()]
    if not topics:
        if on_stage:
            on_stage("Profile has no podcast_focus_keywords — keeping the existing persona")
        return None

    try:
        pool_topics = trunk_store.load(directory).pool.topic_names
    except Exception:
        pool_topics = []

    ### guard for if user profile topics drift from pool topics

    if pool_topics and set(pool_topics) != set(topics):
        stale = len(set(topics) - set(pool_topics))
        if on_stage:
            on_stage(
                f"Profile and trunk pool disagree on {stale}/{len(topics)} topics — "
                "using the pool's. Rebuild the pool to use the newer topics."
            )
        topics = pool_topics

    ## persona object construction
    persona = Persona(persona_id=persona_id, interests=[Interest(topic=topic, expertise=Expertise.intermediate) for topic in topics], tone=_tone_from(profile.get("style_preferences", [])),)

    if on_stage:
        on_stage(f"Built persona from feed — {len(topics)} topics, no LLM call")

    return persona


def resolve(persona_path: str | Path,llm=None,*,prefer_derived: bool = True,on_stage=None,) -> Persona:
    """
    persona session should run with
    """
    from .. import config
    from ..models import load_persona

    if prefer_derived and config.TRUNKS:
        existing = _read(Path(persona_path))
        persona_id = str((existing or {}).get("persona_id") or Path(persona_path).stem)

        derived = derive_persona(persona_id, llm=llm, on_stage=on_stage)
        if derived is not None:
            return derived

    persona = load_persona(persona_path)
    if on_stage:
        on_stage(f"Using hand-written persona {Path(persona_path).name}")
    return persona


def explain(persona: Persona) -> str:
    lines = [f"tone: {persona.tone}", "", "topics:"]
    lines += [f"  {interest.topic}" for interest in persona.interests]
    if persona.avoid:
        lines += ["", "won't re-explain:"] + [f"  - {a}" for a in persona.avoid]
    return "\n".join(lines)
