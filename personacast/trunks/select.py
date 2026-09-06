"""
trunk selection within a chosen topic 
choose_topic() is what we use fr which topic by engagment

this module is to determine which one of topics 5 trunks to move to next
"""



from __future__ import annotations
from ..models import CoveredSource, SessionState
from .schema import Trunk, TrunkPool


def covered_urls(session_state: SessionState, topic: str) -> set[str]:
    """
    get covered cources
    """
    entries: list[CoveredSource] = session_state.memory.covered.get(topic, [])
    return {entry.url for entry in entries}


def next_trunk(pool: TrunkPool, session_state: SessionState, topic: str) -> tuple[Trunk, int] | None:
    """
    selecting next. trunk that hasnt been used yet
    """
    entry = pool.by_topic().get(topic)

    if entry is None or not entry.trunks:
        return None

    seen = covered_urls(session_state, topic) # pull list of covered sources seen

    candidates = [
        (index, trunk)
        for index, trunk in enumerate(entry.trunks) # entry is TrunkTopic for this topic, each trunk topic has 5 trunk objects so we walk that
        if trunk.source_item.url not in seen
    ]
    if not candidates:
        return None

    index, trunk = max( candidates,key=lambda pair: (pair[1].personalized_seed.topic_similarity, -pair[0]),) # the not seen trunks are then checked for highest personalized seed. topicsimilarity and thats the trunk we go to
    return trunk, index


def remaining_counts(pool: TrunkPool, session_state: SessionState) -> dict[str, int]:
    """
    unuesed trunks per topic
    """
    counts: dict[str, int] = {}
    for entry in pool.topics:
        seen = covered_urls(session_state, entry.topic)
        counts[entry.topic] = sum( 1 for trunk in entry.trunks if trunk.source_item.url not in seen)
    return counts


def live_topics(pool: TrunkPool, session_state: SessionState, topics: list[str]) -> list[str]:
    """
    subset of topics that have material left meaning havent exhausetd all 5 trunks
    """
    counts = remaining_counts(pool, session_state)
    alive = [topic for topic in topics if counts.get(topic, 0) > 0]
    return alive or topics
