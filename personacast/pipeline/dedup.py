"""
cross topic deduplication 
in first iteration similar topics often interesected in topic discussion overlap in RAG and recsys for example 
this module sees whole curated set of sources, this is a rather simple impelemenattion that just makes sure the same source isn't showing up across multiple topics 
this is between curation and script generation 

no LLM call, when source under several topics we keep it under the first topic in order ad drop from rest, we record whats dropped
"""

from __future__ import annotations

from ..models import CuratedItem


def _norm_url(url):
    return url.rstrip("/").lower() # removing any trailing slashes in the url 


def dedup_across_topics( curated: dict[str, list[CuratedItem]],) -> tuple[dict[str, list[CuratedItem]], list[str]]:
    """
    returns same dictionary structure of the curated items where the string is topic key and list of curated items for each topic and 
    list[str] is the dedupilication notes of the URLS dropped and why which get added to the pipe_state.notes section
    """
    seen = {}  # maps url to topic, helps track which topic source first appeared in
    deduped_set= {}
    notes = []

    for topic, items in curated.items(): 
        kept = []

        for item in items: 
            url = _norm_url(item.url)
            
            if url in seen: 
                notes.append(
                    f"cross-topic dup dropped: '{item.title[:50]}' from '{topic}' "
                    f"(kept under '{seen[url]}')"
                )
                continue
            
            seen[url] = topic # if not duplicate(not in seen) add this url to this topic
            kept.append(item) # add to list of items that are kept passed dedup
        
        deduped_set[topic] = kept #after all items processes survivng list into a result dict
    
    return deduped_set, notes



