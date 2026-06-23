"""
After getting retrieved sources 
For EACH one topic we curate and keep only items adhering to word budget

v1: One LLM call to evaluate each candidate source (too many calls)
v2 (current): Sens candidates in small batches

we evaluate on 
1. rejecting anything matching persona's avoid list 
2. judge topic relevance, recency, and quality
3. flag near duplicates

wrote summaries of curated sources
"""



from __future__ import annotations

from pydantic import BaseModel

from ..llm.client import LLMClient
from ..models import CuratedItem, Persona, RetrievedItem, TopicPlan

# per candidate character limit for batch processing of candidate items
_MAX_CHARS_PER_CANDIDATE = 2500
# how many candidates per call
_BATCH_SIZE = 8

class _ScoredItem(BaseModel): 
    index:int
    keep: bool
    score: float
    key_facts: list[str] # 3-6 claims from source text, to not invent relationshups (did this a lot in sports)
    dedup_key: str  # if two articles about same event get same dedup key, so when iterating by scores higher scored same key article kept

class _ScoredBatch(BaseModel):
    items: list[_ScoredItem]


### Curator prompt
_MAP_SYSTEM = (
    "You are a strict curator selecting sources for ONE podcast topic. You are given a numbered "
    "batch of candidate sources. Score each one.\n\n"
    "HARD RULES:\n"
    "- keep=false if the source is not SPECIFICALLY about the given topic. Being recent or "
    "generally about AI is NOT enough — a recent paper on an unrelated subject must be rejected. "
    "(This is the most common mistake: recency-sorted results that aren't on-topic.)\n"
    "- keep=false if it matches anything on the AVOID list (low-quality content OR a subject the "
    "listener doesn't want covered).\n"
    "- keep=false for thin/promotional/SEO content with no real substance.\n\n"
    "For kept items: score 0-1 on topic-relevance + newsworthiness, and set dedup_key to a short "
    "slug naming the underlying development so two sources about the same thing share a key.\n\n"
    "KEY FACTS — this is EXTRACTIVE, not a summary you compose. Output 3-6 `key_facts`, where each "
    "fact is a SINGLE claim that is stated directly in the source text, closely paraphrased "
    "(stay near the original wording). Rules:\n"
    "- Every fact must be directly supported by a specific sentence in the source. If the text "
    "doesn't say it, do not write it. Inventing or inferring a relationship is a failure.\n"
    "- Keep proper names verbatim (people, teams, orgs, products, places) WITH the exact role the "
    "source gives them. If the source names someone, name them; do not generalize to 'a veteran'.\n"
    "- Preserve affiliations and sides EXACTLY. If a person appears only as an OPPONENT or in "
    "passing, your fact must reflect that — never fold an opponent into the subject's own team/"
    "roster (e.g. an article about Team A beating Player X's Team B must NOT yield a fact that "
    "Team A drafted/developed Player X). Do not relabel timeframes (regular season vs playoffs).\n"
    "- One fact = one claim. Do not chain unrelated claims into a single fact."
)


def _score_batch( batch: list[RetrievedItem], plan: TopicPlan, persona: Persona, llm: LLMClient) -> list[_ScoredItem]:
    listing = "\n\n".join( # numbered text block of candidates in batch to pass to LLM
        f"[{i}] source={item.source} | title={item.title}\n"
        f"{item.content[:_MAX_CHARS_PER_CANDIDATE]}" # first 2500 chars of content
        for i, item in enumerate(batch)
        )

    user = (
        f"Topic: {plan.topic}\n"
        f"Listener expertise: {plan.expertise.value}\n"
        f"AVOID list: {persona.avoid}\n\n"
        f"Candidates ({len(batch)}):\n{listing}"
    )
    ### llm call at lower temp bc summary is only thing seen downstream want it to be factually based
    result = llm.structured(_MAP_SYSTEM, user, _ScoredBatch, temperature=0.2)
    
    # safe guard against missing index
    return [s for s in result.items if 0 <= s.index < len(batch)] # returned list of _ScoredItem objects

def curate_topic( plan, items, persona, llm ) -> list[CuratedItem]:

    scored: list[tuple[_ScoredItem, RetrievedItem]] = []

    for start in range(0, len(items), _BATCH_SIZE): # iterate in batch sizes
        batch = items[start : start + _BATCH_SIZE]
        
        for scored_item in _score_batch(batch, plan, persona, llm):
            if scored_item.keep: # when scored item is a keeper, we add tuple of scored object with originally retrieved item
                scored.append((scored_item, batch[scored_item.index]))

    scored.sort(key=lambda pair: pair[0].score, reverse=True) # sort. the tuples 

    kept = [] # list of CuratedItem
    seen_keys = set()
    words_used = 0
    for scored_item, item in scored:
        
        if scored_item.dedup_key in seen_keys:
            continue
        summary = " ".join(f.strip() for f in scored_item.key_facts if f.strip()) # key facts become summary
        
        if not summary: # skips items where key facts empty
            continue

        cost = len(summary.split()) # count words in summary

        if words_used + cost > plan.word_budget: # check against word budget
            continue

        seen_keys.add(scored_item.dedup_key) # add dedup key to seen keys so future items in the loop with same dedup key is skipped
        words_used += cost # add to word_used counter so we can keep checking against it=
        
        kept.append(
            CuratedItem(source=item.source, title=item.title, url=item.url, summary=summary)
        )

        return kept