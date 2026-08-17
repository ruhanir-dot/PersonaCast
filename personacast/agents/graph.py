"""
agentic retrieval for one topic orchestrated using langgraph :P
"""

from __future__ import annotations

import operator
from collections import Counter
from datetime import date
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from .. import config
from ..llm.client import LLMClient
from ..models import CuratedItem, Persona, PersonaMemory, RetrievedItem, TopicPlan
from ..pipeline import curation, dedup, queries, topics
from ..pipeline.retrieval import retrieve
from . import tools

class TopicRetrievalState(TypedDict, total = False): #keys optional loose type hint 
    """
    graph state for one topic a fresh state is created per topic

    --NOTES-- 
    not object we mutate each key is a channel 
    we hand node the whole state as a read only dict
    the node then returns a prtial dict containing the keys it wants to change 
    for each returned ket lang graph merges it useing specific channel reducer 
        new_state[k] = reducer(old_state[k], returned[k])
    keys the node didnt return are untouched
    """
    ### immutable attributes written once at user
    plan: TopicPlan
    persona: Persona 
    memory: PersonaMemory | None

    ### what plan_search node decides 
    search_queries: list[str] # newscentric queries for the web 
    arxiv_queries: list[str] #academic phrasing for arxiv if empty arxiv was not chosen for topic 
    sources: list[str] # typical value is ["web"] or ["web", "arxiv"]

    ### what retrieval node gets
    raw_items: list[RetrievedItem] # raw retrieved item from searcg
    curated: list[CuratedItem] # after curation what we keep. 

    ### traces and auditing
    notes: Annotated[list[str], operator.add]

### Search Plan from planning node, queries and where to get from web/arxiv, why these sources--rationale

class _SearchPlan(BaseModel):
    """
    planned queries and the source of info
    this is written by LLM!
    """
    queries: list[str]        
    sources: list[str]        
    arxiv_queries: list[str]  
    rationale: str           

### extention on to queries system prompt 
_PLAN_SYSTEM = queries._SYSTEM + (
    "\n\nYou ALSO choose which sources to search. Pick from exactly these two:\n"
    "- `web`: Tavily news search, results from roughly the last 30 days, works for any subject "
    "on earth. ALWAYS include it.\n"
    "- `arxiv`: academic preprints only, relevance-sorted with NO recency filter. Include it "
    "ONLY if peer-reviewed or preprint research literature genuinely exists for this subject "
    "AND is what a listener at this expertise level actually wants. It costs 3 seconds per "
    "query, so never include it speculatively. A topic merely SOUNDING technical is not "
    "enough: 'AI in fashion retail' is a news topic, not an arXiv topic.\n"
    "If you include `arxiv`, also return up to 3 `arxiv_queries` rewritten in academic "
    "phrasing -- method names, task names, no 'latest' or 'this year'. arXiv search does not "
    "understand recency words and will relevance-match them into unrelated papers.\n"
    "Set `rationale` to ONE short line explaining the source choice.\n"
    "If an ALREADY COVERED list is given, the listener has already been told those things. "
    "Your queries must target a DIFFERENT development -- a later stage of the same story, a "
    "different actor, or an unrelated development within the same topic. Do not re-search what "
    "is listed."
)

def _initial_state(plan:TopicPlan, persona:Persona, memory:PersonaMemory | None) -> TopicRetrievalState:
    """
    plan, persona, memory are immutable and are seeded so nodes read can't write to them 
    notes is channel for node return dict on errors, soures kept and results rationale and queries 
    These notes are done across plan_search, search, and curate node
    """

    return {"plan": plan, "persona": persona, "memory": memory, "notes": []}


### GRAPH CONSTRUCTION

def build_topic_graph(llm:LLMClient, on_stage = None): 
    """
    compile the per-topic retrieval graph
    """

    def stage(label:str) ->None : # progress tracking for streamlit
        if on_stage: 
            on_stage(label)

    ### NODE 1: PLAN NODE
    def plan_search(state:TopicRetrievalState) -> dict: 
        """
        plan search node, decide what to search for and where to search 
        also uses the persona memory and the already coverered block to list what istener has been told about same topic before 
        goal is the queries aim at new places!
        """

        plan, persona = state['plan'], state['persona'] # seed the plan and persona give state

        stage(f'Planning queries + sources: {plan.topic}') 

        try: 
            covered = tools.covered_block(state.get("memory"), plan.topic) # recover covered block 
            # user prompt
            user = (
                     f"Topic: {plan.topic}\n"
                     f"Listener expertise: {plan.expertise.value}\n"
                     f"AVOID list: {persona.avoid}\n"
                     + (f"\nALREADY COVERED with this listener:\n{covered}\n" if covered else "")
                     + f"\nGenerate {config.QUERIES_PER_TOPIC} search queries and choose sources." 
            )

            result = llm.structured(_PLAN_SYSTEM.format(year = date.today().year), user, _SearchPlan, temperature= 0.0)


            query_list = [query.strip() for query in result.queries if query.strip()][:config.QUERIES_PER_TOPIC]

            if not query_list: 
                raise ValueError("planning agent returned no usable queries")

            ### web/arxiv source list
            info_source_list = [s for s in result.sources if s in config.RETRIEVAL_SOURCES] 

            ### backup if nothing then just add web 
            if "web" not in info_source_list:
                info_source_list.append("web") 

            arxiv_query_list = (
                [q.strip() for q in result.arxiv_queries if q.strip()][: config.QUERIES_PER_TOPIC]
                if "arxiv" in info_source_list else []
            )

            note =  f"sources={info_source_list} | {result.rationale}" # add to note section

        except Exception as e: # if error fall back to linear
            query_list = queries.generate_queries(plan, llm)
            info_source_list = ["web"] + (["arxiv"] if retrieve._wants_arxiv(plan.topic) else [])
            arxiv_query_list = []
            note = f"plan_search fell back to the keyword path: {type(e).__name__}: {e}"

        return {
            "search_queries": query_list,
            "arxiv_queries": arxiv_query_list,
            "sources": info_source_list,
            "notes": [note, f"queries: {query_list}"], 
            }


    ### NODE 2: SEARCH NODE
    def search(state:TopicRetrievalState) ->dict: 
        """
        running the searches, throough api calls, web queries are done through thread pool 

        results are dropped if their URL appears in memorycovered[topic] --> so no sources that were seen in past session

        """

        plan = state["plan"]
        query_list, info_source_list = state["search_queries"], state["sources"]
        stage(f"Searching {' + '.join(info_source_list)}: {plan.topic}")

        items, errors = tools.search_web_parallel(query_list)

        if 'arxiv' in info_source_list:
            arxiv_items, arxiv_errors = tools.search_arxiv_serial(state.get("arxiv_queries") or query_list)
            items += arxiv_items
            errors += arxiv_errors

        seen = tools.covered_urls(state.get("memory"), plan.topic) # seen urls check
        novel = tools.novel_items(items, seen) # new items check

        return {
            "raw_items": novel,
            "notes": errors + [ f"{len(items)} hits, {len(novel)} novel ({len(items) - len(novel)} already seen or duplicated)"],
            }

    ### NODE 3 : CURATE NODE

    def curate(state : TopicRetrievalState): 
        """
        score rank and keep sources downstream this will become a ranker 
        unchanged curation from past linear pipeline
        """

        plan = state["plan"]
        items = state.get("raw_items", [])[: config.RETRIEVAL_MAX_CANDIDATES]
        stage(f"Curating {len(items)} sources: {plan.topic}")

        try:
            kept = curation.curate_topic(plan, items, state["persona"], llm)
        except Exception as e:
            return {
                "curated": [],
                "notes": [f"curation failed for {plan.topic}: {type(e).__name__}: {e}"],
            }

        words = sum(len(item.summary.split()) for item in kept)
        mix = dict(Counter(item.source for item in kept))
        return {
            "curated": kept,
            "notes": [f"kept {len(kept)}/{len(items)} sources, {words} words, mix={mix}"],
        }

    ### GRAPH CONSTRUCTION ADDEDING NODES AND EDGES

    graph = StateGraph(TopicRetrievalState)

    ### adding nodes
    graph.add_node("plan_search", plan_search)
    graph.add_node("search", search)
    graph.add_node("curate", curate)

    ### adding edges
    graph.add_edge(START, "plan_search")
    graph.add_edge("plan_search", "search")
    graph.add_edge("search", "curate")
    graph.add_edge("curate", END)

    return graph.compile()



def build_source_pool_agentic(persona: Persona, llm: LLMClient, on_stage=None, *, memory: PersonaMemory | None = None, on_topic_done=None) -> dict[str, list[CuratedItem]] : 
    """
    entry point for Agentic Retrieval Pipeline runs once per topic
    """

    def stage(label: str) -> None:
        if on_stage:
            on_stage(label)

    # stage logging
    stage("Planning Topics")
    plans = topics.plan_topics(persona)

    # compile graph
    graph = build_topic_graph(llm, on_stage)  

    curated: dict[str, list[CuratedItem]] = {}
    for plan in plans:
        final = graph.invoke(_initial_state(plan, persona, memory))
        curated[plan.topic] = final.get("curated", [])
        if on_topic_done:
            on_topic_done(plan.topic, final)

    stage("Removing Duplicates")
    curated, _notes = dedup.dedup_across_topics(curated)

    return curated

        


        








