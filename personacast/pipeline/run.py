"""
pipeline orchestrator (moved here from the top-level run.py so the CLI and any future UI can
both import run_pipeline from one place; the top-level run.py is now just argument parsing)
"""

from __future__ import annotations
from ..llm.client import LLMClient
from ..models import PipelineState
from ..pipeline import curation, dedup, queries, script, state, topics
from ..pipeline.retrieval import retrieve
from ..pipeline import tts

def run_pipeline(persona,*, audio: bool = False, on_stage=None ):
    llm = LLMClient()

    def stage(label): # if a UI passed a callback, report each stage as it starts
        if on_stage:
            on_stage(label)

    pipe_state = PipelineState(run_id = state.new_run_id(), persona = persona) # instiate pipeline object

    ### Topic Extraction/ Word Budget
    stage("Planning topics")
    pipe_state.topic_plans = topics.plan_topics(persona) # construct plan for each topic
    state.log_stage(pipe_state, "topics") # log topic state

    for plan in pipe_state.topic_plans:
        # for each plan in our topic plan
        stage(f"Retrieving & curating: {plan.topic}")

        ### Query
        query = queries.generate_queries(plan, llm) # generate queries for topic
        pipe_state.queries[plan.topic] = query # add queries to pipeline state, keyed topic name

        ### Retrieve
        raw = retrieve.retrieve_topic(plan, query) # retrieve based on queries
        pipe_state.retrieved[plan.topic] = raw # add retrieved to pipeline state, keyed topic name

        ### Curate
        curated_raw = curation.curate_topic(plan, raw, persona, llm)  # Step 4
        pipe_state.curated[plan.topic] = curated_raw # append to pipeline  state key topic name

    state.log_stage(pipe_state, "curated") # log curated state

    ### De-Duplicate
    stage("Removing duplicates")
    pipe_state.curated, dedup_notes = dedup.dedup_across_topics(pipe_state.curated) # dedup across curated content
    pipe_state.notes.extend(dedup_notes) #add the dedup notes in notes of pipeline object
    state.log_stage(pipe_state, "deduped")

    ### Per-Topic Segment Generation
    for plan in pipe_state.topic_plans:
        stage(f"Writing segment: {plan.topic}")
        segment = script.write_segment(
            plan.topic, plan.expertise.value, plan.word_budget,
            pipe_state.curated.get(plan.topic, []), persona, llm
        )
        pipe_state.segments.append(segment)

    ### Stitch Segments
    stage("Stitching episode")
    pipe_state.script = script.stitch(pipe_state.segments, persona, llm)

    state.log_stage(pipe_state, "script")

    ### audio flag given generate audio
    if audio:
        out = state.run_dir(pipe_state.run_id)/"episode.wav"
        pipe_state.audio_path = str(tts.synthesize(pipe_state.script, out))
        state.log_stage(pipe_state, "audio")

    ## Keep script.txt and sources.json artifact
    out_dir = state.save_outputs(pipe_state)
    print(f"Outputs in {out_dir}")

    return pipe_state
