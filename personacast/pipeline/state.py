"""
help with adding things onto our pipelineState, we have the pipeline as one growing state object 
new_run_id() --> timestamp id 
log_stage() --> dump the current PipeLineState to runid folder as a json 
save_outputs --> writes script.txt and the sources used for each topic
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from .. import config
from ..models import PipelineState


def new_run_id(): 
    return datetime.now().strftime("%Y%m%d_%H%M%S") # turn datetime into formated string


def run_dir(run_id): 
    directory = Path(config.RUNS_DIR) / run_id
    directory.mkdir(parents = True, exist_ok = True)
    return directory

stage_counter= {} 
def log_stage(state, stage ) -> None:
    """
    Snapshot the state after a stage completes, stored in json files with the growing state.

    Each log file gets a sequential number like `01_topics.json`, `02_curated.json`, `03_deduped.json`
    """
    n = stage_counter.get(state.run_id, 0) + 1 # keeps track of how many times log_stage is called
    stage_counter[state.run_id] = n
    path = run_dir(state.run_id) / f"{n}_{stage}.json"
    path.write_text(state.model_dump_json(indent=2))

def log_turn(session_state, iteration: int): 
    """
    snapshot sessionstate after turn `iteration` write to 
    runs/<run_id>/turn_{iteration}.json
    """
    path = run_dir(session_state.run_id) / f"turn_{iteration}.json"
    path.write_text(session_state.model_dump_json(indent=2))


def save_outputs(state) -> Path:
    """
    write script.txt and sources.json . returns the run directory.
    """
    directory = run_dir(state.run_id)
    if state.script:
        (directory / "script.txt").write_text(state.script)

    # sources for each topic writen as json 

    # dict comprehensio to convers curated data into json
    sources = {
        topic: [item.model_dump() for item in items] # convert every pydantic object into dict
        for topic, items in state.curated.items() # every topic item pairs
    }
    (directory / "sources.json").write_text(json.dumps(sources, indent=2))
    return directory
