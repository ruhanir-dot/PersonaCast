"""
expertise injection evaluation script, hold proifile constant and change one topic expertise level in this case machine learning 
aim to answer: will we see that the same question over the same frozen sources produces an appropriately
different answer for beginner / intermediate / advanced?

basic config: 
- sources: eval/frozen_sources.json 
  - personas: personas/eval/ml_{beginner,intermediate,advanced}.json (identical tone/avoid/topic)
  - web fallback: OFF (allow_web=False) so the source set can never change across the personas
  - prompt: _QA_EVAL_SYSTEM (longer response to see more difference)

output answers.md, and interactions.json logging questions, persona ids and answers 
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from personacast.llm.client import LLMClient
from personacast.models import load_persona
from personacast.pipeline.qa import (
    _QA_EVAL_SYSTEM,
    answer_question,
    flatten_curated,
    load_curated,
)

## path resolving for correct placement
EVAL_DIR = Path(__file__).resolve().parent
REF_ROOT = EVAL_DIR.parent
FROZEN_SOURCES = EVAL_DIR / "frozen_sources.json"
PERSONA_FILES = [
    REF_ROOT / "personas/eval/ml_beginner.json",
    REF_ROOT / "personas/eval/ml_intermediate.json",
    REF_ROOT / "personas/eval/ml_advanced.json",
]

### Questions
QUESTIONS = [
    "Explain what self-supervised learning is.",
    "How are recent results fine-tuning large language models, and which techniques stand out?",
]

REPEATS = 1

def _persona_expertise(persona) -> str: # for extracting persona expertise level
    return persona.interests[0].expertise.value

def main(): 

    curated = flatten_curated(load_curated(FROZEN_SOURCES))
    personas = [load_persona(persona) for persona in PERSONA_FILES]
    llm = LLMClient()

    interactions = [] # every run is appended into this list linearlly, used for the raw interactions json output
    grid: dict[str, dict[str,list]] = {} # type hinted to a nested dictionay where for each persona_id we have a keyed question and then depending on repetitions different run answers

    total = len(personas) * len(QUESTIONS) * REPEATS # used for cli tracking
    done = 0 

    for persona in personas: 
        grid[persona.persona_id] = {question: [] for question in QUESTIONS} # initialies oene entry for current persona per question
        for question in QUESTIONS:
            for run_index in range(REPEATS): # repeat as much as needed as specfed through the number of rpeats cofugured
                done += 1 # aggregate to the counter
                print(f"[{done}/{total}] {persona.persona_id} | run {run_index} | {question[:50]}…")
                ans = answer_question( question, persona, curated, llm, allow_web=False, system=_QA_EVAL_SYSTEM) # get answer
                grid[persona.persona_id][question].append(ans) # append answer o the grid sturucture we have set up
                
                interactions.append({
                "persona_id": persona.persona_id,
                "expertise": _persona_expertise(persona),
                "question": question,
                "run_index": run_index,
                "answered": ans.answered,
                "used_web": ans.used_web,
                "answer": ans.answer,
                "sources_used": [source.model_dump() for source in ans.sources_used]   
                })
    
    out_dir = EVAL_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    ## interaction log 
    (out_dir / "interactions.json").write_text(json.dumps(interactions, indent=2))

    ## markdown table creation 
    lines = [
        "# Expertise-injection evaluation",
        "",
        f"Frozen sources: `{FROZEN_SOURCES.name}` ({len(curated)} items) "
        f"prompt used: `_QA_EVAL_SYSTEM`"
        "",
        "| expertise | " + " | ".join(f"Q{i + 1}" for i in range(len(QUESTIONS))) + " |",
        "|" + " --- |" * (len(QUESTIONS) + 1),
    ]
    for persona in personas:
        cells = []
        for question in QUESTIONS:
            text = grid[persona.persona_id][question][0].answer
            cells.append(text.replace("\n", "<br>").replace("|", "\\|"))
        lines.append(f"| **{_persona_expertise(persona)}** | " + " | ".join(cells) + " |")
    lines += ["", "## Questions", ""]
    lines += [f"- **Q{i + 1}:** {q}" for i, q in enumerate(QUESTIONS)]
    (out_dir / "answers.md").write_text("\n".join(lines))

    print(f"\nWrote {out_dir}/answers.md and interactions.json")
    return out_dir

if __name__ == "__main__":
    main()



