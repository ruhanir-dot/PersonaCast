from __future__ import annotations

import json
from typing import Any

from src.models.candidate import Candidate
from src.models.session_state import SessionState
from src.utils import PROJECT_ROOT, llm_json

PROMPT_FILE = PROJECT_ROOT / "src" / "prompts" / "candidate_adaptation.txt"


def adapt_tree_script(
    *,
    current_script: str,
    user_question: str,
    matched_question: str,
    matched_script: str,
) -> str:
    payload = {
        "task": "Make a small adaptation to a prepared podcast segment so it directly answers the user's actual question.",
        "current_podcast_segment": current_script,
        "predicted_question": matched_question,
        "actual_user_question": user_question,
        "prepared_script": matched_script,
        "output_schema": {
            "script": "lightly revised content-only spoken podcast segment",
        },
    }

    result = llm_json(
        json.dumps(payload, ensure_ascii=False, indent=2),
        system=(
            "Return valid JSON only. Preserve the facts and main content of the prepared script. "
            "Make only the changes needed to answer the actual user question precisely. "
            "Do not add a greeting, introduction, conclusion, or transition."
            "Write exactly one paragraph containing 85 to 140 "
            "English words, suitable for about 30 to 60 seconds "
            "of speech. Do not use headings, numbered lists, "
            "bullet points, Markdown, a greeting, an introduction, "
            "a conclusion, or a transition."
        ),
        temperature=0.2,
        max_output_tokens=350,
    )

    return str(result.get("script") or matched_script).strip()
