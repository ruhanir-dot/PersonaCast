from __future__ import annotations

import json
from typing import Any

from src.models.session_state import SessionState
from src.utils import llm_json


def generate_tree_fallback(
    *,
    current_node: dict[str, Any],
    user_question: str,
) -> str:
    payload = {
        "task": "Generate a new podcast segment that directly answers the user's question.",
        "user_question": user_question,
        "current_node": {
            "topic": current_node.get("topic"),
            "user_question": current_node.get("user_question"),
            "script": current_node.get("script", ""),
            "recent_path_history": current_node.get("path_history", [])[-3:],
        },
        "output_schema": {
            "script": "new content-only spoken podcast segment that directly answers the user question",
        },
    }

    result = llm_json(
        json.dumps(payload, ensure_ascii=False, indent=2),
        system=(
            "Return valid JSON only. Answer the user's question directly while staying coherent "
            "with the current podcast segment. Do not use an unrelated prepared candidate. "
            "Do not add a greeting, introduction, conclusion, or transition."
            "Write exactly one paragraph containing 85 to 140 "
            "English words, suitable for about 30 to 60 seconds "
            "of speech. Do not use headings, numbered lists, "
            "bullet points, Markdown, a greeting, an introduction, "
            "a conclusion, or a transition."
        ),
        temperature=0.35,
        max_output_tokens=300,
    )

    raw_script = (
        result.get("script")
        or result.get("response")
        or result.get("answer")
    )

    if not raw_script:
        raise RuntimeError(
            f"Fallback LLM returned no script: {result}"
        )

    return str(raw_script)
