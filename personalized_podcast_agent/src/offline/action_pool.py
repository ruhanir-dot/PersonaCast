"""Offline action types shared by the action predictor."""
ACTION_POOL = {
    "continue": "Continue listening without a question; follow the default next node.",
    "clarify": "Ask for a simpler explanation of a concept in the current script.",
    "go_deeper": "Ask for reasons, details or examples about the current subject.",
    "explore_related": "Ask about a related story or perspective.",
    "switch_topic": "Ask to hear a different broad topic.",
    "not_interested": ("Say, as a short statement (not a question), that this segment does not "
                       "interest you and you would rather hear something else."),
}
# Action types the listener says as a statement; they get an acknowledgement, not an answer.
STATEMENT_TYPES = {"not_interested"}

ACTION_SCHEMA = {
    "type": "object",
    "properties": {"actions": {
        "type": "array", "minItems": 3, "maxItems": 3,
        "items": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": list(ACTION_POOL)},
                "label": {"type": "string"},
                "target_id": {"type": "string"},
            },
            "required": ["action_type", "label", "target_id"],
            "additionalProperties": False,
        },
    }},
    "required": ["actions"], "additionalProperties": False,
}
