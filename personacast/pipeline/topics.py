"""
Turns persona.interests into a list of TopicPlan objects with topic, expertise and word budget
Phase 1 splits equally across topics, an idea for future implementtation is by taking counts of mid-podcast questions we can
gauge how interested user is in topic and allott more time for future podcast generation on that topic
"""

from __future__ import annotations
from .. import config 
from ..models import Persona, TopicPlan

def plan_topics(persona : Persona) -> list[TopicPlan]: 
    n = len(persona.interests)# get amount of interests
    budget = config.per_topic_word_budget(n)

    return[
        TopicPlan(topic = interest.topic, expertise = interest.expertise, word_budget = budget)
        for interest in persona.interests
    ]

if __name__ == "__main__": 
    from ..models import load_persona

    persona = load_persona("personas/ruhani.json")

    for plan in plan_topics(persona): 
        print(f"{plan.topic:<22} [{plan.expertise.value:<12}] budget={plan.word_budget} words")

