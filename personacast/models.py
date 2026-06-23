from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

# Persona Object Construction and others
class Expertise(str, Enum):
    beginner = 'beginner'
    intermediate = 'intermediate'
    advanced = 'advanced'


class Interest(BaseModel): 
    """
    pairing topic and expertise as one object
    """
    topic: str 
    expertise: Expertise


class Persona(BaseModel): 
    persona_id: str
    interests: list[Interest]
    tone: str = "technical and " \
    "conversational discussion similar to talking to a peer"
    avoid: list[str] = Field(default_factory = list)

def load_persona(path): 
    """
    load in the persona JSON, validate the JSON using pydantic
    """
    raw = json.loads(Path(path).read_text())
    return Persona.model_validate(raw)


# construction of stage classes for stage tracking
class TopicPlan(BaseModel):
    """
    stage one aggregate topic, expertise, and word budget (equal distribution)
    """ 
    topic: str
    expertise: Expertise
    word_budget: int

class RetrievedItem(BaseModel): 
    """
    retrieval results for each topic
    """
    source: str 
    title: str 
    url: str
    content: str
    published: str | None = None # optional want for recency filtering

class CuratedItem(BaseModel): 
    """
    curated results for each topic
    """
    source: str 
    title: str 
    url : str
    summary : str 

class TopicSegment(BaseModel): 
    """
    for each topic, the seperate generated segments
    """
    topic: str
    text: str

# growing state object

class PipelineState(BaseModel): 
    run_id: str 
    persona: Persona
    topic_plans: list[TopicPlan] = Field(default_factory = list)
    queries: dict[str, list[str]] = Field(default_factory = dict)
    retrieved: dict[str, list[RetrievedItem]] = Field(default_factory = dict)
    curated: dict[str, list[CuratedItem]] = Field(default_factory = dict)
    segments: list[TopicSegment] = Field(default_factory = list)
    script: str | None = None
    audio_path: str | None = None # only on flag 
    notes: list[str] = Field(default_factory = list) # log on gaps like duplicate topics











