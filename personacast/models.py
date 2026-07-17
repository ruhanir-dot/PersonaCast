from __future__ import annotations
import json
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field

##### Unchanged from original #####

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
    additional_context: str = "" # addtional context you can add like walk in the park

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

##### New Interactive Session Pydantic Models

class ReactionType(str, Enum): 
    """
    reaction types a user can give in response to one 60s turn, enum type object 
    text w/ ? we take as a question, nothing entered is none, and otherwise we take as a comment
    """
    comment = "comment"
    question = "question"
    none = "none"

class Reaction(BaseModel):
    """
    model structure to define reaction at some iteration, on certain topic
    """

    iteration: int 
    topic: str 
    type: ReactionType
    text: str = "" # actual comment and question text
    answer: str # when a question is asked we use the qa.answer_question method, store in here
    #this is downstream injected into prompt as context to seamlessly answer question and move on
    requested_topic: str | None = None # we set this to a requested topic when user asks to switch
    ## current logic for this it looks at the active topics in the session and fires if reaction names other topic, could do an LLM call for this maybe
    ## additionally we will use requested topic on next score, do -2 switch penalty on current topic

class PersonaMemory(BaseModel): 
    """
    persistent persona memory keyed by the persona_id, for now is stored on disk looking at ideas to extend for deployed vesion 
    stored at personas/memory/<persona_id>.json, this survives across the sessions 
    per session details are inserted every session by user interest, tone, avoid
    """

    persona_id: str 
    engagement: dict[str, float] = Field(default_factory= dict) # engagement points keyed for each topic
    reactions: list[Reaction] = Field(default_factory= list)  #list of Reaction Objects across sessions we have stored to see reactions of user
    summary: str = "" # concatenates each interactie turn gist, cross session log of what we have covered with user
    updated_at: str = "" # just a straight datetime log 

class InteractiveTurn(BaseModel): 
    """
     to hold the content of one 60s turn, holds its generated tect, a condensed one liner gist
     at turn n we use the turn n-1 gist as context, and holds listener reaction to that turns generated
    """
    iteration: int
    topic: str # topic for turn
    text: str  #generated turn text
    gist: str = ""
    reaction: Reaction | None = None # reaction to the generation, as Reaction object

class SessionState(BaseModel): 
    """
    this is a growing session object, its dunped per turn for any debugging!
    """
    run_id: str
    persona: Persona 
    memory: PersonaMemory
    pool: dict[str, list[CuratedItem]] = Field(default_factory=dict) # pool of surces built once 
    turns : list[InteractiveTurn] = Field(default_factory=list)
    current_topic:  str | None = None # sticky topic we stay at topic unless disengagement leads to engement point decrease and we move to next best on argmax
    # switch if topic change requested w/ penalty






