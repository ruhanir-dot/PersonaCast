"""
persistent per persona memory, this is keyed by persona id and stored at personas/memory/<personaid>.json
as mentioned prior the persona details for that session like interests, and tone, avoid and per sessio additional context are re-entered every session
The persistent memory file carries the users across session interaction history , and their accumulated per topic engagment points 

load_memory(persona) -> load or initialize peristed memory for user, then merge memory with entered persona details fror current session
save_memory(memory) -> put in when last updated and write json to disk 
classify_reaction(text) -> given a reaction infer the ReactionType from text where if we see the ending with ? we claim to be questionm and if nothing we keep as none, anything else is comment 
apply_reaction(memory, reaction ) -> append reaction to history, and add points to engagement[topic]
next_topic(memory, topics) -> choose next topic with simmple argmax of engagement points per topic 
"""

from __future__ import annotations 

import re 
from datetime import datetime
from pathlib import Path 
from .. import config 
from ..models import Persona, PersonaMemory, Reaction, ReactionType


def memory_path(persona_id) -> Path: 
    return Path(config.MEMORY_DIR) / f"{persona_id}.json" # pull persona id from persona 

def load_memory(persona:Persona) -> PersonaMemory:
    """
    load persistent memory for persona.persona_id if exists, else start new memory
    if memory exists merge memory w/ persona_ids current session details
    -make sure every current active interestt opic has engagememt entry
    """ 

    path = memory_path(persona.persona_id) # build memory path for the unique persona id 
    if path.exists(): # if the json path exists
        memory = PersonaMemory.model_validate_json(path.read_text())# read into and turn into PersonaMemory object
    else: 
        memory = PersonaMemory(persona_id = persona.persona_id) # or else initialize PersonaMemory object 

    for interest in persona.interests: 
        memory.engagement.setdefault(interest.topic, config.ENGAGE_BASE) # engaeg base value inseerted and key created if key doesnt exist 
    
    return memory

def save_memory(memory: PersonaMemory) -> Path:
    """
    update updated_at attribite of PersonaMemory object and write memory json to disk w/ directory creation
    """
    memory.updated_at = datetime.now().isoformat(timespec="seconds") # update memory
    path = memory_path(memory.persona_id)  # define path
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(memory.model_dump_json(indent = 2 )) # dump to the path we have 
    return path 

def classify_reaction(text) -> ReactionType: 
    """
    inferring reaction type from text 
    end with ? -> question (goes to answer path)
    empty or whitespace -> none (disengagement signal)
    otherwise -> comment
    """

    stripped = text.strip() # removing trailling whitespace 
    if not stripped: 
        return ReactionType.none
    if stripped.endswith('?'): 
        return ReactionType.question
    
    return ReactionType.comment

REACTION_POINTS = {
    ReactionType.question : config.ENGAGE_QUESTION, 
    ReactionType.comment: config.ENGAGE_COMMENT, 
    ReactionType.none: config.ENGAGE_NONE
}

def detect_requested_topic(text, active_topics: list[str], exclude) -> str | None: 
    """
    if the user reaction explicitly names the other topic in response, we take it as a topic switch request
    take it as whole word and cas insenstive, the exclude guard makes sure that current topic of turn that user reacted to 
    cannot be a possible topic to switch to
    so when we use use exclude = turn.topic
    """
    text_lower = text.lower() 
    hits = [ # collects if topics that are not current topic found in users reaction
        topic for topic in active_topics
        if topic != exclude and re.search(rf"\b{re.escape(topic.lower())}\b", text_lower)
    ]
    return min(hits,key = lambda topic: re.search(rf"\b{re.escape(topic.lower())}\b", text_lower).start(),) if hits else None # if multiple topics mentioned get hits list and check against list their position, pick first 

def _bump(memory: PersonaMemory, topic : str, points: float) -> None: 
    """
    update topic engaegment points by `points` amount and floored at 0 so never goes negative!
    """

    memory.engagement[topic] = max(0.0, memory.engagement.get(topic, config.ENGAGE_BASE) + points)


def apply_reaction(memory: PersonaMemory, reaction: Reaction): 
    """
    apply reaction to memory by appending it to history and then bumping engagment points
    reacted to topic moves by the specific reaction points and in an explicit switch the requested topic gets the points and the 
    topic teh user wants to move away from takes switch away penalty  
    """
    points = REACTION_POINTS[reaction.type] # retrieve point value for specific reaction from config dict 

    if reaction.requested_topic: # if reaction request topic switch 
        _bump(memory, reaction.requested_topic, points) # based on reaction to requested topic bump points respectively
        _bump(memory, reaction.topic, config.ENGAGE_SWITCH_AWAY) # current topic that user wants to switch away from gets penalty 
    else: 
        _bump(memory, reaction.topic, points) # else normal esituation point bump
    
    memory.reactions.append(reaction) # append reaction to reactions section of the PersonaMmeory object

def next_topic(memory: PersonaMemory, active_topics: list[str]): 
    """
    pick next topic uses a quite greedy,  max engagement points method so in the current session active topics we just choose the one with the most engagement 
    in the case of an engagement point tie we just take the first topic in the list, so whatever user listed first in session persona is what we use
    """

    if not active_topics: 
        raise ValueError("active topics are empty ")
    return max(active_topics, key = lambda topic: memory.engagement.get(topic, config.ENGAGE_BASE)) #iterable is active topics from engagement dictionary get engagement points and get argmax