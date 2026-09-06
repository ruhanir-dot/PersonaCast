"""
pydantic mirror of the artifcats written by the personalized podcast agent 
its a data contract, and a reader of what the producer creates 
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from ..models import CuratedItem

class TrunkQA(BaseModel): 
    """
    object defining one predicted question with prewritten and rendered answer
    """

    question: str 
    answer: str 
    qa_id: str = ""
    audio_file: str | None = None
    audio_status: str = 'missing'

    question_type: str | None = None
    probability: float | None = None

    @property

    def has_audio(self) -> bool:
        # check whether the path exists or not downstream usage i
        return self.audio_status == 'ready' and  bool(self.audio_file) # used downstream to check if listener interruption matches

class Seed(BaseModel): 
    """
    feed item seed that we derive steer news query with
    """
    feed_id: str
    source_type:str
    text:str
    original_text: str = ''
    creator: str | None = None
    url: str | None = None
    occurred_at: str | None = None
    interaction_count: int = 1
    topic_similarity: float = 0.0
    mmr_score: float = 0.0

class SourceItem(BaseModel): 
    """
    news article we ground the trunk draft in
    """

    source_type: str = 'news'
    title: str
    summary : str = ""
    url: str 
    published_at: str = "" 
    google_news_url: str = ""
    article_text: str = ""
    article_word_count: int = 0
    relevance_score: float = 0.0


class Trunk(BaseModel): 
    """
    1 of 5 candidate podcast segment for one podcast topic 
    script is a draft that we pass downstream at generate_turn, to rewrite for script
    """
    trunk_id: str
    title: str 
    focus : str = ""
    script: str
    search_query: str = ""
    personalized_seed: Seed
    source_item: SourceItem
    question_answers: list[TrunkQA] = Field(default_factory=list) # list of generated question answer objects attached to this trunk segment 3 predicted questions
    audio_file: str | None = None
    audio_status: str = "missing"

    def to_curated_item(self, *, summary_chars: int = 4000) -> CuratedItem: 
        """
        lets offline source from a trunk be a curated Item, for usage in source grounding qa answers
        """

        body = self.source_item.article_text or self.source_item.summary

        return CuratedItem(source="trunk", title=self.source_item.title, url=self.source_item.url, summary=body[:summary_chars])


class TrunkTopic(BaseModel):
    """
    each topic gets 8 seeds and we walk through the 8 seeds to build trunok from each stop at 5
    """
    topic_id: str 
    topic: str 
    selected_seeds: list[Seed] = Field(default_factory = list)
    trunks: list[Trunk] = Field(default_factory= list)

class TrunkConfiguration(BaseModel): 
    topic_count: int = 0 
    trunks_per_topic: int = 0 
    questions_per_trunk: int = 0 


class TrunkPool(BaseModel):
    configuration: TrunkConfiguration = Field(default_factory=TrunkConfiguration)
    topics: list[TrunkTopic] = Field(default_factory = list)

    @property
    def topic_names(self) -> list[str]:
        return [topic.topic for topic in self.topics]

    def by_topic(self) -> dict[str, TrunkTopic]:
        return {topic.topic: topic for topic in self.topics}

    def all_trunks(self) -> list[Trunk]:
        return [trunk for topic in self.topics for trunk in topic.trunks]

    def trunk_by_id(self, trunk_id: str) -> Trunk | None:
        for trunk in self.all_trunks():
            if trunk.trunk_id == trunk_id:
                return trunk
        return None

    def as_source_pool(self, *, summary_chars: int = 4000) -> dict[str, list[CuratedItem]]:
        """
        shape downstream online consumer expects so we shape into that
        """

        return {
                    topic.topic: [
                        trunk.to_curated_item(summary_chars=summary_chars)
                        for trunk in topic.trunks
                    ]
                    for topic in self.topics
                }

    
class EmbeddingIndexMetadata(BaseModel):
    model_name: str
    embedding_dimension: int
    normalized_embeddings: bool = True
    trunk_embedding_count: int = 0
    question_embedding_count: int = 0

    

class EmbeddingIndex(BaseModel):
    # used of row lookup when comparing utteranc  embeddings and the question embeddings 
    # also for shape validation, and checking dimension chshape and model that produced embeddings
    metadata: EmbeddingIndexMetadata
    trunk_ids: list[str] = Field(default_factory=list)
    question_ids: list[str] = Field(default_factory=list)



