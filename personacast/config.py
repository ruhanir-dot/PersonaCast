
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

### scripth length budget config
# 20 minutes at 155 words/min, 3100 words default
WORDS_PER_MINUTE = 155
TARGET_MINUTES = 20
TOTAL_WORDS = WORDS_PER_MINUTE * TARGET_MINUTES

def per_topic_word_budget(n_topics): 
    if n_topics<= 0: 
        raise ValueError("n_topics must be >= 1")
    
    return TOTAL_WORDS//n_topics 

### LLM config
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("PERSONACAST_MODEL", "")
# free tier rate limit adherence 
LLM_MAX_RETRIES = 6
LLM_BACKOFF_BASE_SECONDS = 2.0 #
LLM_REQUEST_TIMEOUT_SECONDS = 60.0
LLM_MIN_INTERVAL_SECONDS = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "4.5")) # interval betwtween requests

### Retrieval config 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


### Query results config 
QUERIES_PER_TOPIC = 3    # how many search queries generate per topic
RESULTS_PER_QUERY = 4

### recency and arxiv config 
SEARCH_RECENCY_DAYS = 30
ARXIV_RATE_LIMIT_SECONDS = 3.0 

### Keywords to know when to use Arxiv, later have agent autonomously decide
STEM_HINT_KEYWORDS = (
    "artificial intelligence", "machine learning", "deep learning",
    "neural network", "large language model", "computer vision",
    "natural language processing", "software engineering",
    "electrical engineering", "mechanical engineering",
    "bioinformatics", "biophysics", "biochemistry", "biotechnology", "bioengineering",
    "ai", "ml", "llm", "nlp", "rag", "gpt", "chatgpt", "chatbot",
    "recommender", "agent", "algorithm", "transformer", "robot", "robotics",
    "neural", "quantum", "physics", "chemistry", "biology", "astronomy",
    "mathematics", "statistics", "genetics", "geology", "climatology",
    "oceanography", "programming", "coding",
)

### output/tts config 
RUNS_DIR = os.getenv("PERSONACAST_RUNS_DIR", "runs")
TTS_VOICE = "af_heart"
KOKORO_MODEL_PATH = os.getenv("KOKORO_MODEL_PATH", "models/kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.getenv("KOKORO_VOICES_PATH", "models/voices-v1.0.bin")
TTS_CHUNK_CHARS = 1500  # max chars per synth call; kokoro-onnx has a per-call token limit