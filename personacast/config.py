
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

### Interactive Session config
TURN_SECONDS = 60
WORDS_PER_TURN = WORDS_PER_MINUTE * TURN_SECONDS // 60 
MAX_ITERATIONS = 8
MEMORY_DIR = os.getenv("PERSONACAST_MEMORY_DIR", "personas/memory")
## engagement points per reaction type 
ENGAGE_QUESTION = 2.0
ENGAGE_COMMENT = 1.0 
ENGAGE_NONE = -1.0 
ENGAGE_SWITCH_AWAY = -2.0 
ENGAGE_BASE = 1.0 # for a new topic automatically given 1 point 

SUMMARIZE_TURNS = os.getenv("PERSONACAST_SUMMARIZE_TURNS", "1") in ("1", "true", "yes")# flag to basically say to use LLM call to get gist, presents options that work any otherwise considered flag off 
RECENT_TURNS_CONTEXT = int(os.getenv("PERSONACAST_RECENT_TURNS", "4")) # determined how many of the most recent turn gists to feed into next generation turns[-4]
TURN_MODE = os.getenv("PERSONACST_TURN_MODE", "variety") # set default witin turn style to variety


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

### changing TTS config to Gemini Live API, future work would be realtime streaming using websocket 
TTS_API_KEY = os.getenv("GEMINI_API_KEY", "") or LLM_API_KEY
TTS_MODEL = os.getenv("PERSONACAST_TTS_MODEL", "gemini-3.1-flash-live-preview")
TTS_VOICE = os.getenv("PERSONACAST_TTS_VOICE", "Kore")
TTS_SAMPLE_RATE = 24000 # dictating live api output audio to be at 24khz 
TTS_CHUNK_CHARS = 10000 # maximum number of character of text sent to TTS model in one go 


### stremalit records wav clip, normalize to 16khz resampling it through a function, send to gemini live and grab its text transcription
STT_API_KEY = TTS_API_KEY
STT_MODEL = os.getenv("PERSONACAST_STT_MODEL", TTS_MODEL) # using the same gemini-3.1-flash-live-preview
STT_INPUT_SAMPLE_RATE = 16000
