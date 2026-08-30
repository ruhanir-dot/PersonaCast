"""
main changes is that we enforce a sliding window of 60s where  there can be 15 calls done within a minute of first call  of those 60 sec if 
but main point I was conccerned about was that the background calls shouldnt take the sapace of critical calls 
do deal with this we determine the type  of call if its abackgrojnd call and theres more than 2 slots left 
we allow the calls to happen and burst the calls 

"""




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

### how long web search is allowed to still be running
WEB_FALLBACK_TIMEOUT_SECONDS = float(os.getenv("PERSONACAST_WEB_TIMEOUT", "6.0"))

### length of pregenerated opener
OPENER_WORDS = int(os.getenv('PERSONACAST_OPENER_WORDS', '35'))
RECENT_TURNS_CONTEXT = int(os.getenv("PERSONACAST_RECENT_TURNS", "4")) # determined how many of the most recent turn gists to feed into next generation turns[-4]
TURN_MODE = os.getenv("PERSONACST_TURN_MODE", "variety") # set default witin turn style to variety


### LLM config
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
MODEL = os.getenv("PERSONACAST_MODEL", "")

# free tier rate limit adherence 
LLM_MAX_RETRIES = 6
LLM_BACKOFF_BASE_SECONDS = 2.0
LLM_REQUEST_TIMEOUT_SECONDS = 60.0
LLM_MIN_INTERVAL_SECONDS = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0.2")) # interval betwtween requests, reducing throttle time, for request bursting
LLM_MAX_RPM = int(os.getenv('LLM_MAX_RPM', '15')) # max requests per minute for geminie free tier
LLM_RPM_WINDOW_SECONDS = float(os.getenv('LLM_RPM_WINDOW_SECONDS', '60.0')) # sliding window of 60 seconds at first request 60 seconds starts 15 spots allotted, once 60 sec up can make anotehr 15 calls. 

### slotting system for the 15 requsts that can be made in the minute some calls more critical than other  
## distinctevely a opener generating call is less important than an interruption interpretation call/turn generation so we give those priority  
LLM_BACKGROUND_RESERVE = int(os.getenv("LLM_BACKGROUND_RESERVE", "2"))


### Retrieval config 
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


### Query results config 
QUERIES_PER_TOPIC = 3    # how many search queries generate per topic
RESULTS_PER_QUERY = 4

### recency and arxiv config 
SEARCH_RECENCY_DAYS = 30
ARXIV_RATE_LIMIT_SECONDS = 3.0 
ARXIV_REQUEST_TIMEOUT_SECONDS = 15.0
ARXIV_MAX_RETRIES = 1

### source retrieval via api
RETRIEVAL_MAX_CANDIDATES = 40
RETRIEVAL_SOURCES = ("web", "arxiv") # what the planning agent is allowed to choose from

### flag for faster interaction mostly for just my checking to see if its actually workig faster
FAST_INTERACTION = os.getenv("PERSONACAST_FAST_INTERACTION", "1") in ("1", "true", "yes")


### Keywords to know when to use Arxiv, later have agent autonomously decide
### this is now stale(ish) just a fallback whaen the agentic retrieval is off or planning agent has an error
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

### source pool cache because don't want to consue calls just on building the pool as thats the offline generation part 
POOL_CACHE = os.getenv("PERSONACAST_POOL_CACHE", "1") in ("1", "true", "yes")
POOL_CACHE_DIR = os.getenv("PERSONACAST_POOL_CACHE_DIR", "personas/pool")
POOL_CACHE_TTL_HOURS = float(os.getenv("PERSONACAST_POOL_CACHE_TTL", "24"))

### output/tts config 
RUNS_DIR = os.getenv("PERSONACAST_RUNS_DIR", "runs")

### TTS backend, local piper much faster than using gemini live api! 
TTS_BACKEND = os.getenv("PERSONACAST_TTS_BACKEND", 'piper')
PIPER_VOICE_PATH = os.getenv("PERSONACAST_PIPER_VOICE", "models/piper/en_US-lessac-low.onnx")

### STT backend, using whisper, local as well
STT_BACKEND = os.getenv("PERSONACAST_STT_BACKEND", 'whisper') # engine
STT_MODEL_SIZE = os.getenv("PERSONACAST_STT_MODEL_SIZE", "tiny.en") # whisper setting, picks what model file whisper is loading 
## can bump to base.en, if tinyen is to inaccurate in STT

STT_INPUT_SAMPLE_RATE = 16000

### Mic settinsg always on interruption capture config

MIC_ALWAYS_ON = os.getenv("PERSONACAST_MIC_ALWAYS_ON", "0") in ("1", "true", "yes")
VAD_ONSET_MS = int(os.getenv('PERSONACAST_VAD_ONSET_MS', '150')) #how much susteained voice has to be playing before we decide the user has actually started talking 
VAD_SILENCE_MS = int(os.getenv('PERSONACAST_VAD_SILENCE_MS', '2500')) # how many seconds of silence to wait before deciding user is done talking
VAD_PREROLL_MS = int(os.getenv('PERSONACAST_VAD_PREROLL_MS', '300')) # snallk rolling biffer of last 300ms of raw audio to make sure to save entire speech segment
VAD_AGGRESSIVENESS = float(os.getenv("PERSONACAST_VAD_AGGRESSIVENESS", "0.65")) # silero speech probability a chunk must clear to count as voiced

MIC_MIN_UTTERANCE_MS = int(os.getenv("PERSONACAST_MIC_MIN_UTTERANCE_MS", "250")) # measured on voiced audio only, so short affirmations like yeah/mhm still register
MIC_MAX_UTTERANCE_MS = int(os.getenv("PERSONACAST_MIC_MAX_UTTERANCE_MS", "30000")) # this is a max cap to make sure we dont register speaker playback 
MIC_WS_PORT = int(os.getenv("PERSONACAST_MIC_WS_PORT", "8765")) # the websocket server browser mic capture connects to directly  instead of stremalit component

### Bridge Bank and persona style drift 
STYLE_AXES = [a.strip() for a in os.getenv(
    "PERSONACAST_STYLE_AXES", "formality,energy,warmth,technical_register,brevity",
).split(",") if a.strip()] # style axes of personas 
STYLE_UPDATE_MAX_STEP = float(os.getenv("PERSONACAST_STYLE_UPDATE_MAX_STEP", "0.15"))
BRIDGE_BANK_SIZE = int(os.getenv("PERSONACAST_BRIDGE_BANK_SIZE", "10")) # number of candidates for each reaction type 
BRIDGE_RECENCY_WINDOW = int(os.getenv('PERSONACAST_BRIDGE_RECENCY_WINDOW', '3')) # exclude the last N picked bridges in the scored narrative pools so we arent cycling the same bridges and repeats 







# ----------------------    STALE     -------------------------------

# STALE - OLD API USAGE (still here if want to revert back to it)
### changing TTS config to Gemini Live API, future work would be realtime streaming using websocket 
TTS_API_KEY = os.getenv("GEMINI_API_KEY", "") or LLM_API_KEY
TTS_MODEL = os.getenv("PERSONACAST_TTS_MODEL", "gemini-3.1-flash-live-preview")
TTS_VOICE = os.getenv("PERSONACAST_TTS_VOICE", "Kore")
TTS_SAMPLE_RATE = 24000 # dictating live api output audio to be at 24khz 
TTS_CHUNK_CHARS = 10000 # maximum number of character of text sent to TTS model in one go 


# STALE - OLD API USAGE (still here if want to revert back to it)
### stremalit records wav clip, normalize to 16khz resampling it through a function, send to gemini live and grab its text transcription
STT_API_KEY = TTS_API_KEY
STT_MODEL = os.getenv("PERSONACAST_STT_MODEL", TTS_MODEL) # using the same gemini-3.1-flash-live-preview
