# PersonaCast

Personalized spoken news, generated from a listener profile and interruptible mid-episode.

## Overview

A persona — topics, per-topic expertise, tone, and an avoid list — drives retrieval of recent material from the web and arXiv. Retrieved sources are scored and curated per topic, then written into spoken host turns calibrated to the listener's expertise.

Interactive version creates short turns delivered one at a time. The listener can pause mid-turn, react, or ask a question answered from the episode's own sources. Reactions update a per-persona memory that persists across sessions and steers subsequent turns.

Speech synthesis and transcription run locally. 

## Setup

```bash
pip install -r requirements.txt
```

`.env` in the project root — any OpenAI-compatible provider:

```bash
LLM_BASE_URL=https://api.cerebras.ai/v1
LLM_API_KEY=...
PERSONACAST_MODEL=gpt-oss-120b
TAVILY_API_KEY=...
```

Text-to-speech needs its voice file fetched once (~60 MB, gitignored). Transcription downloads its model on first use.

```bash
export SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())")
python -m piper.download_voices en_US-lessac-low
mkdir -p models/piper && mv en_US-lessac-low.onnx* models/piper/
```

## Usage

```bash
python run.py [--persona path.json] [--audio]        # baseline episode
python run_interactive.py [--persona path.json]      # interactive, terminal
streamlit run app.py                                 # baseline, browser
streamlit run app_interactive.py                     # interactive, browser
```

Outputs are written to `runs/<timestamp>/`: script text, per-topic sources, per-stage state, and per-turn snapshots.

## Configuration

Both flags default off; the system runs its original path unless enabled. The interactive app exposes them as sidebar toggles. (Mostly used for debugging on my side to see if the agentic version is faster than our previous version)

| Variable | Effect |
| --- | --- |
| `PERSONACAST_AGENTIC_RETRIEVAL=1` | Query construction and source selection (web / arXiv) are decided per topic by a LangGraph agent rather than a keyword heuristic. Sources previously shown to the listener are excluded. Writes `runs/<id>/retrieval.json`. |
| `PERSONACAST_AGENTIC_INTERACTION=1` | Listener reactions are interpreted by a single structured LLM call — intent, sentiment, engagement, topic-switch requests — rather than punctuation and regex rules. |


## Structure

```
personacast/
  agents/     retrieval graph, reaction interpretation
  pipeline/   topics, queries, retrieval, curation, script, memory, tts, stt
  llm/        rate-limited OpenAI-compatible client
personas/     persona definitions; memory/ holds persistent per-persona state
eval/         expertise-injection evaluation
runs/         per-run outputs
```
