<img width="1920" height="510" alt="personaCast" src="https://github.com/user-attachments/assets/6c9774f7-c810-4b71-99a2-3249f5eac3c1" />

# PersonaCast

An AI system that generates a personalized ~20-minute "news podcast" about the topics a user cares about, then lets the listener pause and ask follow-up questions grounded in the same sources.

A user persona  constructed of topics, per-topic expertise level, tone, and an "avoid" list is the explicit feedbck we take as input. From that persona the pipeline retrieves recent material from the web and arXiv, curates, and writes a single flowing episode tailored to the listener persona profile. The listener can then interrupt with a question (an overarching goal for the future is using these questions and evaluate them as a source of implicit feedback) and get a short, source-grounded answer.

> Note: Reddit is intentionally not a source in this implementation, API requires manual approval

---

## Project status

**Phase 1 — RAG pipeline & generation: complete and deployed.**
Persona → topic planning → per-topic query generation → retrieval (web + conditional arXiv) → extractive map-reduce curation → per-topic segment writing → stitch into one episode → optional local audio. Runnable via CLI and a Streamlit app.

**Phase 2 — interactivity & evaluation: in progress.**
- **Mid-podcast Q&A** (`personacast/pipeline/qa.py`, wired into the app): the listener "pauses" and asks a question. It's answered first from the episode's curated sources; if those don't cover it, it falls back to a fresh web search. "Pause and ask" not real time where user can enter questions about script.
- **Expertise-injection evaluation** (`eval/eval.py`): holds a persona constant and varies only the expertise level, asking the same questions over the same frozen sources to test whether personalization actually changes the answer.

### What we can focus on
- **Novel interaction.** Ideas about personalization methods that can be used for more peronalized and tailored answers for users, detailed in a seperate writeup with an idea had and abalation studies that could be done, also detailing similar work I read pertaining to personalized LLM usage and what their task setup and ground truth was looking like!
---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your keys (any OpenAI-compatible LLM provider works):

```bash
LLM_BASE_URL=https://api.cerebras.ai/v1
LLM_API_KEY=your-llm-key
PERSONACAST_MODEL=gpt-oss-120b
TAVILY_API_KEY=your-tavily-key
```

---

## CLI usage

Runs the full generation pipeline and writes outputs to `runs/<timestamp>/` (script text + per-topic sources + per-stage state).

```bash
python run.py                      # full pipeline on personas/ruhani.json, script only
python run.py --audio              # also render audio narration (local TTS)
python run.py --persona path.json  # use a different persona file
```

## Local app (Streamlit)

```bash
streamlit run app.py
```

Enter a persona in the sidebar and click **Generate script** — a run takes ~6–7 minutes (live retrieval + LLM calls), with a live stage ticker. Then you can:
- **Ask a question** against the episode's sources (with optional web fallback).
- **Generate audio** as a separate, local-only step.

### Audio (local only)

Audio narration uses Kokoro TTS and runs only locally — the ~350 MB voice model can't be deployed on Streamlit Cloud. To enable it:

1. Install [espeak-ng](https://github.com/espeak-ng/espeak-ng) (macOS: `brew install espeak-ng`).
2. Download the model files into a `models/` folder:
   - [`kokoro-v1.0.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx)
   - [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)
3. Install the audio packages: `pip install kokoro-onnx soundfile`.

Then run `python run.py --audio` or use the audio button in the app.

---

## Evaluation

The expertise-injection eval asks the same questions over a **frozen** set of sources, changing only the listener's expertise level, to check that personalization actually changes the answer (web fallback is off so the sources can't vary).

```bash
python eval/eval.py
```

- **Personas:** `personas/eval/ml_{beginner,intermediate,advanced}.json` (identical except expertise)
- **Sources:** `eval/frozen_sources.json`
- **Output:** `eval/results/<timestamp>/answers.md` (side-by-side table) + `interactions.json` (raw log)
