<img width="1920" height="510" alt="personaCast" src="https://github.com/user-attachments/assets/6c9774f7-c810-4b71-99a2-3249f5eac3c1" />

AI system that generates a personalized ~20-minute "news podcast" about topics a user cares about. 
 Phase 1: RAG Implementation

Note: no Reddit in implementation as reddit api needs approval before usage


## Setup for CLI

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

## CLI usage

Runs the full pipeline and writes outputs to `runs/<timestamp>/` (script text + per-topic sources).

```bash
python run.py                      # full pipeline on example personas/ruhani.json, script only
python run.py --audio              # also render audio narration (local TTS)
python run.py --persona path.json  # use a different persona file
```

## Local app (Streamlit)

```bash
streamlit run app.py
```

Enter a persona in the sidebar and click **Generate script**. A run takes ~6–7 minutes (live retrieval + LLM calls). Script generation runs anywhere; audio is a separate, local-only step.

### Audio (local only)

Audio narration uses Kokoro TTS and runs only locally — the ~350 MB voice model can't be deployed on Streamlit Cloud. To enable it:

1. Install [espeak-ng](https://github.com/espeak-ng/espeak-ng) (macOS: `brew install espeak-ng`).
2. Download the model files into a `models/` folder:
   - [`kokoro-v1.0.onnx`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx)
   - [`voices-v1.0.bin`](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin)
3. Install the audio packages: `pip install kokoro-onnx soundfile`.

Then run `python run.py --audio` or use the audio button in the app.
