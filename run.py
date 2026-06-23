"""
pipeline orchestrator

python run.py         --> full pipeline on my testcase `personas/ruhani.json`, script only
python run.py --audio --> full pipeline and audio generation
python run.py --persona path --> use a different persona file
"""

from __future__ import annotations
import argparse
from pathlib import Path
from personacast.models import load_persona
from personacast.pipeline.run import run_pipeline

def main():
    """
    cli setup using argparse, defining flags you can pass using run.py
    """
    ap = argparse.ArgumentParser(description="PersonaCast Phase 1 pipeline")
    ap.add_argument("--persona", default="personas/ruhani.json", help="path to a persona JSON")
    ap.add_argument("--audio", action="store_true", help="also render audio via TTS")
    args = ap.parse_args()

    persona = load_persona(Path(args.persona))
    pipe_state = run_pipeline(persona, audio=args.audio)


if __name__ == "__main__":
    main()
