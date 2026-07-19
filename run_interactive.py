"""
interactive personacast session CLI 

"""

import argparse
from personacast.models import load_persona
from personacast.pipeline.interactive import InteractiveSession

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="personas/ruhani.json")
    parser.add_argument("--context", default="")
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    persona = load_persona(args.persona)
    persona.additional_context = args.context  # per-session vibe

    session = InteractiveSession(persona, on_stage=lambda label: print(f"  → {label}"))
    session.start() # builds pool once

    while not session.done: # done method used as attribute to check when done
        turn = session.next_segment() # build next segment
        print(f"\n--- Turn {turn.iteration} · {turn.topic} ---\n{turn.text}\n")
        reaction = input("React (Enter=none · end with '?' to ask · ':q' to quit): ")
        if reaction.strip() == ":q":
            break
        session.submit_reaction(reaction)

    session.finish()
    print("\nSession saved.")

if __name__ == "__main__":
    main()