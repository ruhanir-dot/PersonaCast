import json
import os
import time
import wave
from pathlib import Path
from typing import Any

from src.offline.generate_main_narrative import MAIN_NARRATIVES_FILE
from src.offline.predict_user_actions import predict_user_questions
from src.utils import PROJECT_ROOT, llm_json


DATA_DIR = PROJECT_ROOT / "data" / "output"
USER_PROFILE_FILE = DATA_DIR / "user_profile.json"
OUTPUT_FILE = DATA_DIR / "candidate_trunks.json"
AUDIO_DIR = DATA_DIR / "audio"

QUESTIONS_PER_TRUNK = 3


def load_user_profile() -> dict[str, Any]:
    with USER_PROFILE_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_candidate_pool(candidate_pool: dict[str, Any]) -> None:
    """Save the current result so completed topics are not lost."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_name(
        f"{OUTPUT_FILE.stem}.{os.getpid()}.tmp"
    )

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(candidate_pool, file, ensure_ascii=False, indent=2)

    for attempt in range(5):
        try:
            os.replace(temporary_file, OUTPUT_FILE)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.1)


def load_main_narratives() -> dict[str, Any]:
    if not MAIN_NARRATIVES_FILE.exists():
        raise FileNotFoundError(
            "Main narratives are missing. Run "
            "python -m src.offline.generate_main_narrative first."
        )
    with MAIN_NARRATIVES_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_story_chunks(
    *,
    story_title: str,
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for step in steps:
        focus = str(step["focus"]).strip()
        source_segment = str(step["source_segment"]).strip()
        result = llm_json(
            prompt=(
                "Rewrite this source section as one clear spoken English podcast "
                "trunk. Keep the supplied facts and sequence. Remove source noise, "
                "advertisements, and repeated filler. Do not add information. Tell the events in third person; "
                "do not use I, we, or you. Keep short sources short; for longer "
                "sources, write about 100 words in three to five natural sentences.\n\n"
                f"Story title:\n{story_title}\n\n"
                f"Source section {step['chunk_order']} of {len(steps)}:\n"
                f"{source_segment}"
            ),
            system="Return valid JSON only.",
            temperature=0.1,
            max_output_tokens=300,
            json_schema={
                "type": "object",
                "properties": {
                    "script": {"type": "string", "maxLength": 1200},
                },
                "required": ["script"],
                "additionalProperties": False,
            },
        )
        script = " ".join(str(result["script"]).split())
        chunks.append(
            {
                "title": story_title,
                "focus": focus,
                "script": script,
                "chunk_order": int(step["chunk_order"]),
                "chunk_role": focus,
                "search_query": step["search_query"],
                "source_item": step["source_item"],
            }
        )

    return chunks


def generate_questions(
    final_script: str,
    user_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    predicted_questions = predict_user_questions(
        trunk_script=final_script,
        user_profile=user_profile,
    )

    return [
        {
            "question": item["question"],
            "answer": "",
        }
        for item in predicted_questions[:QUESTIONS_PER_TRUNK]
    ]


def add_candidate_ids(
    topic_number: int,
    trunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    for trunk_number, trunk in enumerate(trunks, start=1):
        trunk["trunk_id"] = (
            f"topic_{topic_number:02d}_trunk_{trunk_number:02d}"
        )

        for question_number, question_answer in enumerate(
            trunk["question_answers"],
            start=1,
        ):
            question_answer["qa_id"] = (
                f"topic_{topic_number:02d}_trunk_{trunk_number:02d}_"
                f"qa_{question_number:02d}"
            )

    return trunks


def audio_file_has_frames(audio_file: Path) -> bool:
    if not audio_file.exists():
        return False

    try:
        with wave.open(str(audio_file), "rb") as wav_file:
            return wav_file.getnframes() > 0
    except (OSError, wave.Error):
        return False


def set_english_tts_voice(engine: Any) -> None:
    for voice in engine.getProperty("voices"):
        languages = " ".join(
            (
                language.decode(errors="ignore")
                if isinstance(language, bytes)
                else str(language)
            )
            for language in (getattr(voice, "languages", []) or [])
        )
        description = " ".join(
            [
                str(getattr(voice, "id", "")),
                str(getattr(voice, "name", "")),
                languages,
            ]
        ).casefold()

        if "en-us" in description or "english" in description:
            engine.setProperty("voice", voice.id)
            return

    raise RuntimeError(
        "No English TTS voice is installed on this computer."
    )


def generate_candidate_audio(
    trunks: list[dict[str, Any]],
) -> None:
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError(
            "pyttsx3 is not installed. Install it with: pip install pyttsx3"
        ) from exc

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_jobs: list[tuple[dict[str, Any], Path, Path]] = []

    engine = pyttsx3.init()
    set_english_tts_voice(engine)
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    for trunk in trunks:
        trunk_audio_file = AUDIO_DIR / f"{trunk['trunk_id']}.wav"
        trunk_temporary_file = AUDIO_DIR / (
            f"{trunk['trunk_id']}.{os.getpid()}.tmp.wav"
        )
        engine.save_to_file(
            str(trunk["script"]).strip(),
            str(trunk_temporary_file),
        )
        audio_jobs.append(
            (trunk, trunk_temporary_file, trunk_audio_file)
        )

    engine.runAndWait()
    engine.stop()

    for candidate, temporary_file, audio_file in audio_jobs:
        if audio_file_has_frames(temporary_file):
            os.replace(temporary_file, audio_file)
            candidate["audio_file"] = str(
                audio_file.relative_to(DATA_DIR)
            ).replace("\\", "/")
            candidate["audio_status"] = "ready"
        else:
            if temporary_file.exists():
                temporary_file.unlink()
            candidate["audio_file"] = None
            candidate["audio_status"] = "failed"

    ready_count = sum(
        candidate["audio_status"] == "ready"
        for candidate, _, _ in audio_jobs
    )
    print(
        f"Generated {ready_count}/{len(audio_jobs)} audio files.",
        flush=True,
    )


def main() -> None:
    user_profile = load_user_profile()
    main_narrative_pool = load_main_narratives()

    candidate_pool: dict[str, Any] = {
        "configuration": {
            "story_count": len(main_narrative_pool["stories"]),
            "trunks_per_story": "dynamic",
            "questions_per_trunk": QUESTIONS_PER_TRUNK,
            "generation_mode": "daily_youtube_source_story",
        },
        "topics": [],
    }
    save_candidate_pool(candidate_pool)

    for story_number, narrative in enumerate(
        main_narrative_pool["stories"],
        start=1,
    ):
        print(
            f"[{story_number}/{len(main_narrative_pool['stories'])}] "
            "Generating trunks.",
            flush=True,
        )

        try:
            trunks = generate_story_chunks(
                story_title=narrative["story_title"],
                steps=narrative["steps"],
            )
            for trunk in trunks:
                trunk["question_answers"] = generate_questions(
                    final_script=trunk["script"],
                    user_profile=user_profile,
                )
                trunk["personalized_seed"] = narrative["selected_seed"]
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Skipping main narrative: {exc}", flush=True)
            continue

        topic_number = len(candidate_pool["topics"]) + 1
        trunks = add_candidate_ids(topic_number, trunks)
        generate_candidate_audio(trunks)
        candidate_pool["topics"].append(
            {
                "topic_id": f"topic_{topic_number:02d}",
                "topic": narrative["story_title"],
                "story_reason": "Built from sequential related sources.",
                "selected_seeds": [narrative["selected_seed"]],
                "trunks": trunks,
            }
        )
        save_candidate_pool(candidate_pool)

        print(
            f"[{topic_number}/{len(main_narrative_pool['stories'])}] Finished: "
            f"{narrative['story_title']}",
            flush=True,
        )

    if not candidate_pool["topics"]:
        raise ValueError(
            "Could not generate trunks from the main narratives."
        )

    candidate_pool["configuration"]["story_count"] = len(
        candidate_pool["topics"]
    )
    save_candidate_pool(candidate_pool)

    total_trunks = sum(
        len(topic["trunks"]) for topic in candidate_pool["topics"]
    )
    total_questions = sum(
        len(trunk["question_answers"])
        for topic in candidate_pool["topics"]
        for trunk in topic["trunks"]
    )

    print(
        "Offline candidate generation completed: "
        f"{len(candidate_pool['topics'])} daily stories, "
        f"{total_trunks} connected trunks, "
        f"{total_questions} Q&A candidates.\n"
        f"Saved to: {OUTPUT_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
