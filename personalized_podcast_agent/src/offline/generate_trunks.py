import json
import os
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.offline.generate_main_narrative import MAIN_NARRATIVES_FILE
from src.offline.predict_user_actions import predict_user_questions
from src.utils import PROJECT_ROOT, llm_json


DATA_DIR = PROJECT_ROOT / "data" / "output"
USER_PROFILE_FILE = DATA_DIR / "user_profile.json"
OUTPUT_FILE = DATA_DIR / "candidate_trunks.json"
TRUNK_EMBEDDINGS_FILE = DATA_DIR / "candidate_trunk_embeddings.npy"
QUESTION_EMBEDDINGS_FILE = DATA_DIR / "candidate_question_embeddings.npy"
EMBEDDING_IDS_FILE = DATA_DIR / "candidate_embedding_ids.json"
AUDIO_DIR = DATA_DIR / "audio"

TRUNKS_PER_TOPIC = 5
QUESTIONS_PER_TRUNK = 3
EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


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
    if len(steps) != TRUNKS_PER_TOPIC:
        raise ValueError(
            f"Expected {TRUNKS_PER_TOPIC} researched steps, received {len(steps)}."
        )

    chunks: list[dict[str, Any]] = []

    for step in steps:
        focus = str(step["focus"]).strip()
        source_segment = str(step["source_segment"]).strip()
        result = llm_json(
            prompt=(
                "Rewrite this source section as one clear spoken English podcast "
                "segment. Keep the supplied facts and sequence. Remove source noise, "
                "advertisements, and repeated filler. Do not add information. Tell the events in third person; "
                "do not use I, we, or you. Write about 100 words in three to five "
                "natural sentences.\n\n"
                f"Story title:\n{story_title}\n\n"
                f"Source section {step['chunk_order']} of {TRUNKS_PER_TOPIC}:\n"
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


def generate_question_answers(
    source_item: dict[str, Any],
    final_script: str,
    user_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    predicted_questions = predict_user_questions(
        trunk_script=final_script,
        user_profile=user_profile,
    )

    questions_with_ids = [
        {
            "candidate_id": f"q{index}",
            "question": item["question"],
        }
        for index, item in enumerate(predicted_questions)
    ]
    candidate_ids = [
        item["candidate_id"]
        for item in questions_with_ids
    ]

    result = llm_json(
        prompt=(
            "Article title:\n"
            f"{source_item.get('title', '')}\n\n"
            "Article content:\n"
            f"{source_item['article_text']}\n\n"
            "Predicted user questions:\n"
            f"{json.dumps(questions_with_ids, ensure_ascii=False, indent=2)}\n\n"
            "Generate one concise article-supported answer for every "
            "predicted question. Keep each candidate_id unchanged."
        ),
        system=(
            "Answer the supplied predicted questions using the article. "
            "Return valid JSON only."
        ),
        temperature=0.1,
        max_output_tokens=900,
        json_schema={
            "type": "object",
            "properties": {
                "answers": {
                    "type": "array",
                    "minItems": QUESTIONS_PER_TRUNK,
                    "maxItems": QUESTIONS_PER_TRUNK,
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "enum": candidate_ids,
                            },
                            "answer": {
                                "type": "string",
                                "maxLength": 700,
                            },
                        },
                        "required": ["candidate_id", "answer"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["answers"],
            "additionalProperties": False,
        },
    )

    answers_by_id: dict[str, str] = {}

    for item in result["answers"]:
        candidate_id = str(item.get("candidate_id", "")).strip()
        answer = str(item.get("answer", "")).strip()

        # Ignore malformed IDs, duplicate IDs, or empty answers.
        if (
            candidate_id not in candidate_ids
            or candidate_id in answers_by_id
            or not answer
        ):
            continue

        answers_by_id[candidate_id] = answer

    missing_questions = [
        item
        for item in questions_with_ids
        if item["candidate_id"] not in answers_by_id
    ]

    # A small local model may occasionally duplicate an ID.
    # Retry only the missing answers, so one bad response does not stop the daily job.
    if missing_questions:
        try:
            retry_result = llm_json(
                prompt=(
                    "Article title:\n"
                    f"{source_item.get('title', '')}\n\n"
                    "Article content:\n"
                    f"{source_item['article_text']}\n\n"
                    "Answer these questions in exactly the same order:\n"
                    f"{json.dumps([item['question'] for item in missing_questions], ensure_ascii=False, indent=2)}"
                ),
                system=(
                    "Answer each question using the supplied article. "
                    "Return valid JSON only."
                ),
                temperature=0.1,
                max_output_tokens=700,
                json_schema={
                    "type": "object",
                    "properties": {
                        "answers": {
                            "type": "array",
                            "minItems": len(missing_questions),
                            "maxItems": len(missing_questions),
                            "items": {
                                "type": "string",
                                "maxLength": 700,
                            },
                        }
                    },
                    "required": ["answers"],
                    "additionalProperties": False,
                },
            )

            for question_item, answer in zip(
                missing_questions,
                retry_result.get("answers", []),
            ):
                cleaned_answer = str(answer).strip()
                if cleaned_answer:
                    answers_by_id[question_item["candidate_id"]
                                  ] = cleaned_answer

        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            print(f"Warning: Q&A retry failed: {exc}")

    # Last-resort fallback: preserve a usable daily candidate pool.
    for question_item in questions_with_ids:
        candidate_id = question_item["candidate_id"]
        if candidate_id not in answers_by_id:
            answers_by_id[candidate_id] = final_script

    return [
        {
            "question": item["question"],
            "answer": answers_by_id[item["candidate_id"]],
        }
        for item in questions_with_ids
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


def get_piper_synthesizer():
    """
    using personacasts local piper TTS, so sae voice across vocalized stored qa from offline gen 
    replacing pyttsx3!
    """

    repo_root = PROJECT_ROOT.parent # going to repo root
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # imports
    from personacast import config as personacast_config 
    from personacast.pipeline.tts import synthesize

    voice_path = Path(personacast_config.PIPER_VOICE_PATH)
    if not voice_path.is_absolute():
        personacast_config.PIPER_VOICE_PATH = str(repo_root / voice_path)

    return synthesize


def generate_candidate_audio(
    trunks: list[dict[str, Any]],
) -> None:
    synthesize = get_piper_synthesizer()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_jobs: list[tuple[dict[str, Any], Path, Path, str]] = []

    for trunk in trunks: # iterate through trunks
        trunk_audio_file = AUDIO_DIR / f"{trunk['trunk_id']}.wav" # path for trunk
        trunk_temporary_file = AUDIO_DIR / (
            f"{trunk['trunk_id']}.{os.getpid()}.tmp.wav"
        )
        audio_jobs.append(
            (trunk, trunk_temporary_file, trunk_audio_file, trunk["script"])
        )

        for question_answer in trunk["question_answers"]:
            qa_audio_file = AUDIO_DIR / f"{question_answer['qa_id']}.wav"
            qa_temporary_file = AUDIO_DIR / (
                f"{question_answer['qa_id']}.{os.getpid()}.tmp.wav"
            )
            audio_jobs.append(
                (
                    question_answer,
                    qa_temporary_file,
                    qa_audio_file,
                    question_answer["answer"],
                )
            )

    for _, temporary_file, _, text in audio_jobs:
        script = str(text or "").strip()
        if not script:
            continue
        try:
            synthesize(script, temporary_file)
        except Exception as exc:
            print(f"TTS failed for {temporary_file.name}: {exc}", flush=True)

    for candidate, temporary_file, audio_file, _ in audio_jobs:
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
        for candidate, _, _, _ in audio_jobs # extra element for answer text to pass through synthesise()
    )
    print(
        f"Generated {ready_count}/{len(audio_jobs)} audio files.",
        flush=True,
    )


def save_embedding_matrix(
    embeddings: np.ndarray,
    output_file: Path,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = output_file.with_name(output_file.name + ".tmp")

    with temporary_file.open("wb") as file:
        np.save(file, embeddings, allow_pickle=False)

    os.replace(temporary_file, output_file)


def generate_candidate_embeddings(
    candidate_pool: dict[str, Any],
) -> None:
    trunk_ids: list[str] = []
    trunk_texts: list[str] = []
    question_ids: list[str] = []
    question_texts: list[str] = []

    for topic_data in candidate_pool["topics"]:
        for trunk in topic_data["trunks"]:
            trunk_ids.append(str(trunk["trunk_id"]))
            trunk_texts.append(str(trunk["script"]).strip())

            for question_answer in trunk["question_answers"]:
                question_ids.append(str(question_answer["qa_id"]))
                question_texts.append(
                    str(question_answer["question"]).strip()
                )

    if not trunk_texts or not question_texts:
        raise ValueError("No trunks or questions were found for embedding.")

    print(
        f"Loading candidate embedding model on cpu: {EMBEDDING_MODEL}",
        flush=True,
    )
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    all_texts = trunk_texts + question_texts
    all_embeddings = model.encode(
        all_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    all_embeddings = np.asarray(all_embeddings, dtype=np.float32)

    trunk_count = len(trunk_texts)
    trunk_embeddings = all_embeddings[:trunk_count]
    question_embeddings = all_embeddings[trunk_count:]

    save_embedding_matrix(
        trunk_embeddings,
        TRUNK_EMBEDDINGS_FILE,
    )
    save_embedding_matrix(
        question_embeddings,
        QUESTION_EMBEDDINGS_FILE,
    )

    embedding_index = {
        "metadata": {
            "model_name": EMBEDDING_MODEL,
            "embedding_dimension": int(all_embeddings.shape[1]),
            "normalized_embeddings": True,
            "trunk_embedding_count": len(trunk_ids),
            "question_embedding_count": len(question_ids),
        },
        "trunk_ids": trunk_ids,
        "question_ids": question_ids,
    }

    with EMBEDDING_IDS_FILE.open("w", encoding="utf-8") as file:
        json.dump(embedding_index, file, ensure_ascii=False, indent=2)


def main() -> None:
    user_profile = load_user_profile()
    main_narrative_pool = load_main_narratives()

    candidate_pool: dict[str, Any] = {
        "configuration": {
            "story_count": len(main_narrative_pool["stories"]),
            "chunks_per_story": TRUNKS_PER_TOPIC,
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
                trunk["question_answers"] = generate_question_answers(
                    source_item=trunk["source_item"],
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

    generate_candidate_embeddings(candidate_pool)

    print(
        "Offline candidate generation completed: "
        f"{len(candidate_pool['topics'])} daily stories, "
        f"{total_trunks} connected chunks, "
        f"{total_questions} Q&A candidates.\n"
        f"Saved to: {OUTPUT_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
