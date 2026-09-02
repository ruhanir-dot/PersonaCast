import json
import os
import re
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from src.offline.generate_main_narrative import retrieve_recommended_contents
from src.offline.predict_user_actions import predict_user_questions
from src.user_profile.select_feed_seeds import (
    save_feed_seeds,
    select_feed_seeds,
)
from src.utils import PROJECT_ROOT, llm_json


DATA_DIR = PROJECT_ROOT / "data" / "output"
USER_PROFILE_FILE = DATA_DIR / "user_profile.json"
OUTPUT_FILE = DATA_DIR / "candidate_trunks.json"
TRUNK_EMBEDDINGS_FILE = DATA_DIR / "candidate_trunk_embeddings.npy"
QUESTION_EMBEDDINGS_FILE = DATA_DIR / "candidate_question_embeddings.npy"
EMBEDDING_IDS_FILE = DATA_DIR / "candidate_embedding_ids.json"
AUDIO_DIR = DATA_DIR / "audio"

TOPIC_COUNT = 10
TRUNKS_PER_TOPIC = 5
QUESTIONS_PER_TRUNK = 3
SCRIPT_MIN_WORDS = 20
SCRIPT_MAX_WORDS = 40
SCRIPT_MAX_ATTEMPTS = 3
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


def predict_topics(user_profile: dict[str, Any]) -> list[str]:
    topics = [
        str(topic).strip()
        for topic in user_profile.get("podcast_focus_keywords", [])
        if str(topic).strip()
    ]

    if len(topics) < TOPIC_COUNT:
        raise ValueError(
            f"Expected at least {TOPIC_COUNT} podcast focus keywords, "
            f"but received {len(topics)}."
        )

    return topics[:TOPIC_COUNT]


def clean_search_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[@#]+", "", text)
    text = re.sub(r"[|/\\]+", " ", text)
    text = re.sub(r"[^\w\s'’-]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def build_search_queries(topic: str, seed: dict[str, Any]) -> list[str]:
    seed_text = clean_search_text(seed.get("text"))
    creator = clean_search_text(seed.get("creator"))
    clean_topic = clean_search_text(topic)

    queries = [
        f"{clean_topic} {seed_text} latest news",
        f"{clean_topic} {creator} latest news",
    ]

    unique_queries = []
    seen_queries: set[str] = set()

    for query in queries:
        query = " ".join(query.split())
        normalized_query = query.casefold()

        if query and normalized_query not in seen_queries:
            unique_queries.append(query)
            seen_queries.add(normalized_query)

    return unique_queries


def source_key(source_item: dict[str, Any]) -> str:
    for field in ("url", "link", "source_url", "title"):
        value = str(source_item.get(field) or "").strip().casefold()
        if value:
            return value
    return json.dumps(source_item, ensure_ascii=False, sort_keys=True)


def retrieve_source_for_seed(
    topic: str,
    seed: dict[str, Any],
    used_source_keys: set[str],
) -> tuple[str, dict[str, Any]] | None:
    attempted_queries = []

    for search_query in build_search_queries(topic, seed):
        attempted_queries.append(search_query)
        source_items = retrieve_recommended_contents(query=search_query)

        for source_item in source_items:
            key = source_key(source_item)
            if key in used_source_keys:
                continue
            if not source_item.get("article_text"):
                continue

            used_source_keys.add(key)
            return search_query, source_item

    print(
        "Skipped seed because no article was found. "
        f"Attempted queries: {attempted_queries}",
        flush=True,
    )
    return None


def validate_script(script: str) -> bool:
    word_count = len(script.split())
    return (
        SCRIPT_MIN_WORDS <= word_count <= SCRIPT_MAX_WORDS
        and script.endswith((".", "!", "?"))
    )


def generate_short_script(
    topic: str,
    personalized_seed: dict[str, Any],
    source_item: dict[str, Any],
) -> dict[str, Any]:
    previous_feedback = ""

    for _ in range(SCRIPT_MAX_ATTEMPTS):
        result = llm_json(
            prompt=(
                f"Podcast topic:\n{topic}\n\n"
                "Personalized feed seed:\n"
                f"{json.dumps(personalized_seed, ensure_ascii=False)}\n\n"
                "Article title:\n"
                f"{source_item.get('title', '')}\n\n"
                "Article content:\n"
                f"{source_item['article_text']}\n\n"
                "Write exactly 25 English words as one standalone podcast "
                "script using one or two complete sentences. Base every fact "
                "only on the article. Do not include greetings, conclusions, "
                "headings, hashtags, or source descriptions."
                f"{previous_feedback}"
            ),
            system=(
                "Return one concise source-grounded podcast script as valid "
                "JSON only."
            ),
            temperature=0.0,
            max_output_tokens=500,
            json_schema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "maxLength": 180,
                    },
                    "focus": {
                        "type": "string",
                        "maxLength": 300,
                    },
                    "script": {
                        "type": "string",
                        "minLength": 40,
                        "maxLength": 350,
                    },
                },
                "required": ["title", "focus", "script"],
                "additionalProperties": False,
            },
        )

        script = str(result["script"]).strip()
        return {
            "title": str(result["title"]).strip(),
            "focus": str(result["focus"]).strip(),
            "script": script,
        }
        '''if validate_script(script):
            return {
                "title": str(result["title"]).strip(),
                "focus": str(result["focus"]).strip(),
                "script": script,
            }

        previous_feedback = (
            f"\nThe previous script had {len(script.split())} words. "
            "Rewrite it using exactly 25 words and end with complete "
            "punctuation."
        )

    raise ValueError(
        f"Could not generate a valid 20-to-30-word script for topic: {topic}"
    )'''


def generate_question_answers(
    topic: str,
    personalized_seed: dict[str, Any],
    source_item: dict[str, Any],
    final_script: str,
    user_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    predicted_questions = predict_user_questions(
        topic=topic,
        trunk_script=final_script,
        selected_feed_seed=personalized_seed,
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
        candidate_id = str(item["candidate_id"])
        if candidate_id in answers_by_id:
            raise ValueError(
                f"Duplicate answer returned for {candidate_id}."
            )
        answers_by_id[candidate_id] = str(item["answer"]).strip()

    if set(answers_by_id) != set(candidate_ids):
        raise ValueError(
            "The answer generator did not answer every predicted question."
        )

    return [
        {
            "question": item["question"],
            "answer": answers_by_id[item["candidate_id"]],
        }
        for item in questions_with_ids
    ]


def generate_trunk(
    topic: str,
    personalized_seed: dict[str, Any],
    source_item: dict[str, Any],
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    trunk = generate_short_script(
        topic=topic,
        personalized_seed=personalized_seed,
        source_item=source_item,
    )
    trunk["question_answers"] = generate_question_answers(
        topic=topic,
        personalized_seed=personalized_seed,
        source_item=source_item,
        final_script=trunk["script"],
        user_profile=user_profile,
    )
    return trunk


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

        for question_answer in trunk["question_answers"]:
            qa_audio_file = AUDIO_DIR / f"{question_answer['qa_id']}.wav"
            qa_temporary_file = AUDIO_DIR / (
                f"{question_answer['qa_id']}.{os.getpid()}.tmp.wav"
            )
            engine.save_to_file(
                str(question_answer["answer"]).strip(),
                str(qa_temporary_file),
            )
            audio_jobs.append(
                (question_answer, qa_temporary_file, qa_audio_file)
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
    topics = predict_topics(user_profile)

    feed_seed_pool = select_feed_seeds(topics)
    save_feed_seeds(feed_seed_pool)

    candidate_pool: dict[str, Any] = {
        "configuration": {
            "topic_count": TOPIC_COUNT,
            "trunks_per_topic": TRUNKS_PER_TOPIC,
            "questions_per_trunk": QUESTIONS_PER_TRUNK,
        },
        "topics": [],
    }
    save_candidate_pool(candidate_pool)

    used_source_keys: set[str] = set()

    for topic_number, topic_data in enumerate(
        feed_seed_pool["topics"],
        start=1,
    ):
        topic = topic_data["topic"]
        selected_seeds = topic_data["selected_seeds"]
        trunks = []

        print(
            f"[{topic_number}/{TOPIC_COUNT}] Generating topic: {topic}",
            flush=True,
        )

        for seed in selected_seeds:
            if len(trunks) >= TRUNKS_PER_TOPIC:
                break

            retrieved = retrieve_source_for_seed(
                topic=topic,
                seed=seed,
                used_source_keys=used_source_keys,
            )

            if retrieved is None:
                continue

            search_query, source_item = retrieved

            trunk = generate_trunk(
                topic=topic,
                personalized_seed=seed,
                source_item=source_item,
                user_profile=user_profile,
            )
            trunk["personalized_seed"] = seed
            trunk["search_query"] = search_query
            trunk["source_item"] = source_item
            trunks.append(trunk)

            print(
                f"  [{len(trunks)}/{TRUNKS_PER_TOPIC}] Finished trunk",
                flush=True,
            )

        if len(trunks) < TRUNKS_PER_TOPIC:
            raise ValueError(
                f"Only generated {len(trunks)} trunks for topic '{topic}'. "
                f"Expected {TRUNKS_PER_TOPIC}."
            )

        trunks = add_candidate_ids(topic_number, trunks)
        generate_candidate_audio(trunks)
        candidate_pool["topics"].append(
            {
                "topic_id": f"topic_{topic_number:02d}",
                "topic": topic,
                "selected_seeds": selected_seeds,
                "trunks": trunks,
            }
        )
        save_candidate_pool(candidate_pool)

        print(
            f"[{topic_number}/{TOPIC_COUNT}] Finished: "
            f"{len(trunks)} trunks, "
            f"{len(trunks) * QUESTIONS_PER_TRUNK} Q&A candidates",
            flush=True,
        )

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
        f"{len(candidate_pool['topics'])} topics, "
        f"{total_trunks} trunks, "
        f"{total_questions} Q&A candidates.\n"
        f"Saved to: {OUTPUT_FILE}",
        flush=True,
    )


if __name__ == "__main__":
    main()
