from __future__ import annotations

import json
import os
import wave
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.utils import PROJECT_ROOT, llm_json


DATA_DIR = PROJECT_ROOT / "data" / "output"
CANDIDATE_FILE = DATA_DIR / "candidate_trunks.json"
QUESTION_EMBEDDINGS_FILE = DATA_DIR / "candidate_question_embeddings.npy"
EMBEDDING_IDS_FILE = DATA_DIR / "candidate_embedding_ids.json"
AUDIO_DIR = DATA_DIR / "audio"
HTML_FILE = PROJECT_ROOT / "web" / "trunks.html"

QUESTION_SIMILARITY_THRESHOLD = 0.80

app = FastAPI()

_embedding_model: SentenceTransformer | None = None


class CustomQuestionRequest(BaseModel):
    trunk_id: str
    question: str


def load_candidate_pool() -> dict[str, Any]:
    if not CANDIDATE_FILE.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Candidate pool does not exist: {CANDIDATE_FILE}",
        )

    try:
        with CANDIDATE_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="candidate_trunks.json could not be loaded.",
        ) from exc


def find_trunk(
    candidate_pool: dict[str, Any],
    trunk_id: str,
) -> dict[str, Any]:
    for topic in candidate_pool.get("topics", []):
        for trunk in topic.get("trunks", []):
            if str(trunk.get("trunk_id")) == trunk_id:
                return trunk

    raise HTTPException(status_code=404, detail="Trunk not found.")


def find_question_answer(
    candidate_pool: dict[str, Any],
    qa_id: str,
) -> dict[str, Any]:
    for topic in candidate_pool.get("topics", []):
        for trunk in topic.get("trunks", []):
            for question_answer in trunk.get("question_answers", []):
                if str(question_answer.get("qa_id")) == qa_id:
                    return question_answer

    raise HTTPException(status_code=404, detail="Question answer not found.")


def build_audio_url(audio_file: Any) -> str | None:
    if not audio_file:
        return None

    filename = Path(str(audio_file)).name
    return f"/api/trunks/audio/{quote(filename)}"


def serialize_question_answer(
    question_answer: dict[str, Any],
) -> dict[str, Any]:
    return {
        "qa_id": question_answer["qa_id"],
        "question": question_answer["question"],
        "answer": question_answer["answer"],
        "audio_url": build_audio_url(question_answer.get("audio_file")),
        "audio_status": question_answer.get("audio_status"),
    }


def serialize_trunk(trunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "trunk_id": trunk["trunk_id"],
        "title": trunk.get("title", ""),
        "script": trunk["script"],
        "audio_url": build_audio_url(trunk.get("audio_file")),
        "audio_status": trunk.get("audio_status"),
        "question_answers": [
            serialize_question_answer(question_answer)
            for question_answer in trunk.get("question_answers", [])
        ],
    }


def get_embedding_model(model_name: str) -> SentenceTransformer:
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(model_name, device="cpu")

    return _embedding_model


def search_similar_question(
    question: str,
    candidate_pool: dict[str, Any],
) -> tuple[dict[str, Any], float] | None:
    if not QUESTION_EMBEDDINGS_FILE.exists() or not EMBEDDING_IDS_FILE.exists():
        return None

    with EMBEDDING_IDS_FILE.open("r", encoding="utf-8") as file:
        embedding_index = json.load(file)

    question_ids = embedding_index["question_ids"]
    question_embeddings = np.load(
        QUESTION_EMBEDDINGS_FILE,
        allow_pickle=False,
    )

    if len(question_ids) != len(question_embeddings):
        raise HTTPException(
            status_code=500,
            detail="Question embedding IDs do not match the embedding matrix.",
        )

    model_name = embedding_index["metadata"]["model_name"]
    model = get_embedding_model(model_name)
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    similarities = question_embeddings @ query_embedding
    best_index = int(np.argmax(similarities))
    best_similarity = float(similarities[best_index])

    matched = find_question_answer(
        candidate_pool,
        str(question_ids[best_index]),
    )
    return matched, best_similarity


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

    raise HTTPException(
        status_code=500,
        detail="No English TTS voice is installed on this computer.",
    )


def generate_answer_audio(answer: str) -> str:
    try:
        import pyttsx3
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="pyttsx3 is not installed.",
        ) from exc

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_id = f"custom_{uuid4().hex}"
    audio_file = AUDIO_DIR / f"{audio_id}.wav"
    temporary_file = AUDIO_DIR / f"{audio_id}.{os.getpid()}.tmp.wav"

    engine = pyttsx3.init()
    set_english_tts_voice(engine)
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)
    engine.save_to_file(answer, str(temporary_file))
    engine.runAndWait()
    engine.stop()

    if not audio_file_has_frames(temporary_file):
        if temporary_file.exists():
            temporary_file.unlink()
        raise HTTPException(
            status_code=500, detail="Answer audio generation failed.")

    os.replace(temporary_file, audio_file)
    return build_audio_url(audio_file) or ""


def generate_custom_answer(
    trunk: dict[str, Any],
    question: str,
) -> str:
    source_item = trunk.get("source_item", {})
    article_text = str(source_item.get("article_text", "")).strip()

    result = llm_json(
        prompt=(
            "Current podcast script:\n"
            f"{trunk['script']}\n\n"
            "Article content:\n"
            f"{article_text}\n\n"
            "User question:\n"
            f"{question}\n\n"
            "Answer the user question directly as one natural English podcast "
            "paragraph. Use only information supported by the current script "
            "and article. Do not repeat the question, add a heading, greeting, "
            "or closing statement."
        ),
        system="Return the podcast answer as valid JSON only.",
        temperature=0.1,
        max_output_tokens=300,
        json_schema={
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1200,
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    )
    return str(result["answer"]).strip()


@app.get("/", include_in_schema=False)
@app.get("/trunks", include_in_schema=False)
def trunks_page() -> FileResponse:
    if not HTML_FILE.exists():
        raise HTTPException(
            status_code=404, detail="trunks.html was not found.")
    return FileResponse(HTML_FILE)


@app.get("/api/trunks/topics")
def get_topics() -> dict[str, Any]:
    candidate_pool = load_candidate_pool()
    topics = candidate_pool.get("topics", [])
    configured_count = candidate_pool.get("configuration", {}).get(
        "topic_count",
        len(topics),
    )

    return {
        "topic_count": configured_count,
        "topics": [
            {
                "topic_id": topic["topic_id"],
                "topic": topic["topic"],
            }
            for topic in topics
        ],
    }


@app.get("/api/trunks/topics/{topic_id}")
def get_topic(topic_id: str) -> dict[str, Any]:
    candidate_pool = load_candidate_pool()

    for topic in candidate_pool.get("topics", []):
        if str(topic.get("topic_id")) == topic_id:
            return {
                "topic_id": topic["topic_id"],
                "topic": topic["topic"],
                "trunks": [
                    serialize_trunk(trunk)
                    for trunk in topic.get("trunks", [])
                ],
            }

    raise HTTPException(status_code=404, detail="Topic not found.")


@app.get("/api/trunks/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    audio_file = AUDIO_DIR / Path(filename).name

    if not audio_file.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")

    return FileResponse(audio_file, media_type="audio/wav")


@app.post("/api/trunks/custom-question")
def answer_custom_question(
    request: CustomQuestionRequest,
) -> dict[str, Any]:
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=400, detail="Question cannot be empty.")

    candidate_pool = load_candidate_pool()
    trunk = find_trunk(candidate_pool, request.trunk_id)
    search_result = search_similar_question(question, candidate_pool)

    similarity: float | None = None
    if search_result is not None:
        question_answer, similarity = search_result

        if similarity >= QUESTION_SIMILARITY_THRESHOLD:
            return {
                "mode": "matched",
                "answer": question_answer["answer"],
                "audio_url": build_audio_url(
                    question_answer.get("audio_file")
                ),
                "matched_qa_id": question_answer["qa_id"],
                "similarity": similarity,
            }

    answer = generate_custom_answer(trunk, question)
    audio_url = generate_answer_audio(answer)
    return {
        "mode": "generated",
        "answer": answer,
        "audio_url": audio_url,
        "matched_qa_id": None,
        "similarity": similarity,
    }
