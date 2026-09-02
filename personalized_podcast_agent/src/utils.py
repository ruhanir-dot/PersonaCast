from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "output"

USER_IG_FILE = DATA_DIR / "user_ig_profile.txt"
USER_KEYWORDS_FILE = DATA_DIR / "user_keywords.txt"
USER_PROFILE_FILE = OUTPUT_DIR / "user_profile.json"


# ============================================================
# Setup
# ============================================================

def setup() -> None:
    """Load environment variables and create the output directory."""
    load_dotenv(PROJECT_ROOT / ".env")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_env(name: str, default: str | None = None) -> str:
    """Read an environment variable with an optional default."""
    setup()
    value = os.getenv(name, default)

    if value is None or value == "":
        raise RuntimeError(f"Missing environment variable: {name}")

    return value


def get_openai_client() -> Any:
    """Create an OpenAI client for user-profile generation when needed."""
    setup()

    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY.")

    return OpenAI()


def get_local_text_client() -> Any:
    """Create an OpenAI-compatible client for the local Ollama server."""
    setup()

    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")

    host = os.getenv(
        "LOCAL_LLM_HOST",
        "http://127.0.0.1:11434",
    ).rstrip("/")

    if not host.endswith("/v1"):
        host += "/v1"

    return OpenAI(
        base_url=host,
        api_key=os.getenv("LOCAL_LLM_API_KEY", "ollama"),
    )


# ============================================================
# File I/O
# ============================================================

def read_text(path: str | Path, default: str = "") -> str:
    path = Path(path)

    if not path.exists():
        return default

    return path.read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)

    if not path.exists():
        return default

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_user_ig_text() -> str:
    return read_text(USER_IG_FILE, default="")


def read_user_keywords() -> str:
    return read_text(USER_KEYWORDS_FILE, default="").strip()


def ensure_user_keywords_file() -> None:
    if not USER_KEYWORDS_FILE.exists():
        write_text(
            USER_KEYWORDS_FILE,
            "AI\nLLM web browsing\npersonalized podcast\n"
            "recommendation system\n",
        )


# Compatibility names used by the user-profile scripts.
def read_text_file(path: str | Path) -> str:
    return read_text(path, default="")


def write_text_file(path: str | Path, content: str) -> None:
    write_text(path, content)


def read_json_file(path: str | Path) -> Any:
    return read_json(path, default=None)


def write_json_file(path: str | Path, data: Any) -> None:
    write_json(path, data)


# ============================================================
# JSON parsing
# ============================================================

def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from raw or markdown-wrapped model output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        raise ValueError("Parsed JSON is not an object.")
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError("No JSON object found in model output.")

    data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Parsed JSON is not an object.")

    return data


def safe_extract_json_object(text: str) -> dict[str, Any]:
    try:
        return extract_json_object(text)
    except Exception as exc:
        return {
            "raw_llm_output": text,
            "json_parse_error": str(exc),
        }


# ============================================================
# LLM helpers
# ============================================================

def llm_text(
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.3,
    max_output_tokens: int | None = None,
) -> str:
    client = get_local_text_client()
    model = os.getenv("LOCAL_LLM_MODEL", "qwen2.5:1.5b")
    full_prompt = prompt if not system else (
        f"System instruction:\n{system}\n\n"
        f"User task:\n{prompt}"
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "input": full_prompt,
        "temperature": temperature,
    }

    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens

    response = client.responses.create(**kwargs)
    return response.output_text.strip()


def _ollama_json_response(
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_output_tokens: int | None,
    json_schema: dict[str, Any] | None = None,
) -> str:
    setup()
    host = os.getenv(
        "LOCAL_LLM_HOST",
        "http://127.0.0.1:11434",
    ).rstrip("/")

    options: dict[str, Any] = {
        "temperature": max(0.0, min(float(temperature), 0.1)),
    }

    if max_output_tokens is not None:
        options["num_predict"] = int(max_output_tokens)

    response = requests.post(
        f"{host}/api/chat",
        json={
            "model": os.getenv("LOCAL_LLM_MODEL", "qwen2.5:1.5b"),
            "messages": messages,
            "format": json_schema or "json",
            "stream": False,
            "options": options,
        },
        timeout=300,
    )
    response.raise_for_status()

    print(
        "[ollama] output format:",
        "schema" if json_schema is not None else "json",
    )

    return str(response.json()["message"]["content"]).strip()


def llm_json(
    prompt: str,
    *,
    system: str = "",
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    json_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, str]] = []

    if system.strip():
        messages.append({
            "role": "system",
            "content": system.strip(),
        })

    messages.append({
        "role": "user",
        "content": prompt,
    })

    try:
        output = _ollama_json_response(
            messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            json_schema=json_schema,
        )
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    try:
        return extract_json_object(output)
    except Exception as exc:
        raise RuntimeError(
            "Local Qwen returned invalid JSON. "
            f"Parse error: {exc}. "
            f"Raw output: {output[:1500]}"
        ) from exc
