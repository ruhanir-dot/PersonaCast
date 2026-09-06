from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from src.utils import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]


STATUS_FILE = PROJECT_ROOT / "data" / "output" / "daily_generation_status.json"
PIPELINE_MODULES = [
    "src.user_profile.fetch_youtube_daily",
    "src.user_profile.build_personal_feed",
    "src.user_profile.build_feed_embeddings",
    "src.offline.generate_main_narrative",
    "src.offline.generate_trunks",
]


def write_status(payload: dict[str, object]) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = STATUS_FILE.with_suffix(STATUS_FILE.suffix + ".tmp")
    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary_file, STATUS_FILE)


def run_module(module_name: str) -> None:
    print(f"\n=== Running {module_name} ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    started_at = datetime.now(UTC)
    completed_modules: list[str] = []
    write_status(
        {
            "status": "running",
            "started_at": started_at.isoformat(),
            "completed_modules": completed_modules,
        }
    )

    try:
        for module_name in PIPELINE_MODULES:
            run_module(module_name)
            completed_modules.append(module_name)
            write_status(
                {
                    "status": "running",
                    "started_at": started_at.isoformat(),
                    "completed_modules": completed_modules,
                }
            )
    except subprocess.CalledProcessError as exc:
        failed_at = datetime.now(UTC)
        write_status(
            {
                "status": "failed",
                "started_at": started_at.isoformat(),
                "failed_at": failed_at.isoformat(),
                "completed_modules": completed_modules,
                "failed_module": exc.cmd[-1] if exc.cmd else None,
                "return_code": exc.returncode,
            }
        )
        raise

    finished_at = datetime.now(UTC)
    write_status(
        {
            "status": "completed",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "completed_modules": completed_modules,
        }
    )
    print("\nDaily YouTube podcast generation completed.", flush=True)


if __name__ == "__main__":
    main()
