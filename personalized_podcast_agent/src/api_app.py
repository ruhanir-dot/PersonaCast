from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from src.tree_api import router as tree_router

import logging
logger = logging.getLogger("uvicorn.error")


PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI()
app.include_router(tree_router)


@app.get("/tree-player")
def tree_player() -> FileResponse:
    html_file = PROJECT_ROOT / "web" / "tree_player.html"
    return FileResponse(html_file)


@app.on_event("startup")
async def show_tree_player_url() -> None:
    logger.info(
        "Tree player: http://127.0.0.1:8000/tree-player"
    )
