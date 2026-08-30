from __future__ import annotations

import audioop
import json
import threading
import time

import numpy as np

from .. import config
from .mic import MicSession

_active_session: MicSession | None = None
_server_thread: threading.Thread | None = None
_server_lock = threading.Lock()


def set_active_session(session: MicSession | None) -> None:
    global _active_session
    _active_session = session


def _get_active_session() -> MicSession | None:
    return _active_session


def _process_binary_frame(raw_bytes: bytes, source_rate: int, ratecv_state):
    session = _get_active_session()
    target_rate = session.sample_rate if session else config.STT_INPUT_SAMPLE_RATE
    resampled, new_state = audioop.ratecv(raw_bytes, 2, 1, source_rate, target_rate, ratecv_state)
    return np.frombuffer(resampled, dtype=np.int16), new_state


async def _handle_connection(websocket) -> None:
    ratecv_state = None
    source_rate = 48000
    async for message in websocket:
        if isinstance(message, str):
            try:
                data = json.loads(message)
                source_rate = int(data.get("sampleRate", source_rate))
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
            continue

        session = _get_active_session()
        if session is None:
            continue
        pcm, ratecv_state = _process_binary_frame(message, source_rate, ratecv_state)
        if hasattr(session, "frames_seen"):
            session.frames_seen += 1
        if hasattr(session, "last_frame_at"):
            session.last_frame_at = time.monotonic()
        session.push_frame(pcm)


_server_ready = threading.Event()
_server_error: str | None = None


def _run_server(port: int) -> None:
    global _server_error
    import asyncio

    import websockets

    async def _serve():
        async with websockets.serve(_handle_connection, "localhost", port):
            _server_ready.set()
            await asyncio.Future()

    try:
        asyncio.run(_serve())
    except OSError as err:
        _server_error = (
            f"mic WebSocket server could not bind port {port}: {err.strerror or err}. "
            f"Another PersonaCast tab or a stale process is probably holding it — close it, "
            f"or set PERSONACAST_MIC_WS_PORT to a free port."
        )
    except Exception as err:
        _server_error = f"mic WebSocket server crashed: {type(err).__name__}: {err}"
    finally:
        _server_ready.set()


def server_error() -> str | None:
    return _server_error


def ensure_server_running(port: int | None = None, *, timeout: float = 5.0) -> int:
    global _server_thread
    port = port or config.MIC_WS_PORT
    if _server_error is not None:
        raise RuntimeError(_server_error)

    with _server_lock:
        if _server_thread is None:
            _server_thread = threading.Thread(
                target=_run_server, args=(port,), daemon=True, name="pc-mic-ws",
            )
            _server_thread.start()
    _server_ready.wait(timeout=timeout)

    if _server_error is not None:
        raise RuntimeError(_server_error)
    if not _server_ready.is_set():
        raise RuntimeError(
            f"mic WebSocket server did not come up within {timeout:.0f}s on port {port}."
        )
    return port
