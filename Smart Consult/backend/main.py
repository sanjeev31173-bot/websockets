from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio_session import AudioSession
from .config import get_settings
from .llm_client import analyze_consultation
from .transcription_worker import run_transcription_loop


settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: Dict[str, AudioSession] = {}
sessions_lock = asyncio.Lock()


frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve the demo frontend."""
    index_file = frontend_dir / "index.html"
    return FileResponse(index_file)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def get_or_create_session(session_id: str) -> AudioSession:
    async with sessions_lock:
        session = sessions.get(session_id)
        if session is None:
            session = AudioSession(session_id=session_id)
            sessions[session_id] = session
            # Start transcription worker for this session.
            asyncio.create_task(run_transcription_loop(session))
        return session


@app.websocket("/ws/consultation/{session_id}")
async def consultation_ws(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket contract:
    - Client sends:
        * Binary frames with raw PCM16 mono 16kHz audio for the PATIENT side of the call.
        * A final text frame "END" to close the stream and trigger final analysis.
    - Server sends:
        * Text frames with incremental transcript updates as they are produced.
        * A final JSON text frame with LLM analysis once transcription is finished.
    """
    await websocket.accept()
    session = await get_or_create_session(session_id)

    producer = asyncio.create_task(_receive_audio(websocket, session))
    consumer = asyncio.create_task(_send_transcript_updates(websocket, session))

    try:
        await asyncio.gather(producer, consumer)
    except WebSocketDisconnect:
        await session.mark_closed()
    finally:
        # Cleanup (best-effort).
        async with sessions_lock:
            sessions.pop(session_id, None)


async def _receive_audio(websocket: WebSocket, session: AudioSession) -> None:
    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                await session.add_audio_chunk(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                if msg["text"] == "END":
                    await session.mark_closed()
                    break
    except WebSocketDisconnect:
        await session.mark_closed()


async def _send_transcript_updates(websocket: WebSocket, session: AudioSession) -> None:
    # Stream partial transcripts to client.
    async for update in session.transcript_updates():
        await websocket.send_text(update)

    # Once transcription is done, run LLM analysis on full transcript.
    full_text = " ".join(session.full_transcript)
    try:
        analysis = analyze_consultation(full_text)
    except Exception as exc:  # noqa: BLE001
        await websocket.send_json({"type": "analysis_error", "error": str(exc)})
        return

    await websocket.send_json({"type": "analysis", "data": analysis})

