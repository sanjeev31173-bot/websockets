from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio_session import AudioSession
from .config import get_settings
from .transcription_worker import run_transcription_loop

settings = get_settings()
app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.websocket("/ws/consultation/{session_id}")
async def consultation_ws(websocket: WebSocket, session_id: str) -> None:
    """
    WebSocket contract:

    Client → Server:
      * Binary frames: raw PCM16 mono 16 kHz audio
      * Text "END": close the stream
      * Text JSON {"type":"start","sample_rate_hz":N}: optional metadata

    Server → Client (JSON text frames):
      * {"type": "partial", "text": "..."} — current tentative words, will rewrite
      * {"type": "final",   "text": "..."} — confirmed words, append to transcript
      * {"type": "warning", "text": "..."} — non-fatal transcription issue
      * {"type": "end", "transcript_file": "..."} — session complete
    """
    await websocket.accept()
    session = AudioSession(session_id=session_id)
    asyncio.create_task(run_transcription_loop(session))

    producer = asyncio.create_task(_receive_audio(websocket, session))
    consumer = asyncio.create_task(_send_transcript_updates(websocket, session))

    try:
        await asyncio.gather(producer, consumer)
    except WebSocketDisconnect:
        await session.mark_closed()


async def _receive_audio(websocket: WebSocket, session: AudioSession) -> None:
    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                await session.add_audio_chunk(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                text = msg["text"]
                if text == "END":
                    await session.mark_closed()
                    break
                if text.startswith("{"):
                    try:
                        payload = json.loads(text)
                        if payload.get("type") == "start":
                            sr = int(payload.get("sample_rate_hz") or payload.get("sampleRate") or 0)
                            if 8000 <= sr <= 192000:
                                session.client_sample_rate_hz = sr
                    except Exception:
                        pass
    except WebSocketDisconnect:
        await session.mark_closed()


async def _send_transcript_updates(websocket: WebSocket, session: AudioSession) -> None:
    async for msg in session.transcript_updates():
        await websocket.send_text(msg)

    # Persist only the finalized transcript
    transcript_dir = Path(__file__).resolve().parents[1] / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = transcript_dir / f"{session.session_id}.txt"
    text = " ".join(session.full_transcript).strip()
    transcript_path.write_text(text + ("\n" if text else ""), encoding="utf-8")

    await websocket.send_json({"type": "end", "transcript_file": str(transcript_path)})