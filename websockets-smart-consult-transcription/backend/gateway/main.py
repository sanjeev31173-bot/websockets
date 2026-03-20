from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .websocket_handler import handle_consultation

from ..config import get_settings
from ..queue.redis_client import push_audio, publish_end, subscribe_transcript


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
    index_file = frontend_dir / "index.html"
    return FileResponse(index_file)




@app.websocket("/ws/consultation/{session_id}")
async def consultation_ws(websocket: WebSocket, session_id: str):
    await handle_consultation(websocket, session_id)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.websocket("/ws/consultation/{session_id}")
async def consultation_ws(websocket: WebSocket, session_id: str) -> None:
    """
    Gateway server:
    - receives audio from browser
    - pushes to Redis queue
    - receives transcript from workers
    - sends transcript back to browser
    """

    await websocket.accept()

    producer = asyncio.create_task(_receive_audio(websocket, session_id))
    consumer = asyncio.create_task(_send_transcripts(websocket, session_id))

    try:
        await asyncio.gather(producer, consumer)
    except WebSocketDisconnect:
        publish_end(session_id)


async def _receive_audio(websocket: WebSocket, session_id: str) -> None:
    try:
        while True:

            msg = await websocket.receive()

            if "bytes" in msg and msg["bytes"] is not None:
                push_audio(session_id, msg["bytes"])

            elif "text" in msg and msg["text"] is not None:

                text = msg["text"]

                if text == "END":
                    publish_end(session_id)
                    break

                if text.startswith("{"):
                    try:
                        payload = json.loads(text)
                        # metadata message from client
                        if payload.get("type") == "start":
                            pass
                    except Exception:
                        pass

    except WebSocketDisconnect:
        publish_end(session_id)


async def _send_transcripts(websocket: WebSocket, session_id: str):

    pubsub = subscribe_transcript(session_id)

    transcript = []

    for message in pubsub.listen():

        if message["type"] != "message":
            continue

        data = message["data"]

        if isinstance(data, bytes):
            text = data.decode()

        else:
            text = str(data)

        if text == "__END__":
            break

        transcript.append(text)

        await websocket.send_text(text)

    # save transcript
    transcript_dir = Path(__file__).resolve().parents[1] / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    transcript_path = transcript_dir / f"{session_id}.txt"

    transcript_path.write_text(" ".join(transcript), encoding="utf-8")

    await websocket.send_json(
        {
            "type": "end",
            "transcript_file": str(transcript_path)
        }
    )

    