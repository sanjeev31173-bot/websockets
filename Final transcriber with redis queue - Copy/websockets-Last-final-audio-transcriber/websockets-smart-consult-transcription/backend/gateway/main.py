import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path

from ..config import get_settings
from ..queue.redis_client import push_audio, push_new_session, subscribe_transcript, publish_end

settings = get_settings()
app = FastAPI(title=settings.APP_NAME + " (Gateway)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(frontend_dir / "index.html")

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})

@app.websocket("/ws/consultation/{session_id}")
async def consultation_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await push_new_session(session_id)
    
    producer = asyncio.create_task(_receive_audio(websocket, session_id))
    consumer = asyncio.create_task(_send_transcripts(websocket, session_id))
    
    try:
        await asyncio.gather(producer, consumer)
    except WebSocketDisconnect:
        await publish_end(session_id)

async def _receive_audio(websocket: WebSocket, session_id: str):
    try:
        while True:
            msg = await websocket.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                await push_audio(session_id, msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                text = msg["text"]
                if text == "END":
                    await publish_end(session_id)
                    break
    except WebSocketDisconnect:
        await publish_end(session_id)

async def _send_transcripts(websocket: WebSocket, session_id: str):
    pubsub = await subscribe_transcript(session_id)
    transcript_dir = Path(__file__).resolve().parents[2] / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    full_text = []

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            
            data = json.loads(message["data"].decode('utf-8'))
            msg_type = data.get("type")
            
            if msg_type == "end":
                transcript_path = transcript_dir / f"{session_id}.txt"
                transcript_path.write_text(" ".join(full_text).strip() + "\n", encoding="utf-8")
                data["transcript_file"] = str(transcript_path)
                await websocket.send_json(data)
                break
            
            if msg_type == "final":
                full_text.append(data.get("text", ""))
                
            await websocket.send_json(data)
    finally:
        await pubsub.unsubscribe()