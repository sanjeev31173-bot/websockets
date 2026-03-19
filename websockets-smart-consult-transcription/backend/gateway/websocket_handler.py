from fastapi import WebSocket, WebSocketDisconnect
from ..queue.redis_client import push_audio, publish_end, subscribe_transcript
import asyncio


async def handle_consultation(websocket: WebSocket, session_id: str):

    await websocket.accept()

    pubsub = subscribe_transcript(session_id)

    producer = asyncio.create_task(receive_audio(websocket, session_id))
    consumer = asyncio.create_task(send_transcripts(websocket, pubsub))

    try:
        await asyncio.gather(producer, consumer)
    except WebSocketDisconnect:
        publish_end(session_id)


async def receive_audio(websocket: WebSocket, session_id: str):

    while True:

        msg = await websocket.receive()

        if "bytes" in msg and msg["bytes"] is not None:
            print("Audio chunk received")
            push_audio(session_id, msg["bytes"])

        elif "text" in msg and msg["text"] is not None:

            text = msg["text"]

            if text == "END":
                publish_end(session_id)
                break


async def send_transcripts(websocket: WebSocket, pubsub):

    for message in pubsub.listen():

        if message["type"] != "message":
            continue

        data = message["data"]

        if isinstance(data, bytes):
            text = data.decode()
        else:
            text = str(data)

        await websocket.send_text(text)