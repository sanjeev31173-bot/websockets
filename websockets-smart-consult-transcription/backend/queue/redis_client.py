import redis
import json

r = redis.Redis(host="localhost", port=6379)


def push_audio(session_id: str, chunk: bytes):

    payload = {
        "session_id": session_id,
        "audio": chunk.hex()
    }

    r.rpush("audio_queue", json.dumps(payload))


def publish_end(session_id: str):

    payload = {
        "session_id": session_id,
        "audio": "END"
    }

    r.rpush("audio_queue", json.dumps(payload))


def subscribe_transcript(session_id: str):

    pubsub = r.pubsub()
    pubsub.subscribe(f"transcript:{session_id}")
    return pubsub