import redis.asyncio as aioredis
import json

redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=False)

async def push_audio(session_id: str, chunk: bytes):
    await redis_client.rpush(f"audio:{session_id}", chunk)

async def publish_transcript(session_id: str, data: dict):
    await redis_client.publish(f"transcript:{session_id}", json.dumps(data).encode('utf-8'))

async def subscribe_transcript(session_id: str):
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"transcript:{session_id}")
    return pubsub

async def publish_end(session_id: str):
    await redis_client.rpush(f"audio:{session_id}", b"END")

async def push_new_session(session_id: str):
    await redis_client.rpush("new_sessions", session_id)

async def pop_new_session(timeout: int = 0):
    result = await redis_client.blpop("new_sessions", timeout=timeout)
    if result:
        return result[1].decode('utf-8')
    return None

async def pop_audio_batch(session_id: str):
    pipe = redis_client.pipeline()
    pipe.lrange(f"audio:{session_id}", 0, -1)
    pipe.delete(f"audio:{session_id}")
    results = await pipe.execute()
    return results[0]