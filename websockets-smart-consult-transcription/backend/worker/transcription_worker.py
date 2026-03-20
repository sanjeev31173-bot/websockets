import redis 
import whisper
import numpy as np
import json

r = redis.Redis()

model = whisper.load_model("base")

print("Worker started")

while True:

    _, data = r.blpop("audio_queue")

    payload = json.loads(data)

    session_id = payload["session_id"]

    if payload["audio"] == "END":
        continue

    audio_bytes = bytes.fromhex(payload["audio"])

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768

    result = model.transcribe(audio_np, fp16=False)

    text = result["text"]

    r.publish(f"transcript:{session_id}", text)