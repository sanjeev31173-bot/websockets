import asyncio
import numpy as np
import json
import time
import sys
import threading
from faster_whisper import WhisperModel

from ..queue.redis_client import pop_new_session, pop_audio_batch, publish_transcript, redis_client
from ..config import get_settings
import os

settings = get_settings()

print(f"Loading faster-whisper model '{settings.WHISPER_MODEL_SIZE}' on {settings.WHISPER_DEVICE}...")
_threads = max(4, os.cpu_count() or 4)
model = WhisperModel(
    settings.WHISPER_MODEL_SIZE,
    device=settings.WHISPER_DEVICE,
    compute_type=settings.WHISPER_COMPUTE_TYPE,
    cpu_threads=_threads
)
print("Warming up CTranslate2 compiler and VAD model with a dummy pass to ensure 0ms first-click latency...")
list(model.transcribe(
    np.zeros(16000, dtype=np.float32),
    vad_filter=settings.WHISPER_VAD_FILTER,
    beam_size=settings.WHISPER_BEAM_SIZE 
))
print("Worker GPU Model Loaded! Ready to accept jobs from Redis.")

async def process_session(session_id: str):
    print(f"[{session_id}] Claimed active session. Worker allocated.")
    audio_buffer_f32 = np.array([], dtype=np.float32)
    last_transcribe_time = 0.0
    last_partial_text = ""
    is_closed = False
    
    target_sr = 16000
    transcribe_every_sec = settings.TRANSCRIBE_EVERY_MS / 1000.0
    buffer_max_sec = settings.MAX_BUFFER_SEC
    silence_tail_sec = settings.SILENCE_TAIL_SEC
    
    while not is_closed:
        await asyncio.sleep(0.05)
        
        # Pull incoming chunks quickly
        chunks = await pop_audio_batch(session_id)
        if chunks:
            raw_floats = []
            for chunk in chunks:
                if chunk == b"END":
                    is_closed = True
                else:
                    audio_np = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                    raw_floats.append(audio_np)
            
            if raw_floats:
                audio_buffer_f32 = np.concatenate([audio_buffer_f32] + raw_floats)
                
        duration_sec = audio_buffer_f32.size / target_sr
        min_window_samples = int((settings.MIN_TRANSCRIBE_WINDOW_MS / 1000.0) * target_sr)
        
        if audio_buffer_f32.size < min_window_samples and not is_closed:
            continue
            
        now = time.monotonic()
        if now - last_transcribe_time < transcribe_every_sec and not is_closed:
            continue
            
        if duration_sec == 0:
            continue
            
        # Transcribe
        segments_gen, _ = await asyncio.to_thread(
            model.transcribe,
            audio_buffer_f32,
            language=settings.WHISPER_LANGUAGE if settings.WHISPER_LANGUAGE else None,
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=settings.MEDICAL_VOCAB_PROMPT,
            vad_filter=settings.WHISPER_VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=500) if settings.WHISPER_VAD_FILTER else None,
            beam_size=settings.WHISPER_BEAM_SIZE,
        )
        
        # Move the throttle timer to AFTER the inference executes. 
        # This gives the CPU a guaranteed ~200ms breathing room, entirely preventing 100% saturation and lag spirals!
        last_transcribe_time = time.monotonic()
        
        segments = list(segments_gen)
        text = " ".join([seg.text.strip() for seg in segments]).strip()
        
        if not text or not segments:
            if duration_sec > 2.0:
                # Keep the last 1.5s of audio to avoid truncating words just starting to form
                drop_samples = max(0, audio_buffer_f32.size - int(1.5 * target_sr))
                if drop_samples > 0:
                    audio_buffer_f32 = audio_buffer_f32[drop_samples:]
            
            if last_partial_text:
                await publish_transcript(session_id, {"type": "final", "text": last_partial_text})
                last_partial_text = ""
            if is_closed:
                await publish_transcript(session_id, {"type": "end"})
            continue

        all_words = []
        for seg in segments:
            if hasattr(seg, "words") and seg.words:
                all_words.extend(seg.words)
                
        last_word_end = all_words[-1].end if all_words else segments[-1].end
        is_tail_silent = (duration_sec - float(last_word_end)) >= silence_tail_sec
        
        if is_tail_silent or is_closed:
            await publish_transcript(session_id, {"type": "final", "text": text})
            last_partial_text = ""
            audio_buffer_f32 = np.array([], dtype=np.float32)
        else:
            if duration_sec > buffer_max_sec:
                commit_end_time = 0.0
                commit_words = []
                partial_words = []
                
                if all_words:
                    for w in all_words:
                        if duration_sec - float(w.end) > 1.0:
                            commit_words.append(w.word)
                            commit_end_time = float(w.end)
                        else:
                            partial_words.append(w.word)
                            
                    commit_text = "".join(commit_words).strip()
                    if commit_text:
                        await publish_transcript(session_id, {"type": "final", "text": commit_text})
                        
                    last_partial_text = "".join(partial_words).strip()
                    if last_partial_text:
                        await publish_transcript(session_id, {"type": "partial", "text": last_partial_text})
                else:
                    final_segs = []
                    partial_segs = []
                    for seg in segments:
                        if duration_sec - float(seg.end) > 1.5:
                            final_segs.append(seg.text)
                            commit_end_time = float(seg.end)
                        else:
                            partial_segs.append(seg.text)

                    commit_text = "".join(final_segs).strip()
                    if commit_text:
                        await publish_transcript(session_id, {"type": "final", "text": commit_text})
                    last_partial_text = "".join(partial_segs).strip()
                    if last_partial_text:
                        await publish_transcript(session_id, {"type": "partial", "text": last_partial_text})
                        
                if commit_end_time > 0:
                    drop_samples = int(commit_end_time * target_sr)
                    audio_buffer_f32 = audio_buffer_f32[drop_samples:]
            else:
                if text != last_partial_text:
                    await publish_transcript(session_id, {"type": "partial", "text": text})
                    last_partial_text = text

        if is_closed:
            await publish_transcript(session_id, {"type": "end"})
            break
            
    print(f"[{session_id}] Session processed completely.")

async def worker_loop():
    print("Worker pool active. Listening for new patient sessions on Redis...")
    while True:
        try:
            session_id = await pop_new_session(timeout=0)
            if session_id:
                # Fire and forget isolation. This allows concurrent patient streams!
                asyncio.create_task(process_session(session_id))
        except Exception as e:
            print(f"Redis polling error (is Redis running?): {e}")
            await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(worker_loop())
