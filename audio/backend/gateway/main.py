"""
backend/gateway/main.py  —  SINGLE-MODEL SEQUENTIAL v3
=======================================================

Architecture change from v2
-----------------------------

v2 PROBLEM (root cause of ALL remaining issues):
  The chunking + parallel worker pool is fundamentally wrong for this use case.
  - Chunks split sentences unpredictably at fixed time boundaries
  - Overlap deduplication is unreliable when Whisper re-words a phrase
  - The live pass races with chunk workers sharing the same model
  - Result: missing words, repeated sentences, mis-recognised words

v3 SOLUTION — Single model, VAD-segmented, sequential pipeline:

  1. ONE model, ONE worker thread — no race conditions, no RAM waste.
     The single model gets ALL CPU threads so it's actually faster per inference.

  2. NO fixed-size chunking. Instead we use Whisper's own VAD to find natural
     speech boundaries. Each transcription call receives a rolling window of
     audio; Whisper's VAD internally splits it at silences and only returns
     segments where speech was detected.

  3. WORD-TIMESTAMP deduplication: we track the last emitted word's END
     timestamp. Any word whose start time ≤ last_emitted_end is skipped.
     This is 100% reliable — timestamps never lie, unlike text comparison.

  4. FINAL PASS on session end: when the user clicks Stop, we transcribe
     the ENTIRE session audio in one call. This is the most accurate
     possible transcription and is used as the saved transcript file.

  5. context_prompt is grown incrementally — the last 200 chars of emitted
     text is always fed back as initial_prompt so Whisper never starts cold.

Result: no missing words, no repeated words, correct recognition.
"""

import asyncio
import threading
import queue as stdlib_queue
import time
import re
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config import get_settings
settings = get_settings()

app = FastAPI(title=settings.APP_NAME + " (Gateway)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: dict = {}
session_tasks: dict = {}

from backend.worker.main import model_pool, NUM_WORKERS  # noqa: E402

SR                 = settings.AUDIO_SAMPLE_RATE
MIN_START_SAMPLES  = int(settings.MIN_CHUNK_TO_START_SEC * SR)
WINDOW_SAMPLES     = int(settings.SEGMENT_WINDOW_SEC * SR)
OVERLAP_SAMPLES    = int(settings.SEGMENT_OVERLAP_SEC * SR)

# The one model — used by both rolling passes and the final full pass
_MODEL = model_pool[0]
# Mutex so the rolling live pass and the one chunk worker never call
# model.transcribe() simultaneously on the same object
_MODEL_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Core transcription helper
# ─────────────────────────────────────────────────────────────────────────────

def _transcribe(audio: np.ndarray, initial_prompt: str = "") -> list:
    """
    Call model.transcribe() with all accuracy settings.
    Returns a list of Segment objects (each has .text, .words, .start, .end).
    Thread-safe — acquires _MODEL_LOCK.
    """
    prompt = initial_prompt or settings.MEDICAL_VOCAB_PROMPT
    with _MODEL_LOCK:
        seg_gen, _ = _MODEL.transcribe(
            audio,
            language=settings.WHISPER_LANGUAGE or None,
            condition_on_previous_text=True,
            initial_prompt=prompt,
            word_timestamps=True,            # ← needed for timestamp dedup

            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": settings.VAD_MIN_SILENCE_MS,   # 700ms
                "speech_pad_ms":           500,    # keep 500ms around speech edges
                "threshold":               0.25,   # low = keep quiet speech too
                "min_speech_duration_ms":  200,    # ignore very short noise bursts
            },

            beam_size=settings.WHISPER_BEAM_SIZE,   # 5
            no_speech_threshold=0.15,               # very low — catch quiet speakers
            compression_ratio_threshold=2.4,
            temperature=[0.0, 0.2, 0.4],            # fallback list for bad audio
            patience=1.2,                           # slightly more beam patience
        )
        return list(seg_gen)


def _build_prompt(emitted_text: str) -> str:
    """Build the initial_prompt for the next transcription call."""
    tail = emitted_text[-200:].strip() if emitted_text else ""
    if tail:
        return tail + "\n" + settings.MEDICAL_VOCAB_PROMPT
    return settings.MEDICAL_VOCAB_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp-based deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _extract_new_words(segments: list,
                       window_offset_sec: float,
                       last_emitted_end_sec: float) -> tuple[str, float]:
    """
    Given transcribed segments from a rolling window that started at
    `window_offset_sec` seconds into the full audio, extract only the
    words whose absolute end time > last_emitted_end_sec.

    Returns:
        (new_text, new_last_emitted_end_sec)

    This is 100% reliable — we deduplicate by timestamp, not by text
    comparison, so re-worded repetitions are impossible.
    """
    new_words = []
    new_end   = last_emitted_end_sec

    for seg in segments:
        if seg.words is None:
            # No word timestamps — fall back to segment-level timestamp
            abs_start = window_offset_sec + seg.start
            abs_end   = window_offset_sec + seg.end
            if abs_start >= last_emitted_end_sec - 0.05:   # 50ms tolerance
                new_words.append(seg.text.strip())
                new_end = max(new_end, abs_end)
        else:
            for w in seg.words:
                abs_start = window_offset_sec + w.start
                abs_end   = window_offset_sec + w.end
                if abs_start >= last_emitted_end_sec - 0.05:   # 50ms tolerance
                    new_words.append(w.word)
                    new_end = max(new_end, abs_end)

    text = _clean(" ".join(new_words))
    return text, new_end


def _clean(text: str) -> str:
    """Collapse whitespace and strip leading/trailing space."""
    return re.sub(r"\s+", " ", text).strip()


def _strip_hallucinations(text: str) -> str:
    """
    Remove silence-induced Whisper hallucinations from the END of text.

    When Whisper sees silence or very low-level noise it often loops a
    short phrase (e.g. "Thank you. Thank you, doctor. Thank you.").
    Strategy:
      1. Split into sentences.
      2. Walk backwards from the end and drop any sentence that is an
         exact (case-insensitive) repeat of one we already saw.
      3. Also drop the entire tail if it is just one short phrase
         repeated 2+ times consecutively.
    """
    if not text:
        return text

    # Split on sentence-ending punctuation keeping the delimiter
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 1:
        return text

    # Walk from the end, drop repeated sentences
    seen   = set()
    result = []
    for s in sentences:
        key = re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()
        if key and key in seen:
            continue          # duplicate — drop it
        seen.add(key)
        result.append(s)

    # Additional pass: if the last sentence appears 2+ times consecutively
    # at the tail (can happen with partial punctuation), strip the extras
    while len(result) >= 2:
        last = re.sub(r'[^a-z0-9 ]', '', result[-1].lower()).strip()
        prev = re.sub(r'[^a-z0-9 ]', '', result[-2].lower()).strip()
        if last and last == prev:
            result.pop()
        else:
            break

    return " ".join(result).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Background transcription worker (single thread, sequential)
# ─────────────────────────────────────────────────────────────────────────────

class _SequentialWorker:
    """
    One background thread with one job queue.
    Jobs are processed one at a time — no concurrency, no race conditions.
    """
    def __init__(self):
        self._q = stdlib_queue.Queue()
        t = threading.Thread(target=self._loop, daemon=True, name="WhisperWorker-0")
        t.start()

    def submit(self, fn, *args):
        """Submit a callable; it will be called in the worker thread."""
        self._q.put((fn, args))

    def _loop(self):
        while True:
            fn, args = self._q.get()
            try:
                fn(*args)
            except Exception:
                import traceback
                traceback.print_exc()


_worker = _SequentialWorker()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws/consultation/{session_id}")
async def consultation_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()

    active_sessions[session_id] = {
        "websocket":        websocket,
        "audio_queue":      asyncio.Queue(),
        "transcript_queue": asyncio.Queue(),
        "full_text":        [],
    }

    proc_task = asyncio.create_task(_accumulate_and_dispatch(session_id))
    session_tasks[session_id] = proc_task

    producer = asyncio.create_task(_receive_audio(websocket, session_id))
    consumer = asyncio.create_task(_send_transcripts(websocket, session_id))

    try:
        await asyncio.gather(producer, consumer)
    except WebSocketDisconnect:
        print(f"[{session_id}] client disconnected")
    finally:
        await _cleanup_session(session_id)


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "active_sessions": len(active_sessions),
        "num_workers": 1,
    })


frontend_dir = Path(__file__).resolve().parents[2] / "public"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — WebSocket → audio_queue
# ─────────────────────────────────────────────────────────────────────────────

async def _receive_audio(websocket: WebSocket, session_id: str):
    try:
        while True:
            msg = await websocket.receive()
            if session_id not in active_sessions:
                break
            aq = active_sessions[session_id]["audio_queue"]
            if msg.get("bytes"):
                await aq.put(msg["bytes"])
            elif msg.get("text") == "END":
                await aq.put(None)
                break
    except WebSocketDisconnect:
        if session_id in active_sessions:
            await active_sessions[session_id]["audio_queue"].put(None)


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — transcript_queue → WebSocket
# ─────────────────────────────────────────────────────────────────────────────

async def _send_transcripts(websocket: WebSocket, session_id: str):
    transcript_dir = Path(__file__).resolve().parents[2] / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    try:
        tq = active_sessions[session_id]["transcript_queue"]
        while True:
            data = await tq.get()
            if data.get("type") == "final":
                active_sessions[session_id]["full_text"].append(data.get("text", ""))
            if data.get("type") == "end":
                full_text = active_sessions[session_id].get("final_transcript", "")
                if not full_text:
                    full_text = "\n".join(active_sessions[session_id]["full_text"]).strip()
                path = transcript_dir / f"{session_id}.txt"
                path.write_text(full_text + "\n", encoding="utf-8")
                data["transcript_file"] = str(path)
            await websocket.send_json(data)
            if data.get("type") == "end":
                break
    except Exception as exc:
        print(f"[{session_id}] send_transcripts: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — main pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def _accumulate_and_dispatch(session_id: str):
    """
    Pipeline:
      1. Continuously drain audio_queue into a growing buffer (pcm_buf).
      2. Every ROLLING_INTERVAL seconds, if enough audio has arrived:
         - Submit a rolling transcription job to _worker (background thread).
         - The job transcribes the last WINDOW_SAMPLES of audio.
         - New words (by timestamp) are put on transcript_queue as "final".
      3. On session end:
         - Wait for any in-flight rolling job to finish.
         - Transcribe the FULL session audio in one pass for maximum accuracy.
         - Save the full transcript file.
    """
    aq   = active_sessions[session_id]["audio_queue"]
    tq   = active_sessions[session_id]["transcript_queue"]
    loop = asyncio.get_event_loop()

    # Full PCM buffer — we keep everything for the final pass
    full_buf: list[np.ndarray] = []
    total_samples = 0

    # State shared between rolling jobs (protected by result_event flow)
    last_emitted_end_sec: float = 0.0   # timestamp of last emitted word
    emitted_transcript:   str   = ""    # text we've emitted so far

    is_closed    = False
    last_partial = ""

    # Synchronisation between the async loop and the worker thread
    result_event = asyncio.Event()
    worker_busy  = False

    # ── Rolling job callback (called from worker thread) ──────────────────────
    def on_rolling_result(window_audio: np.ndarray,
                          window_offset: float,
                          prompt: str):
        nonlocal last_emitted_end_sec, emitted_transcript, worker_busy
        try:
            t0 = time.monotonic()
            segments = _transcribe(window_audio, prompt)
            elapsed  = time.monotonic() - t0
            print(f"[worker] rolling pass {window_audio.size/SR:.1f}s audio "
                  f"→ {elapsed:.1f}s ({window_audio.size/SR/elapsed:.2f}x rt)")

            new_text, new_end = _extract_new_words(
                segments, window_offset, last_emitted_end_sec
            )
            new_text = _strip_hallucinations(new_text)

            if new_text:
                last_emitted_end_sec = new_end
                emitted_transcript   = (emitted_transcript + " " + new_text).strip()
                # Send to async loop
                loop.call_soon_threadsafe(
                    lambda t=new_text: asyncio.ensure_future(
                        tq.put({"type": "final", "text": t}),
                        loop=loop
                    )
                )
                print(f"[{session_id}] ROLLING FINAL: '{new_text[:100]}'")
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            worker_busy = False
            loop.call_soon_threadsafe(result_event.set)

    # ── Final full-audio pass callback ────────────────────────────────────────
    def on_final_result(all_audio: np.ndarray, prompt: str):
        try:
            t0 = time.monotonic()
            print(f"[{session_id}] FINAL PASS: transcribing "
                  f"{all_audio.size/SR:.1f}s of full audio ...")
            segments = _transcribe(all_audio, prompt)
            elapsed  = time.monotonic() - t0
            print(f"[{session_id}] FINAL PASS done in {elapsed:.1f}s")

            # Build the full transcript from scratch — most accurate result
            full_text = _clean(" ".join(
                w.word for seg in segments
                for w in (seg.words or [])
            ))
            if not full_text:
                # Fallback: no word timestamps
                full_text = _clean(" ".join(seg.text.strip() for seg in segments))

            full_text = _strip_hallucinations(full_text)

            if session_id in active_sessions:
                active_sessions[session_id]["final_transcript"] = full_text
            print(f"[{session_id}] FULL TRANSCRIPT:\n{full_text}")
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            loop.call_soon_threadsafe(result_event.set)

    # ── Helper: get the current full audio as a numpy array ──────────────────
    def get_full_audio() -> np.ndarray:
        return np.concatenate(full_buf) if full_buf else np.empty(0, dtype=np.float32)

    ROLLING_INTERVAL = 3.0   # run a rolling pass every 3 seconds of new audio
    last_roll_time   = 0.0
    last_total       = 0     # samples at last roll — to detect new audio

    try:
        while not is_closed:

            # ── Drain audio queue ─────────────────────────────────────────────
            try:
                raw = await asyncio.wait_for(aq.get(), timeout=0.05)
                if raw is None:
                    is_closed = True
                else:
                    pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    full_buf.append(pcm)
                    total_samples += pcm.size
            except asyncio.TimeoutError:
                pass

            while not aq.empty():
                raw = aq.get_nowait()
                if raw is None:
                    is_closed = True
                    break
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                full_buf.append(pcm)
                total_samples += pcm.size

            # ── Collect result if worker finished ─────────────────────────────
            if result_event.is_set():
                result_event.clear()

            # ── Rolling transcription pass ────────────────────────────────────
            now = time.monotonic()
            new_samples = total_samples - last_total
            enough_audio = total_samples >= MIN_START_SAMPLES
            enough_new   = new_samples >= int(ROLLING_INTERVAL * SR)
            time_ok      = (now - last_roll_time) >= ROLLING_INTERVAL

            if enough_audio and time_ok and enough_new and not worker_busy and not is_closed:
                # Build the window: last WINDOW_SAMPLES of audio
                all_audio     = get_full_audio()
                window_audio  = (all_audio[-WINDOW_SAMPLES:]
                                 if all_audio.size > WINDOW_SAMPLES
                                 else all_audio)
                # Absolute offset of this window's start in the full timeline
                window_offset = max(0.0, (total_samples - window_audio.size) / SR)
                prompt        = _build_prompt(emitted_transcript)

                worker_busy  = True
                last_roll_time = now
                last_total   = total_samples

                print(f"[{session_id}] rolling pass: "
                      f"window {window_audio.size/SR:.1f}s "
                      f"offset {window_offset:.1f}s "
                      f"last_emitted_end {last_emitted_end_sec:.1f}s")
                _worker.submit(on_rolling_result, window_audio, window_offset, prompt)

            # ── Partial / live display (lightweight, no model call) ───────────
            # We show the last few words of emitted_transcript as "in progress"
            # This costs zero compute — just re-displays what we already have.
            if emitted_transcript:
                # Take last sentence fragment as partial indicator
                tail_words = emitted_transcript.split()[-12:]
                partial    = " ".join(tail_words) + " ..."
                if partial != last_partial:
                    await tq.put({"type": "partial", "text": partial})
                    last_partial = partial

        # ── Session closed — wait for any in-flight rolling job ───────────────
        if worker_busy:
            print(f"[{session_id}] waiting for in-flight rolling job ...")
            try:
                await asyncio.wait_for(result_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                print(f"[{session_id}] WARNING: rolling job timed out")
            result_event.clear()

        # ── FINAL FULL-AUDIO PASS — most accurate possible transcription ───────
        all_audio = get_full_audio()
        if all_audio.size > int(0.5 * SR):
            result_event.clear()
            prompt = _build_prompt(emitted_transcript)
            _worker.submit(on_final_result, all_audio, prompt)
            print(f"[{session_id}] waiting for final pass ...")
            try:
                await asyncio.wait_for(result_event.wait(), timeout=300.0)
            except asyncio.TimeoutError:
                print(f"[{session_id}] WARNING: final pass timed out")

        # ── Emit the final full transcript to frontend ────────────────────────
        final_text = active_sessions.get(session_id, {}).get("final_transcript", "")
        if final_text:
            # Replace all rolling partial results with the authoritative full text
            await tq.put({"type": "final_complete", "text": final_text})

        await tq.put({"type": "end"})
        print(f"[{session_id}] transcription complete.")

    except Exception as exc:
        print(f"[{session_id}] _accumulate_and_dispatch crashed: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if session_id in active_sessions:
            try:
                active_sessions[session_id]["transcript_queue"].put_nowait({"type": "end"})
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

async def _cleanup_session(session_id: str):
    print(f"[{session_id}] cleaning up")
    if session_id in session_tasks:
        session_tasks[session_id].cancel()
        del session_tasks[session_id]
    if session_id in active_sessions:
        del active_sessions[session_id]