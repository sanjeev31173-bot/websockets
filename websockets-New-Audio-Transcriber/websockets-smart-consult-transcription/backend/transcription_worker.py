from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from typing import Optional

import numpy as np
import whisper

from .audio_session import AudioSession
from .config import get_settings

settings = get_settings()

_model_lock = asyncio.Lock()
_model: Optional["whisper.Whisper"] = None

# How many seconds of audio we consider "tentative" / partial at the tail of the window.
# Segments ending within the last PARTIAL_TAIL_SEC are sent as partial and can be rewritten.
PARTIAL_TAIL_SEC = 2.5


def _load_model() -> "whisper.Whisper":
    global _model
    if _model is None:
        _model = whisper.load_model(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
        )
    return _model


async def _load_model_async() -> "whisper.Whisper":
    async with _model_lock:
        if _model is not None:
            return _model
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load_model)


# ---------------------------------------------------------------------------
# Silence / VAD guard
# ---------------------------------------------------------------------------

def _is_silent(audio_f32: np.ndarray, threshold: float = 0.01) -> bool:
    """Return True if the audio chunk is below the RMS silence threshold."""
    if audio_f32.size == 0:
        return True
    rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
    return rms < threshold


# ---------------------------------------------------------------------------
# Main transcription loop
# ---------------------------------------------------------------------------

async def run_transcription_loop(session: AudioSession) -> None:
    model = await _load_model_async()

    target_sr = settings.AUDIO_SAMPLE_RATE
    max_samples = int(settings.AUDIO_BUFFER_SECONDS * target_sr)
    min_window_samples = int((settings.MIN_TRANSCRIBE_WINDOW_MS / 1000) * target_sr)
    transcribe_every_sec = settings.TRANSCRIBE_EVERY_MS / 1000

    chunks: deque[np.ndarray] = deque()
    window_samples = 0
    abs_samples_received = 0
    last_finalized_end_sec = 0.0
    last_transcribe_at = 0.0

    # Track the last partial text we sent so we only send diffs
    last_partial_text: str = ""

    async for chunk in session.audio_iter():
        if not chunk:
            continue

        arr = np.frombuffer(chunk, dtype=np.int16)
        if arr.size == 0:
            continue

        in_sr = int(getattr(session, "client_sample_rate_hz", target_sr) or target_sr)
        if in_sr != target_sr:
            arr = _resample_pcm16(arr, in_sr=in_sr, out_sr=target_sr)

        abs_samples_received += arr.size
        chunks.append(arr)
        window_samples += arr.size

        # Keep rolling window bounded
        while window_samples > max_samples and chunks:
            dropped = chunks.popleft()
            window_samples -= dropped.size

        now = time.monotonic()

        if window_samples < min_window_samples:
            continue
        if (now - last_transcribe_at) < transcribe_every_sec:
            continue

        last_transcribe_at = now

        # Build audio float32
        audio_i16 = np.concatenate(list(chunks))
        audio_f32 = audio_i16.astype(np.float32) / 32768.0

        # Skip entirely silent windows — avoids Whisper hallucinations on silence
        if _is_silent(audio_f32):
            continue

        try:
            result = await _transcribe(model, audio_f32)
        except Exception as exc:
            await session.push_partial_transcript(
                json.dumps({"type": "warning", "text": str(exc)})
            )
            continue

        segments = result.get("segments", [])
        if not segments:
            continue

        window_start_sec = max(0, abs_samples_received - window_samples) / target_sr
        window_duration_sec = window_samples / target_sr

        final_texts: list[str] = []
        new_partial_text: str = ""
        consumed_samples = 0

        for seg in segments:
            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))
            abs_end = window_start_sec + end
            txt = str(seg.get("text", "")).strip()

            if not txt:
                continue

            # Already finalized — skip
            if abs_end <= last_finalized_end_sec + 0.05:
                continue

            # Is this segment in the "safe" finalized zone?
            # A segment is final if it ends more than PARTIAL_TAIL_SEC before the window end.
            secs_from_window_end = window_duration_sec - end
            if secs_from_window_end > PARTIAL_TAIL_SEC:
                final_texts.append(txt)
                last_finalized_end_sec = max(last_finalized_end_sec, abs_end)
                consumed_samples = int(end * target_sr)
            else:
                # Partial — accumulate for the "rewriting" zone
                new_partial_text = (new_partial_text + " " + txt).strip()

        # Emit final segments
        for txt in final_texts:
            msg = json.dumps({"type": "final", "text": txt})
            await session.push_partial_transcript(msg)

        # Emit partial only if it changed
        if new_partial_text != last_partial_text:
            msg = json.dumps({"type": "partial", "text": new_partial_text})
            await session.push_partial_transcript(msg)
            last_partial_text = new_partial_text

        # Trim consumed audio from the front of the window
        if consumed_samples > 0:
            removed = 0
            while chunks and removed < consumed_samples:
                c = chunks.popleft()
                if removed + c.size <= consumed_samples:
                    removed += c.size
                    window_samples -= c.size
                else:
                    remain = consumed_samples - removed
                    chunks.appendleft(c[remain:])
                    window_samples -= remain
                    break

    # Session ended — flush any remaining partial as final
    if last_partial_text:
        msg = json.dumps({"type": "final", "text": last_partial_text})
        await session.push_partial_transcript(msg)

    await session.mark_transcription_finished()


# ---------------------------------------------------------------------------
# Whisper inference (off the event loop thread)
# ---------------------------------------------------------------------------

async def _transcribe(model: "whisper.Whisper", audio_f32: np.ndarray) -> dict:
    def _run():
        return model.transcribe(
            audio_f32,
            language=settings.WHISPER_LANGUAGE,
            fp16=False,
            condition_on_previous_text=False,
            verbose=None,
            # Whisper's own VAD-style — no_speech_prob threshold
            # segments with high no_speech_prob are suppressed automatically
        )

    return await asyncio.to_thread(_run)


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample_pcm16(x: np.ndarray, *, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return x
    if x.size < 2:
        return x
    ratio = out_sr / in_sr
    out_len = int(round(x.size * ratio))
    xp = np.arange(x.size, dtype=np.float32)
    x_new = np.linspace(0, x.size - 1, out_len, dtype=np.float32)
    y = np.interp(x_new, xp, x.astype(np.float32))
    return y.astype(np.int16)