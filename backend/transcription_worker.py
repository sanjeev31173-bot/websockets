from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Optional

import numpy as np
import whisper

from .audio_session import AudioSession
from .config import get_settings
import uuid

settings = get_settings()

_model_lock = asyncio.Lock()
_model: Optional["whisper.Whisper"] = None


def _load_model() -> "whisper.Whisper":
    global _model
    if _model is None:
        _model = whisper.load_model(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE
        )
    return _model


async def _load_model_async() -> "whisper.Whisper":
    async with _model_lock:
        model = _model
        if model is not None:
            return model
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load_model)


async def run_transcription_loop(session: AudioSession) -> None:

    model = await _load_model_async()

    target_sr = settings.AUDIO_SAMPLE_RATE
    max_samples = int(settings.AUDIO_BUFFER_SECONDS * target_sr)
    min_window_samples = int((settings.MIN_TRANSCRIBE_WINDOW_MS / 1000) * target_sr)
    transcribe_every_sec = settings.TRANSCRIBE_EVERY_MS / 1000

    chunks: "deque[np.ndarray]" = deque()

    window_samples = 0
    abs_samples_received = 0

    last_emitted_end_sec = 0.0
    last_emitted_text: Optional[str] = None
    last_transcribe_at = 0.0

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

        while window_samples > max_samples and chunks:
            dropped = chunks.popleft()
            window_samples -= dropped.size

        now = time.monotonic()

        if window_samples < min_window_samples:
            continue

        if (now - last_transcribe_at) < transcribe_every_sec:
            continue

        last_transcribe_at = now

        try:

            emitted_texts, last_emitted_end_sec, consumed_samples = await _transcribe_window_incremental(
                model=model,
                chunks=chunks,
                window_samples=window_samples,
                abs_samples_received=abs_samples_received,
                last_emitted_end_sec=last_emitted_end_sec,
            )

        except Exception as exc:
            await session.push_partial_transcript(f"[transcription_warning] {exc}")
            continue

        # REMOVE PROCESSED AUDIO (prevents retranscribing old speech)
        if consumed_samples > 0:

            removed = 0

            while chunks and removed < consumed_samples:
                c = chunks.popleft()

                if removed + c.size <= consumed_samples:
                    removed += c.size
                    window_samples -= c.size

                else:
                    remain = consumed_samples - removed
                    new_chunk = c[remain:]
                    chunks.appendleft(new_chunk)
                    window_samples -= remain
                    removed = consumed_samples

        for t in emitted_texts:

            if t and t != last_emitted_text:
                await session.push_partial_transcript(t)
                last_emitted_text = t

    await session.mark_transcription_finished()


async def _transcribe_window_incremental(
    *,
    model: "whisper.Whisper",
    chunks: "deque[np.ndarray]",
    window_samples: int,
    abs_samples_received: int,
    last_emitted_end_sec: float,
):

    sr = settings.AUDIO_SAMPLE_RATE

    window_start_abs_samples = max(0, abs_samples_received - window_samples)
    window_start_sec = window_start_abs_samples / sr

    audio_i16 = np.concatenate(list(chunks))
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    def _do_transcribe():

        emitted: list[str] = []
        new_last_end = last_emitted_end_sec
        consumed_samples = 0

        result = model.transcribe(
            audio_f32,
            language=settings.WHISPER_LANGUAGE,
            fp16=False,

            # IMPORTANT FIX
            condition_on_previous_text=False,

            verbose=None,
        )

        for seg in result.get("segments", []):

            start = float(seg.get("start", 0))
            end = float(seg.get("end", 0))

            abs_end = window_start_sec + end

            if abs_end <= new_last_end + 0.05:
                continue

            txt = str(seg.get("text", "")).strip()

            if txt:
                emitted.append(txt)

            new_last_end = max(new_last_end, abs_end)

            consumed_samples = int(end * sr)

        return emitted, new_last_end, consumed_samples

    emitted_texts, new_last_end, consumed_samples = await asyncio.to_thread(_do_transcribe)

    return emitted_texts, new_last_end, consumed_samples


def _resample_pcm16(x: np.ndarray, *, in_sr: int, out_sr: int) -> np.ndarray:

    if in_sr == out_sr:
        return x

    if x.size < 2:
        return x

    ratio = out_sr / in_sr

    out_len = int(round(x.size * ratio))

    xp = np.arange(x.size, dtype=np.float32)
    fp = x.astype(np.float32)

    x_new = np.linspace(0, x.size - 1, out_len, dtype=np.float32)

    y = np.interp(x_new, xp, fp)

    return y.astype(np.int16)


import asyncio


if __name__ == "__main__":

    session_id = str(uuid.uuid4())

    session = AudioSession(session_id)

    asyncio.run(run_transcription_loop(session))