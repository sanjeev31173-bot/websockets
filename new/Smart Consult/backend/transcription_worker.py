from __future__ import annotations

import asyncio
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


def _load_model() -> "whisper.Whisper":
    global _model
    if _model is None:
        _model = whisper.load_model(settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE)
    return _model


async def _load_model_async() -> "whisper.Whisper":
    # Load model in a thread so we don't block the event loop on startup.
    async with _model_lock:
        model = _model
        if model is not None:
            return model
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _load_model)


async def run_transcription_loop(session: AudioSession) -> None:
    """
    Consume audio chunks from the session and push partial transcripts.

    Strategy for scalability & long sessions:
    - Keep a fixed-size rolling audio buffer (seconds) so RAM cannot grow unbounded.
    - Transcribe periodically and emit only NEW segments using timestamps (no prefix-diff drift).
    - Run Whisper inference off the event loop so we don't block other sockets.
    """
    model = await _load_model_async()

    target_sr = settings.AUDIO_SAMPLE_RATE
    max_samples = max(1, int(settings.AUDIO_BUFFER_SECONDS * target_sr))
    min_window_samples = max(1, int((settings.MIN_TRANSCRIBE_WINDOW_MS / 1000) * target_sr))
    transcribe_every_sec = max(0.05, settings.TRANSCRIBE_EVERY_MS / 1000)

    # Deque of int16 arrays, representing a rolling window of audio.
    chunks: "deque[np.ndarray]" = deque()
    window_samples = 0
    abs_samples_received = 0

    # To avoid duplicate text, track last emitted segment end time in absolute seconds.
    last_emitted_end_sec = 0.0
    last_transcribe_at = 0.0

    async for chunk in session.audio_iter():
        if not chunk:
            continue
        # bytes -> int16 PCM; chunk size is small, so frombuffer is fine.
        arr = np.frombuffer(chunk, dtype=np.int16)
        if arr.size == 0:
            continue

        # Resample to 16kHz if the browser is actually sending a different rate.
        # (Browsers often capture at 48kHz even if you request 16kHz.)
        in_sr = int(getattr(session, "client_sample_rate_hz", target_sr) or target_sr)
        if in_sr != target_sr:
            arr = _resample_pcm16(arr, in_sr=in_sr, out_sr=target_sr)

        abs_samples_received += int(arr.size)
        chunks.append(arr)
        window_samples += int(arr.size)

        # Trim rolling window to fixed size.
        while window_samples > max_samples and chunks:
            dropped = chunks.popleft()
            window_samples -= int(dropped.size)

        now = time.monotonic()
        if window_samples < min_window_samples:
            continue
        if (now - last_transcribe_at) < transcribe_every_sec:
            continue
        last_transcribe_at = now

        try:
            emitted_texts, last_emitted_end_sec = await _transcribe_window_incremental(
                model=model,
                chunks=chunks,
                window_samples=window_samples,
                abs_samples_received=abs_samples_received,
                last_emitted_end_sec=last_emitted_end_sec,
            )
        except Exception as exc:  # noqa: BLE001
            # If Whisper fails transiently, keep the session alive and try again on next tick.
            await session.push_partial_transcript(f"[transcription_warning] {exc}")
            continue

        for t in emitted_texts:
            await session.push_partial_transcript(t)

    # Flush at end: final pass over whatever remains in window.
    if window_samples >= min_window_samples and chunks:
        try:
            emitted_texts, _last_end = await _transcribe_window_incremental(
                model=model,
                chunks=chunks,
                window_samples=window_samples,
                abs_samples_received=abs_samples_received,
                last_emitted_end_sec=last_emitted_end_sec,
            )
            for t in emitted_texts:
                await session.push_partial_transcript(t)
        except Exception as exc:  # noqa: BLE001
            await session.push_partial_transcript(f"[transcription_warning] {exc}")

    await session.mark_transcription_finished()


async def _transcribe_window_incremental(
    *,
    model: "whisper.Whisper",
    chunks: "deque[np.ndarray]",
    window_samples: int,
    abs_samples_received: int,
    last_emitted_end_sec: float,
) -> tuple[list[str], float]:
    """
    Transcribe the current rolling window and return only newly completed segments.

    We rely on segment timestamps to avoid duplicates when the rolling window shifts.
    """
    sr = settings.AUDIO_SAMPLE_RATE
    # window start in absolute seconds:
    window_start_abs_samples = max(0, abs_samples_received - window_samples)
    window_start_sec = window_start_abs_samples / sr

    # Build float32 array for Whisper.
    audio_i16 = np.concatenate(list(chunks)) if len(chunks) > 1 else chunks[0]
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    def _do_transcribe() -> tuple[list[str], float]:
        emitted: list[str] = []
        new_last_end = last_emitted_end_sec

        # NOTE: `openai-whisper` doesn't offer true streaming; we "simulate" it by
        # repeatedly transcribing a fixed-size rolling window and emitting only new segments.
        # This keeps memory bounded and avoids unbounded latency on long calls.
        result = model.transcribe(
            audio_f32,
            language=settings.WHISPER_LANGUAGE,
            fp16=False,
            condition_on_previous_text=True,
            verbose=None,
        )
        for seg in result.get("segments", []):
            abs_end = window_start_sec + float(seg.get("end", 0.0))
            if abs_end <= (new_last_end + 0.05):
                continue
            txt = str(seg.get("text", "")).strip()
            if txt:
                emitted.append(txt)
            new_last_end = max(new_last_end, abs_end)
        return emitted, new_last_end

    emitted_texts, new_last_end = await asyncio.to_thread(_do_transcribe)
    return emitted_texts, new_last_end


def _resample_pcm16(x: np.ndarray, *, in_sr: int, out_sr: int) -> np.ndarray:
    if in_sr == out_sr:
        return x
    if x.size < 2:
        return x
    # Linear interpolation resampler (fast, good enough for speech STT).
    ratio = out_sr / in_sr
    out_len = max(1, int(round(x.size * ratio)))
    xp = np.arange(x.size, dtype=np.float32)
    fp = x.astype(np.float32)
    x_new = np.linspace(0, x.size - 1, out_len, dtype=np.float32)
    y = np.interp(x_new, xp, fp)
    return y.astype(np.int16)

