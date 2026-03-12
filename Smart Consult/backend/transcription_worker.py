from __future__ import annotations

import asyncio
import io
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel

from .audio_session import AudioSession
from .config import get_settings


settings = get_settings()

_model_lock = asyncio.Lock()
_model: Optional[WhisperModel] = None


def _load_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
    return _model


async def _load_model_async() -> WhisperModel:
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
    - We accumulate audio into a rolling buffer and call Whisper on windows (e.g. ~20–30 seconds).
    - We diff against previous transcript text so we only send incremental updates.
    - For RAM efficiency, we keep only small rolling context.
    """
    model = await _load_model_async()

    rolling_buffer = io.BytesIO()
    last_text = ""

    async for chunk in session.audio_iter():
        if not chunk:
            continue
        rolling_buffer.write(chunk)

        # When buffer passes threshold, transcribe the window.
        if rolling_buffer.tell() >= settings.AUDIO_SAMPLE_RATE * 2 * 10:  # ~10 seconds @16kHz, 16-bit mono
            text = await _transcribe_bytes(model, rolling_buffer.getvalue())
            # For simplicity, treat entire text as latest snapshot and diff by prefix length.
            if text and text != last_text:
                incremental = text[len(last_text) :].strip()
                if incremental:
                    await session.push_partial_transcript(incremental)
                    last_text = text

            # Truncate buffer to keep only last N seconds (rolling context).
            keep_ratio = 0.3
            data = rolling_buffer.getvalue()
            keep_from = int(len(data) * (1 - keep_ratio))
            rolling_buffer = io.BytesIO(data[keep_from:])

    # Flush any remainder at end of call.
    if rolling_buffer.tell() > 0:
        text = await _transcribe_bytes(model, rolling_buffer.getvalue())
        if text:
            final_incremental = text[len(last_text) :].strip()
            if final_incremental:
                await session.push_partial_transcript(final_incremental)

    await session.mark_transcription_finished()


async def _transcribe_bytes(model: WhisperModel, raw: bytes) -> str:
    """
    Run Whisper on raw PCM16 bytes.

    In production you might receive Opus/WebM or other formats; in that case
    transcode with ffmpeg first.
    """
    # Interpret raw bytes as int16 PCM.
    audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    loop = asyncio.get_running_loop()

    def _do_transcribe() -> str:
        segments, _info = model.transcribe(
            audio_np,
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=True,
            language="en",
        )
        return " ".join(seg.text.strip() for seg in segments)

    return await loop.run_in_executor(None, _do_transcribe)

