from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from .config import get_settings


_settings = get_settings()


@dataclass
class AudioSession:
    """Holds per-consultation state."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    client_sample_rate_hz: int = _settings.AUDIO_SAMPLE_RATE
    audio_queue: "asyncio.Queue[bytes]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=_settings.MAX_AUDIO_QUEUE_CHUNKS)
    )
    transcript_queue: "asyncio.Queue[str]" = field(
        default_factory=lambda: asyncio.Queue(maxsize=_settings.MAX_TRANSCRIPT_QUEUE_MESSAGES)
    )
    full_transcript: list[str] = field(default_factory=list)
    is_closed: bool = False

    async def add_audio_chunk(self, chunk: bytes) -> None:
        if self.is_closed:
            return
        try:
            await asyncio.wait_for(self.audio_queue.put(chunk), timeout=_settings.AUDIO_PUT_TIMEOUT_SEC)
        except TimeoutError:
            # Client is producing faster than we can consume; fail closed to protect memory.
            await self.mark_closed()

    async def mark_closed(self) -> None:
        self.is_closed = True
        # Sentinel to indicate end-of-stream for consumers (guaranteed).
        while True:
            try:
                self.audio_queue.put_nowait(b"")
                break
            except asyncio.QueueFull:
                try:
                    self.audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)

    async def audio_iter(self) -> AsyncIterator[bytes]:
        """Yield audio chunks until session is closed."""
        while True:
            chunk = await self.audio_queue.get()
            if chunk == b"":
                break
            yield chunk

    async def push_partial_transcript(self, text: str) -> None:
        """Push an updated transcript snapshot to the websocket producer."""
        self.full_transcript.append(text)
        try:
            self.transcript_queue.put_nowait(text)
        except asyncio.QueueFull:
            # If the consumer is slow, drop transcript messages rather than crashing.
            # The full transcript is still accumulated for final analysis.
            pass

    async def transcript_updates(self) -> AsyncIterator[str]:
        """Yield transcript updates."""
        while True:
            update = await self.transcript_queue.get()
            if update == "__END__":
                break
            yield update

    async def mark_transcription_finished(self) -> None:
        while True:
            try:
                self.transcript_queue.put_nowait("__END__")
                break
            except asyncio.QueueFull:
                try:
                    self.transcript_queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0)

