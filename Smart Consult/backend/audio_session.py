from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional


@dataclass
class AudioSession:
    """Holds per-consultation state."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    audio_queue: "asyncio.Queue[bytes]" = field(default_factory=asyncio.Queue)
    transcript_queue: "asyncio.Queue[str]" = field(default_factory=asyncio.Queue)
    full_transcript: list[str] = field(default_factory=list)
    is_closed: bool = False

    async def add_audio_chunk(self, chunk: bytes) -> None:
        if self.is_closed:
            return
        # Backpressure: wait if queue is full
        await self.audio_queue.put(chunk)

    async def mark_closed(self) -> None:
        self.is_closed = True
        # Sentinel to indicate end-of-stream for consumers
        await self.audio_queue.put(b"")

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
        await self.transcript_queue.put(text)

    async def transcript_updates(self) -> AsyncIterator[str]:
        """Yield transcript updates."""
        while True:
            update = await self.transcript_queue.get()
            if update == "__END__":
                break
            yield update

    async def mark_transcription_finished(self) -> None:
        await self.transcript_queue.put("__END__")

