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
    # Only accumulate final segments for the saved transcript
    full_transcript: list[str] = field(default_factory=list)
    is_closed: bool = False

    async def add_audio_chunk(self, chunk: bytes) -> None:
        if self.is_closed:
            return
        try:
            await asyncio.wait_for(
                self.audio_queue.put(chunk),
                timeout=_settings.AUDIO_PUT_TIMEOUT_SEC,
            )
        except TimeoutError:
            await self.mark_closed()

    async def mark_closed(self) -> None:
        self.is_closed = True
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
        while True:
            chunk = await self.audio_queue.get()
            if chunk == b"":
                break
            yield chunk

    async def push_partial_transcript(self, msg: str) -> None:
        """
        Push a JSON-encoded transcript message to the websocket consumer.
        msg should be one of:
          {"type": "final",   "text": "..."}
          {"type": "partial", "text": "..."}
          {"type": "warning", "text": "..."}
        """
        import json
        try:
            parsed = json.loads(msg)
            if parsed.get("type") == "final":
                self.full_transcript.append(parsed.get("text", ""))
        except Exception:
            pass

        try:
            self.transcript_queue.put_nowait(msg)
        except asyncio.QueueFull:
            # Drop if the consumer is slow; full_transcript still preserved
            pass

    async def transcript_updates(self) -> AsyncIterator[str]:
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