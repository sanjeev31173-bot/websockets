import os
from functools import lru_cache


class Settings:
    """Application configuration."""

    APP_NAME: str = "NHS Realtime Consult"
    ENV: str = os.getenv("APP_ENV", "local")

    # Whisper / transcription
    # Default to `base` for CPU-friendly realtime. Override to `small/medium/large` via env.
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")  # or "cuda"
    # `openai-whisper` doesn't support compute_type quantization like faster-whisper.
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_LANGUAGE: str = os.getenv("WHISPER_LANGUAGE", "en")
    WHISPER_VAD_FILTER: bool = os.getenv("WHISPER_VAD_FILTER", "true").lower() in {"1", "true", "yes"}
    WHISPER_BEAM_SIZE: int = int(os.getenv("WHISPER_BEAM_SIZE", "1"))

    # Audio streaming
    MAX_SESSION_MINUTES: int = int(os.getenv("MAX_SESSION_MINUTES", "60"))
    MAX_CONNECTIONS_PER_WORKER: int = int(os.getenv("MAX_CONNECTIONS_PER_WORKER", "2000"))
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))
    AUDIO_BUFFER_SECONDS: int = int(os.getenv("AUDIO_BUFFER_SECONDS", "30"))
    # Whisper CPU decoding is expensive; these defaults avoid "stuck" behaviour on Windows CPU.
    TRANSCRIBE_EVERY_MS: int = int(os.getenv("TRANSCRIBE_EVERY_MS", "2500"))
    MIN_TRANSCRIBE_WINDOW_MS: int = int(os.getenv("MIN_TRANSCRIBE_WINDOW_MS", "6000"))
    MAX_AUDIO_QUEUE_CHUNKS: int = int(os.getenv("MAX_AUDIO_QUEUE_CHUNKS", "400"))
    MAX_TRANSCRIPT_QUEUE_MESSAGES: int = int(os.getenv("MAX_TRANSCRIPT_QUEUE_MESSAGES", "2000"))
    AUDIO_PUT_TIMEOUT_SEC: float = float(os.getenv("AUDIO_PUT_TIMEOUT_SEC", "2.0"))

    # LLM settings removed for now (streaming + transcription only).


@lru_cache
def get_settings() -> Settings:
    return Settings()

