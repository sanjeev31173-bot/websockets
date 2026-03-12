import os
from functools import lru_cache


class Settings:
    """Application configuration."""

    APP_NAME: str = "NHS Realtime Consult"
    ENV: str = os.getenv("APP_ENV", "local")

    # Whisper / transcription
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "medium")
    WHISPER_DEVICE: str = os.getenv("WHISPER_DEVICE", "cpu")  # or "cuda"
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # Audio streaming
    MAX_SESSION_MINUTES: int = int(os.getenv("MAX_SESSION_MINUTES", "60"))
    MAX_CONNECTIONS_PER_WORKER: int = int(os.getenv("MAX_CONNECTIONS_PER_WORKER", "2000"))
    AUDIO_SAMPLE_RATE: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    AUDIO_CHANNELS: int = int(os.getenv("AUDIO_CHANNELS", "1"))

    # LLM
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

    # Domain-specific
    NHS_GUIDELINES_SYSTEM_PROMPT: str = os.getenv(
        "NHS_GUIDELINES_SYSTEM_PROMPT",
        (
            "You are an NHS UK clinical decision support assistant. "
            "You receive an exact transcript of a consultation between a patient and a doctor. "
            "Summarize the history in structured form, highlight red flags, and list 3–5 likely "
            "differential diagnoses with brief rationale. Do NOT give treatment plans; this is "
            "for clinicians only."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()

