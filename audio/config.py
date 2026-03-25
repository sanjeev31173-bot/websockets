"""
config.py  —  v3 single-worker configuration
"""
import os


class Settings:
    APP_NAME: str = "NHS Realtime Consult"
    ENV: str = os.getenv("APP_ENV", "local")

    # ── Whisper ────────────────────────────────────────────────────────────────
    WHISPER_MODEL_SIZE: str   = os.getenv("WHISPER_MODEL_SIZE", "small.en")
    WHISPER_DEVICE: str       = os.getenv("WHISPER_DEVICE", "cpu")
    WHISPER_COMPUTE_TYPE: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    WHISPER_LANGUAGE: str     = os.getenv("WHISPER_LANGUAGE", "en")
    WHISPER_VAD_FILTER: bool  = os.getenv("WHISPER_VAD_FILTER", "true").lower() in {"1", "true", "yes"}
    WHISPER_BEAM_SIZE: int    = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

    # Medical vocab hint — prepended as initial_prompt to every transcription
    MEDICAL_VOCAB_PROMPT: str = os.getenv(
        "MEDICAL_VOCAB_PROMPT",
        "Doctor patient consultation. "
        "Good morning, doctor. I'm not feeling well. I have a headache and a sore throat. "
        "How long have you had these symptoms? For about three days. Do you have a fever? "
        "Yes, a little. Mostly at night. Are you taking any medicine right now? "
        "Just some painkillers, but they don't help much. Do you have any allergies? No, I don't. "
        "I'll give you some medicine and advise rest. How many days should I rest? "
        "At least three to five days. Drink warm fluids and take the medicine on time. "
        "NHS UK, GP, patient, symptoms, diagnosis, prescription, referral, "
        "paracetamol, ibuprofen, amoxicillin, hypertension, diabetes, asthma, "
        "blood pressure, triage, outpatient, inpatient, discharge."
    )

    # ── Audio ──────────────────────────────────────────────────────────────────
    AUDIO_SAMPLE_RATE: int    = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    MAX_SESSION_MINUTES: int  = int(os.getenv("MAX_SESSION_MINUTES", "60"))
    AUDIO_CHANNELS: int       = int(os.getenv("AUDIO_CHANNELS", "1"))

    # ── Single-worker sequential pipeline ─────────────────────────────────────
    # With 1 model we process the FULL audio when session ends (most accurate),
    # AND stream rolling VAD-gated segments during the session for live display.
    NUM_WORKERS: int = 1

    # VAD segment window: how much audio we feed per rolling pass
    # 20s is wide enough to catch any sentence without splitting it.
    SEGMENT_WINDOW_SEC: float = float(os.getenv("SEGMENT_WINDOW_SEC", "20.0"))

    # Overlap between consecutive rolling windows (catches boundary sentences)
    SEGMENT_OVERLAP_SEC: float = float(os.getenv("SEGMENT_OVERLAP_SEC", "5.0"))

    # How many seconds of audio must accumulate before the first pass fires
    MIN_CHUNK_TO_START_SEC: float = float(os.getenv("MIN_CHUNK_TO_START_SEC", "1.0"))

    # VAD settings — conservative to avoid cutting sentences
    VAD_MIN_SILENCE_MS: int = int(os.getenv("VAD_MIN_SILENCE_MS", "700"))

    # Live partial display window (seconds shown in the "typing..." indicator)
    LIVE_WINDOW_SEC: float  = float(os.getenv("LIVE_WINDOW_SEC", "10.0"))
    SILENCE_TAIL_SEC: float = float(os.getenv("SILENCE_TAIL_SEC", "0.8"))

    # ── Queue limits ───────────────────────────────────────────────────────────
    MAX_AUDIO_QUEUE_CHUNKS: int        = int(os.getenv("MAX_AUDIO_QUEUE_CHUNKS", "400"))
    MAX_TRANSCRIPT_QUEUE_MESSAGES: int = int(os.getenv("MAX_TRANSCRIPT_QUEUE_MESSAGES", "2000"))
    AUDIO_PUT_TIMEOUT_SEC: float       = float(os.getenv("AUDIO_PUT_TIMEOUT_SEC", "2.0"))
    MAX_CONNECTIONS_PER_WORKER: int    = int(os.getenv("MAX_CONNECTIONS_PER_WORKER", "2000"))


def get_settings() -> Settings:
    return Settings()