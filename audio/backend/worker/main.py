"""
backend/worker/main.py  —  SINGLE MODEL v3
==========================================

One WhisperModel instance, using ALL available CPU threads.
No pool, no parallelism — just one model that processes audio sequentially.

Why single model?
-----------------
For live consultation transcription, audio arrives in real time.
You never have more than one chunk to process at a time, so multiple
workers buy you nothing — they just waste RAM and startup time.

One model with all 12 CPU threads runs FASTER than 12 models with 1 thread
each because CTranslate2's internal BLAS operations scale well within a
single inference call.
"""

import os
import sys
import numpy as np
from faster_whisper import WhisperModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import get_settings

settings = get_settings()

SR          = settings.AUDIO_SAMPLE_RATE
NUM_WORKERS = 1   # always 1 — the pool interface still uses this constant

cpu_count = os.cpu_count() or 4
print(f"[worker] Loading 1 × faster-whisper '{settings.WHISPER_MODEL_SIZE}' "
      f"on {settings.WHISPER_DEVICE}/{settings.WHISPER_COMPUTE_TYPE} "
      f"with {cpu_count} CPU threads ...")

# Single model — all CPU threads for maximum single-inference speed
_model = WhisperModel(
    settings.WHISPER_MODEL_SIZE,
    device=settings.WHISPER_DEVICE,
    compute_type=settings.WHISPER_COMPUTE_TYPE,
    cpu_threads=cpu_count,   # give it everything
    num_workers=1,
)

# model_pool[0] is the one model; gateway imports model_pool
model_pool: list[WhisperModel] = [_model]

print(f"[worker] model loaded ({cpu_count} CPU threads)")

# Warmup — white noise pass so the first real inference isn't slow
print("[worker] Warming up ...")
_noise = (np.random.randn(SR * 2).astype(np.float32) * 0.05)
list(_model.transcribe(
    _noise,
    language="en",
    beam_size=settings.WHISPER_BEAM_SIZE,
    vad_filter=False,
    word_timestamps=False,
    temperature=0.0,
))
print("[worker] model warmed up — ready.")