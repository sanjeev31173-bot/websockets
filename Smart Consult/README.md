## NHS realtime doctor–patient consult demo

This project is a production-oriented skeleton for streaming **doctor–patient audio** over WebSockets, **transcribing in realtime with Whisper**, and sending the resulting transcript to an **LLM with NHS-style guidelines** for structured clinical summarisation and differential diagnosis.

### Architecture overview

- **Backend** (`backend/`):
  - FastAPI app exposing:
    - `GET /health` – readiness / liveness
    - `WS /ws/consultation/{session_id}` – bidirectional audio + transcript stream
  - `AudioSession` abstraction that:
    - Buffers audio for each consultation with bounded queues (backpressure).
    - Streams partial transcript updates back to the WebSocket.
    - Aggregates full transcript text for the LLM at the end.
  - `faster-whisper`–based worker:
    - Uses a **rolling buffer** to transcribe long calls without unbounded RAM.
    - Runs heavy Whisper inference in a threadpool to avoid blocking the event loop.
  - `llm_client`:
    - Sends final transcript to an LLM (e.g. OpenAI `gpt-4o-mini`) using an NHS-focused system prompt.
    - Returns structured JSON suitable for downstream storage or display.

- **Frontend** (`frontend/`):
  - Single-page HTML/JS app using `MediaRecorder` to capture microphone audio.
  - Streams audio chunks over WebSocket to the backend and renders:
    - **Live transcript** as text arrives.
    - **Final LLM analysis** once transcription finishes.

### Running locally

1. **Create and activate a virtual environment** (recommended).

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Set OpenAI credentials** (for the LLM analysis step):

```bash
set OPENAI_API_KEY=sk-...
```

4. **Start the backend**:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

5. **Serve the frontend**:

You can use any static file server. For example, with Python:

```bash
cd frontend
python -m http.server 5173
```

Then open `http://localhost:5173/index.html` in your browser. The page will connect back to `http://localhost:8000` for the WebSocket if hosted on the same machine.

> In production you would typically serve the SPA and API behind a single HTTPS origin and terminate TLS at a reverse proxy (e.g. Nginx, Envoy, or an API gateway).

### Production considerations (NHS context)

This repository is a **starting point**, not a complete NHS-ready deployment. For a real implementation you will additionally need:

- **Security & compliance**
  - Mutual TLS, OAuth2 / OIDC, and strict RBAC for clinicians.
  - At-rest encryption for audio and transcripts.
  - Proper logging, auditing, and data retention policies that satisfy NHS and UK GDPR requirements.
- **Scalability**
  - Run multiple `uvicorn` workers per node.
  - Horizontally scale nodes behind a load balancer with sticky sessions.
  - Offload Whisper to GPU-backed workers (e.g. via a queue) for heavy throughput.
- **Resilience**
  - Graceful handling of WebSocket drops and reconnections (session ID is already part of the URL).
  - Centralised monitoring (Prometheus, OpenTelemetry) and circuit breaking around the LLM provider.

You can evolve this design into a full microservice architecture (separate **audio gateway**, **transcription workers**, **LLM service**, and **EHR integration service**) while keeping the same basic streaming protocol.

