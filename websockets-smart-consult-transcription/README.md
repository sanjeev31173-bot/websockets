## NHS realtime doctor–patient consult demo

This project is a production-oriented skeleton for streaming **doctor–patient audio** over WebSockets and **transcribing in realtime with Whisper** for clinician documentation.

### Architecture overview

- **Backend** (`backend/`):
  - FastAPI app exposing:
    - `GET /health` – readiness / liveness
    - `WS /ws/consultation/{session_id}` – bidirectional audio + transcript stream
  - `AudioSession` abstraction that:
    - Buffers audio for each consultation with bounded queues (backpressure).
    - Streams partial transcript updates back to the WebSocket.
- Whisper worker:
    - Uses a **fixed-size rolling buffer** to transcribe long calls without unbounded RAM.
    - Emits incremental text using **segment timestamps** (avoids duplication as the buffer rolls).
    - Runs heavy Whisper inference in a threadpool to avoid blocking the event loop.

- **Frontend** (`frontend/`):
  - Single-page HTML/JS app using `MediaRecorder` to capture microphone audio.
  - Streams audio chunks over WebSocket to the backend and renders:
    - **Live transcript** as text arrives.

### Running locally

1. **Install dependencies** (no venv required):

```bash
pip install -r requirements.txt
```

4. **Start the backend**:

```bash
python server.py
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
  - Centralised monitoring (Prometheus, OpenTelemetry).

You can evolve this design into a full microservice architecture (separate **audio gateway**, **transcription workers**, and **EHR integration service**) while keeping the same basic streaming protocol.

