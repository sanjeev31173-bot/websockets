from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    # Run with a single worker by default for Windows simplicity.
    # In production Linux, prefer multiple workers: `uvicorn backend.main:app --workers N`
    uvicorn.run("backend.main:app", host=host, port=port, reload=os.getenv("RELOAD", "0") == "1")


if __name__ == "__main__":
    main()
