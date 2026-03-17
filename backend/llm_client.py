from __future__ import annotations

from typing import Any, Dict

from openai import OpenAI

from .config import get_settings


settings = get_settings()


def _get_client() -> OpenAI:
    if not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set but is required for LLM analysis.")
    return OpenAI(api_key=settings.OPENAI_API_KEY)


def analyze_consultation(transcript: str) -> Dict[str, Any]:
    """
    Send the full consultation transcript to the LLM and return a structured analysis.

    This is called at the end of the call (or periodically for long consults).
    """
    client = _get_client()

    completion = client.chat.completions.create(
        model=settings.LLM_MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": settings.NHS_GUIDELINES_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Here is the verbatim consultation transcript between patient and doctor. "
                    "Return a JSON object with keys: 'history', 'red_flags', 'differentiels'.\n\n"
                    f"{transcript}"
                ),
            },
        ],
    )

    content = completion.choices[0].message.content
    # The client guarantees JSON when response_format is json_object,
    # but types may still be str depending on version.
    if isinstance(content, str):
        import json

        return json.loads(content)
    return content  # type: ignore[return-value]

