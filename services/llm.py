"""Groq LLM client with retries and shared configuration."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 3
BACKOFF_BASE = 1.0


def get_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Groq(api_key=api_key)


def get_llm_model() -> str:
    return DEFAULT_LLM_MODEL


def chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = 2048,
) -> str:
    """Invoke Groq chat completions with exponential backoff on transient errors."""
    client = get_client()
    model = model or get_llm_model()
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            if choice.message.content:
                return choice.message.content.strip()
            return ""
        except Exception as e:
            last_error = e
            logger.warning("LLM call failed (attempt %s/%s): %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE * (2**attempt))

    raise RuntimeError(f"Groq LLM failed after {MAX_RETRIES} attempts: {last_error}") from last_error
