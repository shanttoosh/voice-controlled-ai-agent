"""Groq Whisper speech-to-text."""

from __future__ import annotations

import io
import logging
import os
import time
from typing import BinaryIO

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
MAX_RETRIES = 3
BACKOFF_BASE = 1.0


def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
    return Groq(api_key=api_key)


def transcribe_audio(
    audio_data: bytes,
    *,
    filename: str = "audio.wav",
    model: str | None = None,
    mime_hint: str | None = None,
) -> str:
    """
    Transcribe audio bytes using Groq Whisper API.

    Args:
        audio_data: Raw audio file bytes (WAV, MP3, etc.).
        filename: Filename used for API hint (extension matters for some backends).
        model: Whisper model id on Groq.
        mime_hint: Optional MIME type if known.

    Returns:
        Transcribed text.
    """
    model = model or DEFAULT_WHISPER_MODEL
    client = _client()
    last_error: Exception | None = None

    # Groq SDK accepts a readable file-like object; set .name for multipart filename
    for attempt in range(MAX_RETRIES):
        try:
            buf = io.BytesIO(audio_data)
            buf.name = filename
            transcription = client.audio.transcriptions.create(
                file=buf,
                model=model,
                response_format="json",
            )
            text = getattr(transcription, "text", None) or ""
            return text.strip()
        except Exception as e:
            last_error = e
            logger.warning("STT failed (attempt %s/%s): %s", attempt + 1, MAX_RETRIES, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_BASE * (2**attempt))

    raise RuntimeError(f"Groq Whisper transcription failed after {MAX_RETRIES} attempts: {last_error}") from last_error


def transcribe_file_object(file_obj: BinaryIO, filename: str) -> str:
    """Read all bytes from a file-like object and transcribe."""
    data = file_obj.read()
    return transcribe_audio(data, filename=filename)
