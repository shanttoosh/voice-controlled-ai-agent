"""Helpers for converting recorded/uploaded audio to WAV bytes for Groq STT."""

from __future__ import annotations

import io
from typing import Any

import numpy as np


def numpy_to_wav_bytes(
    audio_array: Any,
    sample_rate: int,
) -> bytes:
    """
    Convert numpy audio (from audio-recorder-streamlit or sounddevice) to WAV bytes.

    Handles mono/stereo; normalizes shape for soundfile.
    """
    import soundfile as sf

    if audio_array is None:
        raise ValueError("No audio data provided")

    arr = np.asarray(audio_array)
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            # int16 PCM is typical for browser / mic capture
            peak = float(2 ** (8 * arr.dtype.itemsize - 1))
            arr = (arr.astype(np.float32) / peak).clip(-1.0, 1.0)
        else:
            arr = arr.astype(np.float32)
    if arr.size == 0:
        raise ValueError("Empty audio buffer")

    # Mono: (n,) or (n,1); stereo: (n,2)
    if arr.ndim == 1:
        pass
    elif arr.ndim == 2 and arr.shape[1] <= 2:
        pass
    else:
        arr = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 2 else arr.flatten()[:, np.newaxis]

    buf = io.BytesIO()
    sf.write(buf, arr, sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def guess_extension_from_name(filename: str) -> str:
    """Return lowercase extension including dot, default .wav."""
    lower = filename.lower()
    if lower.endswith(".mp3"):
        return ".mp3"
    if lower.endswith(".m4a"):
        return ".m4a"
    if lower.endswith(".webm"):
        return ".webm"
    if lower.endswith(".flac"):
        return ".flac"
    return ".wav"
