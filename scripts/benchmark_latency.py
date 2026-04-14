"""
Optional latency benchmark for Groq STT + LLM (requires GROQ_API_KEY in .env).

Usage:
  python scripts/benchmark_latency.py
  python scripts/benchmark_latency.py --audio path/to/file.wav
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def time_llm_intent() -> list[float]:
    from services.llm import chat_completion

    messages = [
        {"role": "system", "content": 'Reply with JSON only: {"intent":"general_chat","reply":"ok"}'},
        {"role": "user", "content": "ping"},
    ]
    times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        chat_completion(messages, temperature=0.0, max_tokens=64)
        times.append(time.perf_counter() - t0)
    return times


def time_stt(audio_path: Path) -> list[float]:
    from services.stt import transcribe_audio

    data = audio_path.read_bytes()
    name = audio_path.name
    times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        transcribe_audio(data, filename=name)
        times.append(time.perf_counter() - t0)
    return times


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, help="WAV/MP3 file for STT timing")
    args = parser.parse_args()

    print("=== Groq LLM (intent-style short call) ===")
    try:
        llm_times = time_llm_intent()
        print(f"Runs (s): {[round(t, 3) for t in llm_times]}")
        print(f"Median (s): {statistics.median(llm_times):.3f}")
    except Exception as e:
        print(f"LLM benchmark failed: {e}")

    if args.audio:
        if not args.audio.is_file():
            print(f"Audio not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        print("\n=== Groq Whisper STT ===")
        try:
            stt_times = time_stt(args.audio)
            print(f"Runs (s): {[round(t, 3) for t in stt_times]}")
            print(f"Median (s): {statistics.median(stt_times):.3f}")
        except Exception as e:
            print(f"STT benchmark failed: {e}")
    else:
        print("\n(STT skipped — pass --audio path/to.wav for Whisper timing)")


if __name__ == "__main__":
    main()
