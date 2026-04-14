"""Session memory helpers: format history for prompts and append turns."""

from __future__ import annotations

from typing import Any

MAX_TURNS = 5


def format_history_for_prompt(history: list[dict[str, Any]] | None, max_turns: int = MAX_TURNS) -> str:
    """Turn recent session turns into a short string for LLM context."""
    if not history:
        return ""

    lines: list[str] = []
    for turn in history[-max_turns:]:
        role = turn.get("role", "user")
        tr = turn.get("transcript", "")
        intent = turn.get("intent", "")
        result = turn.get("result", "")
        lines.append(f"- [{role}] said: {tr!r} | intent: {intent} | outcome: {result[:200]!s}")

    return "\n".join(lines)


def append_turn(
    history: list[dict[str, Any]] | None,
    *,
    transcript: str,
    intent: str,
    action_taken: str,
    tool_result: str,
) -> list[dict[str, Any]]:
    """Return a new history list with one assistant/user turn appended."""
    h = list(history or [])
    h.append(
        {
            "role": "user",
            "transcript": transcript,
            "intent": intent,
            "action_taken": action_taken,
            "result": tool_result,
        }
    )
    return h
