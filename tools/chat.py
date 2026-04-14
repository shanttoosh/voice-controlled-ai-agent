"""General chat using Groq LLM."""

from __future__ import annotations

import logging
from typing import Any

from services.llm import chat_completion

logger = logging.getLogger(__name__)


def execute_general_chat(
    intent_details: dict[str, Any],
    transcript: str,
    history_snippet: str | None = None,
) -> dict[str, str]:
    """Friendly assistant reply without file operations."""
    system = (
        "You are a helpful voice-controlled assistant. "
        "Be concise. The user spoke their message; respond naturally."
    )
    user_parts = [f"User said: {transcript}"]
    if intent_details.get("description"):
        user_parts.append(f"Context: {intent_details['description']}")
    if history_snippet:
        user_parts.append(f"Earlier in session:\n{history_snippet}")

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]

    try:
        reply = chat_completion(messages, temperature=0.6)
        return {
            "action_taken": "Responded in general chat mode",
            "tool_result": reply,
        }
    except RuntimeError as e:
        logger.exception("Chat failed: %s", e)
        return {
            "action_taken": "Chat response failed",
            "tool_result": "Sorry, I could not generate a reply. Please try again.",
        }
