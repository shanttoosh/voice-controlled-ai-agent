"""Intent classification using Groq LLM with structured JSON output."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from services.llm import chat_completion

logger = logging.getLogger(__name__)

INTENT_CREATE_FILE = "create_file"
INTENT_WRITE_CODE = "write_code"
INTENT_SUMMARIZE = "summarize"
INTENT_GENERAL_CHAT = "general_chat"

VALID_INTENTS = frozenset(
    {
        INTENT_CREATE_FILE,
        INTENT_WRITE_CODE,
        INTENT_SUMMARIZE,
        INTENT_GENERAL_CHAT,
    }
)

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a voice-controlled AI agent.
Given the user's transcribed speech and optional recent conversation context, classify the intent.

Supported intents (exactly ONE primary intent unless compound is true):
- "create_file": user wants to create a new empty file or folder under the sandbox output folder
- "write_code": user wants to generate programming code and save it to a file
- "summarize": user wants to summarize provided text or content they will paste / describe
- "general_chat": greetings, questions not matching above, or unclear requests

If the user gives MULTIPLE commands in one utterance (e.g. "summarize this and save to summary.txt"),
set "compound": true and fill "intents" with an ordered list of steps, each with the same schema fields needed for that step.

Respond ONLY with valid JSON (no markdown fences):
{
  "intent": "<primary_intent_name>",
  "compound": false,
  "intents": [],
  "filename": "<suggested filename or null>",
  "folder": "<optional subfolder under output or null>",
  "language": "<programming language or null>",
  "description": "<brief description of what the user wants>",
  "content": "<any text content to summarize or null>",
  "is_folder": false
}

Use null for unknown optional fields. Use "intents" only when compound is true; each item: {"intent": "...", "filename": ..., "description": ..., "content": ..., "language": ...}.
"""


def _extract_json_block(text: str) -> str:
    """Strip markdown code fences and extract JSON object or array."""
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return text


def _parse_intent_payload(text: str) -> dict[str, Any]:
    raw = _extract_json_block(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find first {...} block
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError("Could not parse intent JSON")


def normalize_intent_result(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Ensure payload has intent, intent_details, and optional compound_steps.
    """
    compound = bool(payload.get("compound"))
    steps: list[dict[str, Any]] = []

    if compound and payload.get("intents"):
        for item in payload["intents"]:
            if not isinstance(item, dict):
                continue
            inn = item.get("intent", INTENT_GENERAL_CHAT)
            if inn not in VALID_INTENTS:
                inn = INTENT_GENERAL_CHAT
            steps.append(
                {
                    "intent": inn,
                    "filename": item.get("filename"),
                    "folder": item.get("folder"),
                    "language": item.get("language"),
                    "description": item.get("description") or payload.get("description"),
                    "content": item.get("content"),
                    "is_folder": bool(item.get("is_folder", False)),
                }
            )

    intent = payload.get("intent", INTENT_GENERAL_CHAT)
    if intent not in VALID_INTENTS:
        intent = INTENT_GENERAL_CHAT

    intent_details: dict[str, Any] = {
        "filename": payload.get("filename"),
        "folder": payload.get("folder"),
        "language": payload.get("language"),
        "description": payload.get("description"),
        "content": payload.get("content"),
        "is_folder": bool(payload.get("is_folder", False)),
    }

    return {
        "intent": intent,
        "intent_details": intent_details,
        "compound": compound and len(steps) > 0,
        "compound_steps": steps if (compound and steps) else None,
    }


def classify_intent(transcript: str, history_context: str | None = None) -> dict[str, Any]:
    """
    Classify user transcript into intent + details. On failure, returns general_chat.

    Returns:
        dict with keys: intent, intent_details, compound (bool), compound_steps (optional list)
    """
    ctx = ""
    if history_context:
        ctx = f"\n\nRecent conversation (for context):\n{history_context}\n"

    user_msg = f"Transcript:{ctx}\n\n\"\"\"\n{transcript}\n\"\"\""

    messages = [
        {"role": "system", "content": INTENT_CLASSIFICATION_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = chat_completion(messages, temperature=0.1)
        payload = _parse_intent_result(raw)
        return normalize_intent_result(payload)
    except Exception as e:
        logger.exception("Intent classification failed: %s", e)
        return {
            "intent": INTENT_GENERAL_CHAT,
            "intent_details": {
                "filename": None,
                "folder": None,
                "language": None,
                "description": transcript,
                "content": None,
                "is_folder": False,
            },
            "compound": False,
            "compound_steps": None,
            "parse_error": str(e),
        }


def _parse_intent_result(raw: str) -> dict[str, Any]:
    return _parse_intent_payload(raw)
