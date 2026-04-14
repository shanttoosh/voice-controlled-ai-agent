"""LangGraph agent state schema."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    """State carried through the LangGraph pipeline."""

    # Audio / STT
    audio_bytes: bytes | None
    audio_filename: str
    skip_transcribe: bool
    skip_classify: bool

    # Transcript & intent
    transcript: str
    intent: str
    intent_details: dict[str, Any]
    compound: bool
    compound_steps: list[dict[str, Any]] | None

    # Human-in-the-loop
    require_hitl: bool
    human_approved: bool
    human_pending: bool

    # Tool outputs
    tool_result: str
    action_taken: str
    error: str | None

    # Session memory (read-only context for classification / chat; UI persists separately)
    history: list[dict[str, Any]]
