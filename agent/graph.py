"""LangGraph definition: transcribe, classify, optional HITL gate, tool execution."""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agent.intent import (
    INTENT_CREATE_FILE,
    INTENT_GENERAL_CHAT,
    INTENT_SUMMARIZE,
    INTENT_WRITE_CODE,
    classify_intent,
)
from agent.memory import format_history_for_prompt
from agent.state import AgentState
from services.stt import transcribe_audio
from tools.chat import execute_general_chat
from tools.code_gen import execute_write_code
from tools.file_ops import execute_create_file
from tools.summarizer import execute_summarize

logger = logging.getLogger(__name__)

FILE_INTENTS = frozenset({INTENT_CREATE_FILE, INTENT_WRITE_CODE})


def transcribe_node(state: AgentState) -> dict[str, Any]:
    if state.get("skip_transcribe") and state.get("transcript"):
        return {}

    audio = state.get("audio_bytes")
    if not audio:
        return {
            "error": "No audio data provided.",
            "transcript": "",
            "tool_result": "No audio to transcribe.",
            "action_taken": "Stopped: missing audio",
        }

    name = state.get("audio_filename") or "audio.wav"

    try:
        text = transcribe_audio(audio, filename=name)
        return {"transcript": text, "error": None}
    except Exception as e:
        logger.exception("Transcription failed: %s", e)
        return {
            "error": str(e),
            "transcript": "",
            "tool_result": "Speech-to-text failed. Check your audio format and API key.",
            "action_taken": "Transcription error",
        }


def classify_node(state: AgentState) -> dict[str, Any]:
    if state.get("skip_classify") and state.get("intent"):
        return {}

    if state.get("error") and not (state.get("transcript") or "").strip():
        return {}

    transcript = state.get("transcript") or ""
    if not transcript.strip():
        return {
            "intent": INTENT_GENERAL_CHAT,
            "intent_details": {"description": "empty transcript"},
            "compound": False,
            "compound_steps": None,
            "tool_result": "No speech detected in the audio.",
            "action_taken": "Empty transcription",
        }

    hist = state.get("history") or []
    ctx = format_history_for_prompt(hist)
    out = classify_intent(transcript, history_context=ctx or None)

    return {
        "intent": out["intent"],
        "intent_details": out["intent_details"],
        "compound": out.get("compound", False),
        "compound_steps": out.get("compound_steps"),
        "error": None,
    }


def hitl_node(state: AgentState) -> dict[str, Any]:
    return {
        "human_pending": True,
        "action_taken": "Waiting for your confirmation before changing files",
        "tool_result": (
            "This action will create or modify files under the output/ folder. "
            "Confirm in the UI to proceed."
        ),
    }


def route_after_classify(state: AgentState) -> Literal["hitl", "tools", "end"]:
    if state.get("error") and not (state.get("transcript") or "").strip():
        return "end"
    if (not (state.get("transcript") or "").strip()) and state.get("action_taken") == "Empty transcription":
        return "end"

    if state.get("require_hitl") and not state.get("human_approved"):
        if state.get("intent") in FILE_INTENTS:
            return "hitl"
    return "tools"


def _run_single(intent: str, details: dict[str, Any], transcript: str, history: list | None) -> dict[str, str]:
    hist_snippet = format_history_for_prompt(history or [])

    if intent == INTENT_CREATE_FILE:
        return execute_create_file(details, transcript)
    if intent == INTENT_WRITE_CODE:
        return execute_write_code(details, transcript)
    if intent == INTENT_SUMMARIZE:
        return execute_summarize(details, transcript)
    return execute_general_chat(details, transcript, history_snippet=hist_snippet)


def tools_node(state: AgentState) -> dict[str, Any]:
    transcript = state.get("transcript") or ""
    history = state.get("history") or []

    if state.get("compound") and state.get("compound_steps"):
        parts_action: list[str] = []
        parts_result: list[str] = []
        # Pass text from summarize/chat (or similar) into a following create_file step
        chained_text: str | None = None
        for i, step in enumerate(state["compound_steps"] or [], start=1):
            inn = step.get("intent", INTENT_GENERAL_CHAT)
            details = dict(step)
            if inn == INTENT_CREATE_FILE and chained_text:
                details.setdefault("content", chained_text)
            res = _run_single(inn, details, transcript, history)
            parts_action.append(f"Step {i} ({inn}): {res['action_taken']}")
            parts_result.append(f"Step {i}:\n{res['tool_result']}")
            if inn in (INTENT_SUMMARIZE, INTENT_GENERAL_CHAT):
                chained_text = res["tool_result"]
        return {
            "action_taken": " | ".join(parts_action),
            "tool_result": "\n\n".join(parts_result),
            "human_pending": False,
        }

    intent = state.get("intent") or INTENT_GENERAL_CHAT
    details = state.get("intent_details") or {}
    res = _run_single(intent, details, transcript, history)
    return {
        "action_taken": res["action_taken"],
        "tool_result": res["tool_result"],
        "human_pending": False,
    }


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("transcribe", transcribe_node)
    g.add_node("classify", classify_node)
    g.add_node("hitl", hitl_node)
    g.add_node("tools", tools_node)

    g.set_entry_point("transcribe")
    g.add_edge("transcribe", "classify")
    g.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "hitl": "hitl",
            "tools": "tools",
            "end": END,
        },
    )
    g.add_edge("hitl", END)
    g.add_edge("tools", END)

    return g.compile()


def run_pipeline(initial: AgentState) -> AgentState:
    """Run the compiled graph and return final state."""
    graph = build_graph()
    result = graph.invoke(initial)
    return result  # type: ignore[return-value]


def run_tools_only(state: AgentState) -> AgentState:
    """Execute tool node logic without STT/classify (after HITL approval)."""
    out = tools_node(state)
    merged: AgentState = {**state, **out}
    return merged
