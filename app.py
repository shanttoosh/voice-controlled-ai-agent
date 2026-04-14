"""
Streamlit UI: microphone or file upload, Groq STT + LangGraph agent, results display.
"""

from __future__ import annotations

import logging

import streamlit as st

from agent.graph import run_pipeline, run_tools_only
from agent.memory import append_turn, format_history_for_prompt
from agent.state import AgentState
from config import OUTPUT_DIR
from utils.audio import numpy_to_wav_bytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional mic widget
try:
    from audio_recorder_streamlit import audio_recorder
except ImportError:
    audio_recorder = None


def _init_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []
    if "pending_hitl" not in st.session_state:
        st.session_state.pending_hitl = None
    if "last_pipeline_result" not in st.session_state:
        st.session_state.last_pipeline_result = None


def main() -> None:
    st.set_page_config(page_title="Voice AI Agent", page_icon="🎙️", layout="wide")
    _init_session()

    st.title("Voice-Controlled AI Agent")
    st.caption("Groq Whisper + LLaMA via Groq · LangGraph · Files restricted to `output/`")

    with st.sidebar:
        st.header("Settings")
        require_hitl = st.checkbox(
            "Confirm before file operations",
            value=False,
            help="When enabled, you must click Approve before any file is created or code is saved under output/.",
        )
        st.session_state.require_hitl = require_hitl

        st.divider()
        st.subheader("Session memory")
        hist_text = format_history_for_prompt(st.session_state.history)
        if hist_text:
            st.text_area("Recent context (used by classifier)", hist_text, height=160, disabled=True)
        else:
            st.info("No turns yet this session.")

        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.pending_hitl = None
            st.session_state.last_pipeline_result = None
            st.rerun()

    col1, col2 = st.columns([1, 1])
    with col1:
        input_mode = st.radio("Audio input", ["Upload file", "Microphone"], horizontal=True)

    audio_bytes: bytes | None = None
    audio_filename = "audio.wav"

    with col1:
        if input_mode == "Upload file":
            up = st.file_uploader("Audio file", type=["wav", "mp3", "m4a", "webm", "flac"])
            if up is not None:
                audio_bytes = up.getvalue()
                audio_filename = up.name or "audio.wav"
        else:
            if audio_recorder is None:
                st.warning("Install `audio-recorder-streamlit` for microphone support, or use file upload.")
            else:
                st.caption("Allow microphone access in the browser when prompted.")
                audio = audio_recorder(
                    text="Click to record",
                    recording_color="#e74c3c",
                    neutral_color="#6c757d",
                    icon_name="microphone",
                    icon_size="2x",
                )
                if audio is not None:
                    try:
                        if isinstance(audio, (bytes, bytearray)):
                            audio_bytes = bytes(audio)
                            audio_filename = "microphone.wav"
                        else:
                            sr = 44100
                            audio_bytes = numpy_to_wav_bytes(audio, sr)
                            audio_filename = "microphone.wav"
                    except Exception as e:
                        st.error(f"Could not encode microphone audio: {e}")

    pending = st.session_state.pending_hitl
    run = st.button(
        "Process audio",
        type="primary",
        use_container_width=True,
        disabled=bool(pending),
        help="Finish Approve/Cancel below first if a confirmation is pending." if pending else None,
    )

    if run:
        if not audio_bytes:
            st.error("Please record audio or upload a file first.")
        else:
            with st.spinner("Running pipeline (STT → intent → tools)…"):
                initial: AgentState = {
                    "audio_bytes": audio_bytes,
                    "audio_filename": audio_filename,
                    "require_hitl": require_hitl,
                    "human_approved": False,
                    "history": list(st.session_state.history),
                }
                try:
                    result = run_pipeline(initial)
                except Exception as e:
                    logger.exception(e)
                    st.error("Pipeline failed. Verify GROQ_API_KEY and network.")
                    return

                st.session_state.last_pipeline_result = result

                if result.get("human_pending"):
                    st.session_state.pending_hitl = {
                        "transcript": result.get("transcript", ""),
                        "intent": result.get("intent", ""),
                        "intent_details": result.get("intent_details") or {},
                        "compound": result.get("compound", False),
                        "compound_steps": result.get("compound_steps"),
                        "audio_bytes": None,
                        "audio_filename": audio_filename,
                        "error": None,
                    }
                else:
                    _append_history_from_result(result)

    if st.session_state.last_pipeline_result:
        _render_result(
            st.session_state.last_pipeline_result,
            pending_confirm=bool(st.session_state.pending_hitl),
        )
    elif not run and not pending:
        st.info("Upload or record audio, then click **Process audio**.")

    with st.expander("Output folder"):
        st.write(f"Sandbox path: `{OUTPUT_DIR}`")
        if OUTPUT_DIR.exists():
            files = sorted(OUTPUT_DIR.rglob("*"))
            paths = [str(p.relative_to(OUTPUT_DIR)) for p in files if p.is_file()]
            if paths:
                st.code("\n".join(paths[:200]), language=None)
            else:
                st.caption("(empty)")


def _append_history_from_result(result: AgentState) -> None:
    st.session_state.history = append_turn(
        st.session_state.history,
        transcript=result.get("transcript") or "",
        intent=result.get("intent") or "",
        action_taken=result.get("action_taken") or "",
        tool_result=result.get("tool_result") or "",
    )


def _render_hitl_confirmation() -> None:
    """Buttons to approve or cancel pending file operations (reads session state)."""
    pending = st.session_state.pending_hitl
    if not pending:
        return

    st.divider()
    st.markdown("#### File operation confirmation")
    st.warning(
        "Your request would **create or change files** under `output/`. "
        "Click **Approve** to run tools, or **Cancel** to dismiss."
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Approve and run tools", type="primary", key="hitl_approve"):
            state: AgentState = {
                **pending,
                "human_approved": True,
                "skip_transcribe": True,
                "skip_classify": True,
                "require_hitl": st.session_state.get("require_hitl", False),
                "history": list(st.session_state.history),
            }
            try:
                result = run_tools_only(state)
                _append_history_from_result(result)
                st.session_state.pending_hitl = None
                st.session_state.last_pipeline_result = result
                st.success("Tools executed after approval.")
                st.rerun()
            except Exception as e:
                logger.exception(e)
                st.error("Execution failed. Check logs.")
    with c2:
        if st.button("Cancel", key="hitl_cancel"):
            st.session_state.pending_hitl = None
            st.session_state.last_pipeline_result = None
            st.info("Cancelled. You can process new audio when ready.")
            st.rerun()


def _render_result(result: AgentState, *, pending_confirm: bool) -> None:
    st.subheader("Pipeline result")

    t = result.get("transcript") or ""
    st.markdown("### Transcription")
    st.write(t if t.strip() else "_(empty)_")

    st.markdown("### Intent")
    intent = result.get("intent") or "—"
    st.code(intent)

    if result.get("compound") and result.get("compound_steps"):
        st.caption("Compound steps detected")
        st.json(result.get("compound_steps"))

    st.markdown("### Action taken")
    st.info(result.get("action_taken") or "—")

    st.markdown("### Final output")
    out = result.get("tool_result") or result.get("error") or "—"
    if result.get("error"):
        st.error(out)
    else:
        st.success(out)

    if pending_confirm:
        _render_hitl_confirmation()


if __name__ == "__main__":
    # Streamlit runs app.py as script; badge may be unavailable in older Streamlit
    main()
