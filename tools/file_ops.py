"""Create files and folders under the sandbox output directory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import OUTPUT_DIR
from utils.sanitize import resolve_output_subdir, safe_path

logger = logging.getLogger(__name__)


def execute_create_file(intent_details: dict[str, Any], transcript: str) -> dict[str, str]:
    """
    Create an empty file or folder under output/.

    Returns:
        action_taken: human-readable description
        tool_result: success message or error (user-safe)
    """
    filename = intent_details.get("filename") or "note.txt"
    folder = intent_details.get("folder")
    is_folder = bool(intent_details.get("is_folder", False))

    try:
        if folder:
            sub = resolve_output_subdir(str(folder), OUTPUT_DIR)
            target = safe_path(filename, base_dir=sub)
        else:
            target = safe_path(filename, base_dir=OUTPUT_DIR)

        if is_folder:
            target.mkdir(parents=True, exist_ok=True)
            action = f"Created folder '{target.relative_to(OUTPUT_DIR)}' under output/"
            return {"action_taken": action, "tool_result": f"Folder ready: {target}"}

        body = intent_details.get("content")
        body_str = str(body).strip() if body is not None else ""

        target.parent.mkdir(parents=True, exist_ok=True)
        if body_str:
            target.write_text(body_str, encoding="utf-8")
            action = f"Wrote content to '{target.relative_to(OUTPUT_DIR)}'"
        elif not target.exists():
            target.write_text("", encoding="utf-8")
            action = f"Created empty file '{target.relative_to(OUTPUT_DIR)}'"
        else:
            action = f"File already existed (left unchanged): '{target.relative_to(OUTPUT_DIR)}'"

        return {
            "action_taken": action,
            "tool_result": f"Path: {target.relative_to(OUTPUT_DIR)}",
        }
    except ValueError as e:
        logger.warning("Create file rejected: %s", e)
        return {
            "action_taken": "Rejected unsafe path",
            "tool_result": "Could not create file: invalid or unsafe path.",
        }
    except OSError as e:
        logger.exception("Filesystem error: %s", e)
        return {
            "action_taken": "File operation failed",
            "tool_result": "Could not create the file or folder. Please try a different name.",
        }
