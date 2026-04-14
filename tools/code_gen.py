"""Generate code with Groq LLM and write to output/."""

from __future__ import annotations

import logging
from typing import Any

from config import OUTPUT_DIR
from services.llm import chat_completion
from utils.sanitize import safe_path

logger = logging.getLogger(__name__)

CODE_GEN_PROMPT = """You are an expert programmer. Generate clean, production-ready code based on the user's request.
Request: {description}
Language: {language}
Return ONLY the code, no markdown fences, no explanations before or after."""


def execute_write_code(intent_details: dict[str, Any], transcript: str) -> dict[str, str]:
    """
    Generate code from description and save to a file under output/.
    """
    language = (intent_details.get("language") or "python").strip().lower()
    description = intent_details.get("description") or transcript
    filename = intent_details.get("filename")

    if not filename:
        ext = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "rust": "rs",
            "go": "go",
            "java": "java",
            "c": "c",
            "cpp": "cpp",
            "csharp": "cs",
            "ruby": "rb",
            "shell": "sh",
            "bash": "sh",
        }.get(language, "txt")
        filename = f"generated.{ext}"

    try:
        prompt = CODE_GEN_PROMPT.format(description=description, language=language)
        messages = [
            {"role": "system", "content": "You write only code, no markdown."},
            {"role": "user", "content": prompt},
        ]
        code = chat_completion(messages, temperature=0.2)
        # Strip accidental fences
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        target = safe_path(filename, base_dir=OUTPUT_DIR)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")

        rel = target.relative_to(OUTPUT_DIR)
        return {
            "action_taken": f"Generated {language} code and saved to '{rel}'",
            "tool_result": f"Saved {len(code)} characters to output/{rel}",
        }
    except ValueError as e:
        logger.warning("Write code path error: %s", e)
        return {
            "action_taken": "Rejected unsafe file path",
            "tool_result": "Could not save code: invalid filename.",
        }
    except OSError as e:
        logger.exception("Write failed: %s", e)
        return {
            "action_taken": "Failed to write file",
            "tool_result": "Could not write the file to disk.",
        }
    except RuntimeError as e:
        logger.exception("LLM failed: %s", e)
        return {
            "action_taken": "Code generation failed",
            "tool_result": "The model could not generate code. Please try again.",
        }
