"""Summarize text using Groq LLM."""

from __future__ import annotations

import logging
from typing import Any

from services.llm import chat_completion

logger = logging.getLogger(__name__)

SUMMARIZE_PROMPT = """Summarize the following text concisely, preserving key points.

Text:
{content}
"""

# When there is no long passage—only a topic like "article on AI"—generate instead of "summarize empty text"
GENERATE_ARTICLE_PROMPT = """Write a short, informative article in Markdown.
Use a # title and 2–4 sections with ## subheadings. Be substantive (several paragraphs).
Do not apologize or say the topic is empty.

Topic or request:
{topic}
"""


def _is_substantial_passage(text: str) -> bool:
    """Heuristic: enough words to be a real passage to summarize (not just a title)."""
    t = text.strip()
    if len(t) < 120:
        return False
    words = t.split()
    return len(words) >= 35


def execute_summarize(intent_details: dict[str, Any], transcript: str) -> dict[str, str]:
    content = intent_details.get("content") or intent_details.get("description") or transcript
    if not content.strip():
        return {
            "action_taken": "Nothing to summarize",
            "tool_result": "No text was provided to summarize. Please include text in your request.",
        }

    try:
        if _is_substantial_passage(content):
            prompt = SUMMARIZE_PROMPT.format(content=content)
            messages = [
                {"role": "system", "content": "You produce clear, accurate summaries."},
                {"role": "user", "content": prompt},
            ]
            summary = chat_completion(messages, temperature=0.3)
            return {
                "action_taken": "Summarized the provided text",
                "tool_result": summary,
            }

        topic = content.strip()
        prompt = GENERATE_ARTICLE_PROMPT.format(topic=topic)
        messages = [
            {
                "role": "system",
                "content": "You write clear Markdown articles from a short topic or title.",
            },
            {"role": "user", "content": prompt},
        ]
        article = chat_completion(messages, temperature=0.5, max_tokens=2048)
        return {
            "action_taken": "Generated article from topic (no long source text was given)",
            "tool_result": article,
        }
    except RuntimeError as e:
        logger.exception("Summarization failed: %s", e)
        return {
            "action_taken": "Summarization failed",
            "tool_result": "Could not summarize. Please try again.",
        }
