from __future__ import annotations

from typing import Literal


SummaryLength = Literal["short", "medium", "long"]

SUMMARIZER_MODEL = "facebook/bart-large-cnn"
REFINER_MODEL = "google/flan-t5-base"
QA_MODEL = "distilbert-base-cased-distilled-squad"

INITIAL_SUMMARY_PROMPT = (
    "You are a careful summarization assistant.\n"
    "Create a factual and detail-preserving intermediate summary of the source text.\n"
    "Preserve key facts, names, roles, dates, achievements, records, awards, and major events.\n"
    "Retain enough supporting detail so the summary can later be compressed into short, medium, or long forms.\n"
    "Use complete sentences and keep the summary information-rich.\n"
    "Write a multi-sentence intermediate summary, not a brief headline.\n"
    "When the source contains many important facts, include broad coverage of them.\n"
    "Prefer preserving useful details over aggressive compression.\n"
    "Write only the summary.\n\n"
    "Source text:\n{source_text}"
)

LENGTH_SETTINGS: dict[SummaryLength, dict[str, int | str]] = {
    "short": {
        "instruction": (
            "You are a careful summarization assistant.\n"
            "Rewrite the input as a concise overview for a busy reader.\n"
            "Keep only the most important identity, role, and headline achievements.\n"
            "Prefer a compact summary with minimal supporting detail.\n"
            "Preserve at most one especially important date if it is central to the topic.\n"
            "Do not invent facts. Write only the rewritten summary."
        ),
        "target_ratio": 0.35,
        "min_new_tokens_floor": 72,
        "min_ratio": 0.45,
        "max_new_tokens_floor": 140,
        "max_new_tokens_cap": 260,
    },
    "medium": {
        "instruction": (
            "You are a careful summarization assistant.\n"
            "Rewrite the input as a balanced summary for a general reader.\n"
            "Keep the main topic, key achievements, important dates, and notable milestones.\n"
            "Include enough detail to answer common follow-up questions without becoming too long.\n"
            "Do not invent facts. Write only the rewritten summary."
        ),
        "target_ratio": 0.6,
        "min_new_tokens_floor": 120,
        "min_ratio": 0.48,
        "max_new_tokens_floor": 220,
        "max_new_tokens_cap": 520,
    },
    "long": {
        "instruction": (
            "You are a careful summarization assistant.\n"
            "Rewrite the input as a detailed summary for a reader who wants context.\n"
            "Preserve important names, dates, achievements, awards, milestones, and major events.\n"
            "Retain enough supporting detail so that downstream question answering remains reliable.\n"
            "Do not invent facts. Write only the rewritten summary."
        ),
        "target_ratio": 0.9,
        "min_new_tokens_floor": 180,
        "min_ratio": 0.5,
        "max_new_tokens_floor": 320,
        "max_new_tokens_cap": 900,
    },
}
