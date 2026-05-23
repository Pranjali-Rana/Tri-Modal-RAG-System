from __future__ import annotations

import re

from .parsing import strip_reference_artifacts


def clean_generated_text(text: str) -> str:
    cleaned = strip_reference_artifacts(text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = collapse_repeated_clauses(cleaned)
    cleaned = dedupe_sentences(cleaned)
    return cleaned.strip()


def dedupe_blocks(blocks: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()

    for block in blocks:
        cleaned_block = clean_generated_text(block)
        key = normalize_for_compare(cleaned_block)
        if not key or key in seen:
            continue
        if any(key in existing or existing in key for existing in seen if len(key) > 40):
            continue
        deduped.append(cleaned_block)
        seen.add(key)

    return deduped or blocks


def dedupe_sentences(text: str) -> str:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    cleaned_paragraphs: list[str] = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        unique_sentences: list[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            normalized = normalize_for_compare(sentence)
            if not normalized or normalized in seen:
                continue
            unique_sentences.append(sentence.strip())
            seen.add(normalized)
        if unique_sentences:
            cleaned_paragraphs.append(" ".join(unique_sentences))

    return "\n\n".join(cleaned_paragraphs) if cleaned_paragraphs else text


def collapse_repeated_clauses(text: str) -> str:
    def collapse_paragraph(paragraph: str) -> str:
        parts = [part.strip() for part in paragraph.split(",")]
        if len(parts) < 4:
            return paragraph

        collapsed: list[str] = []
        seen_counts: dict[str, int] = {}
        for part in parts:
            normalized = normalize_for_compare(part)
            if not normalized:
                continue
            seen_counts[normalized] = seen_counts.get(normalized, 0) + 1
            if seen_counts[normalized] == 1:
                collapsed.append(part)

        if len(collapsed) >= max(2, len(parts) // 2):
            return ", ".join(collapsed)
        return paragraph

    paragraphs = [collapse_paragraph(paragraph.strip()) for paragraph in text.split("\n\n")]
    return "\n\n".join(paragraphs)


def normalize_for_compare(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def sentence_windows(text: str, window_size: int = 3, stride: int = 2) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    if len(sentences) <= window_size:
        return [" ".join(sentences)] if sentences else [text]

    windows: list[str] = []
    for start in range(0, len(sentences), stride):
        window = sentences[start : start + window_size]
        if not window:
            continue
        windows.append(" ".join(window))
        if start + window_size >= len(sentences):
            break
    return windows or [text]


def keyword_overlap_score(question: str, context: str) -> float:
    question_tokens = set(tokenize_for_match(question))
    context_tokens = set(tokenize_for_match(context))
    if not question_tokens or not context_tokens:
        return 0.0

    overlap = len(question_tokens & context_tokens)
    score = overlap / max(len(question_tokens), 1)

    years_in_question = {token for token in question_tokens if token.isdigit() and len(token) == 4}
    if years_in_question:
        matching_years = years_in_question & context_tokens
        score += 0.25 * len(matching_years)

    return score


def tokenize_for_match(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())
