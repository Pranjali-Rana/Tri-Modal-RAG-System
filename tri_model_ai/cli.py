from __future__ import annotations

from typing import cast

from .assistant import build_graph
from .config import LENGTH_SETTINGS, SummaryLength
from .parsing import extract_text_from_path, normalize_source_text
from .state import AssistantState


def read_input_text() -> str:
    print("Tri-Model AI Assistant")
    print("=" * 60)
    choice = input("Load text from a file? (y/n): ").strip().lower()
    if choice == "y":
        file_path = input("Enter the file path: ").strip()
        return extract_text_from_path(file_path)

    print("\nPaste your large text below. Finish with a blank line.")
    lines: list[str] = []
    while True:
        line = input()
        if not line and lines:
            break
        lines.append(line)
    return normalize_source_text("\n".join(lines))


def read_summary_length() -> SummaryLength:
    while True:
        value = input("Choose summary length (short / medium / long): ").strip().lower()
        if value in LENGTH_SETTINGS:
            return cast(SummaryLength, value)
        print("Please choose: short, medium, or long.")


def main() -> None:
    input_text = read_input_text()
    if not input_text:
        raise ValueError("Input text cannot be empty.")

    summary_length = read_summary_length()
    app = build_graph()
    final_state = app.invoke(
        cast(
            AssistantState,
            {
                "input_text": input_text,
                "summary_length": summary_length,
                "initial_summary": "",
                "final_summary": "",
                "qa_context": "",
                "question": "",
                "answer": "",
                "qa_history": [],
            },
        )
    )

    if final_state["qa_history"]:
        print("\nSession transcript")
        print("-" * 60)
        print("\n\n".join(final_state["qa_history"]))
