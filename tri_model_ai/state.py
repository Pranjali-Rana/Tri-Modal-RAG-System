from __future__ import annotations

import operator

from typing_extensions import Annotated, TypedDict

from .config import SummaryLength


class AssistantState(TypedDict):
    input_text: str
    summary_length: SummaryLength
    initial_summary: str
    final_summary: str
    qa_context: str
    question: str
    answer: str
    qa_history: Annotated[list[str], operator.add]
