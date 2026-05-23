from __future__ import annotations

from .models import get_models
from .utils import clean_generated_text, keyword_overlap_score, sentence_windows


class QAService:
    def __init__(self) -> None:
        self.models = get_models()

    def answer_question(self, question: str, summary: str) -> str:
        candidate_contexts = self._select_contexts(question, summary)
        best_answer = ""
        best_score = 0.0

        for context in candidate_contexts:
            result = self.models.qa_model(question=question, context=context)
            answer = clean_generated_text(result.get("answer", "").strip())
            score = float(result.get("score", 0.0))
            if not answer:
                continue
            adjusted_score = score + keyword_overlap_score(question, context)
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_answer = answer

        if not best_answer or best_score < 0.08:
            return "I could not find a reliable answer in the summary."
        return best_answer

    def _select_contexts(self, question: str, summary: str) -> list[str]:
        paragraphs = [paragraph.strip() for paragraph in summary.split("\n\n") if paragraph.strip()]
        if not paragraphs:
            return [summary]

        if len(paragraphs) == 1:
            paragraphs = sentence_windows(paragraphs[0])

        scored_contexts = [
            (keyword_overlap_score(question, passage), passage)
            for passage in paragraphs
            if passage.strip()
        ]
        scored_contexts.sort(key=lambda item: item[0], reverse=True)

        top_passages = [passage for _, passage in scored_contexts[:4] if passage.strip()]
        if not top_passages:
            return [summary]
        return top_passages


_QA_SERVICE: QAService | None = None


def get_qa_service() -> QAService:
    global _QA_SERVICE
    if _QA_SERVICE is None:
        _QA_SERVICE = QAService()
    return _QA_SERVICE
