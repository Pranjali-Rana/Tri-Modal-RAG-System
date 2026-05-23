from __future__ import annotations

from transformers import pipeline

from .config import QA_MODEL, REFINER_MODEL, SUMMARIZER_MODEL


class ModelRegistry:
    def __init__(self) -> None:
        self._summarizer = None
        self._refiner = None
        self._qa_model = None

    @property
    def summarizer(self):
        if self._summarizer is None:
            self._summarizer = pipeline("summarization", model=SUMMARIZER_MODEL)
        return self._summarizer

    @property
    def refiner(self):
        if self._refiner is None:
            self._refiner = pipeline("text2text-generation", model=REFINER_MODEL)
        return self._refiner

    @property
    def qa_model(self):
        if self._qa_model is None:
            self._qa_model = pipeline("question-answering", model=QA_MODEL)
        return self._qa_model


_MODELS: ModelRegistry | None = None


def get_models() -> ModelRegistry:
    global _MODELS
    if _MODELS is None:
        _MODELS = ModelRegistry()
    return _MODELS
