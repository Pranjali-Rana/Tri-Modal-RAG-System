from __future__ import annotations

from .config import INITIAL_SUMMARY_PROMPT, LENGTH_SETTINGS, SummaryLength
from .models import get_models
from .parsing import strip_reference_artifacts
from .text_splitter import build_text_splitter
from .utils import clean_generated_text, dedupe_blocks


class SummaryPipeline:
    def __init__(self) -> None:
        self.models = get_models()
        self.text_splitter = build_text_splitter()

    def summarize_text(self, text: str) -> str:
        chunks = self._chunk_text(text)
        partial_summaries = [
            self._summarize_chunk(
                chunk,
                ratio=1.0,
                max_new_tokens_cap=700,
            )
            for chunk in chunks
        ]
        partial_summaries = dedupe_blocks(partial_summaries)

        if len(partial_summaries) == 1:
            return partial_summaries[0]
        return "\n\n".join(partial_summaries)

    def refine_summary(self, summary: str, summary_length: SummaryLength) -> str:
        settings = LENGTH_SETTINGS[summary_length]
        min_new_tokens, max_new_tokens = self._adaptive_generation_bounds(summary, summary_length)
        prompt = (
            f"{settings['instruction']}\n"
            "Stay faithful to the source summary.\n"
            "Do not remove important dates, titles, or victories unless the requested level of detail clearly requires compression.\n\n"
            f"Summary:\n{summary}"
        )
        result = self.models.refiner(
            prompt,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        rewritten = result[0]["generated_text"].strip()
        return clean_generated_text(rewritten or summary)

    def generate_summary_bundle(self, input_text: str, summary_length: SummaryLength) -> dict[str, str]:
        initial_summary = self.summarize_text(input_text)
        return self.refine_summary_bundle(initial_summary, summary_length, source_text=input_text)

    def refine_summary_bundle(
        self,
        initial_summary: str,
        summary_length: SummaryLength,
        *,
        source_text: str | None = None,
    ) -> dict[str, str]:
        final_summary = self.refine_summary(initial_summary, summary_length)
        qa_context = strip_reference_artifacts(f"{initial_summary}\n\n{final_summary}")
        return {
            "source_text": source_text or "",
            "summary_length": summary_length,
            "initial_summary": initial_summary,
            "final_summary": final_summary,
            "qa_context": qa_context,
        }

    def _summarize_chunk(
        self,
        text: str,
        *,
        ratio: float,
        max_new_tokens_cap: int,
    ) -> str:
        min_length, max_length = self._adaptive_initial_summary_bounds(
            text,
            ratio=ratio,
            max_new_tokens_cap=max_new_tokens_cap,
        )
        result = self.models.summarizer(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        return clean_generated_text(result[0]["summary_text"].strip())

    def _chunk_text(self, text: str) -> list[str]:
        if not text.strip():
            return [""]
        return self.text_splitter.split_text(text)

    def _adaptive_generation_bounds(self, summary: str, summary_length: SummaryLength) -> tuple[int, int]:
        settings = LENGTH_SETTINGS[summary_length]
        summary_tokens = len(self.models.refiner.tokenizer(summary, add_special_tokens=False)["input_ids"])
        target_ratio = float(settings["target_ratio"])
        min_ratio = float(settings["min_ratio"])
        max_new_tokens_floor = int(settings["max_new_tokens_floor"])
        max_new_tokens_cap = int(settings["max_new_tokens_cap"])
        min_new_tokens_floor = int(settings["min_new_tokens_floor"])

        scaled_tokens = int(summary_tokens * target_ratio)
        max_new_tokens = max(max_new_tokens_floor, min(scaled_tokens, max_new_tokens_cap))

        adaptive_min = max(min_new_tokens_floor, int(max_new_tokens * min_ratio))
        adaptive_min = min(adaptive_min, max_new_tokens - 12) if max_new_tokens > 24 else adaptive_min
        return adaptive_min, max_new_tokens

    def _adaptive_initial_summary_bounds(
        self,
        text: str,
        *,
        ratio: float,
        max_new_tokens_cap: int,
    ) -> tuple[int, int]:
        input_tokens = len(self.models.summarizer.tokenizer(text, add_special_tokens=False)["input_ids"])
        scaled_tokens = int(input_tokens * ratio)
        max_new_tokens = max(160, min(scaled_tokens, max_new_tokens_cap))
        min_new_tokens = max(80, int(max_new_tokens * 0.3))
        min_new_tokens = min(min_new_tokens, max_new_tokens - 12) if max_new_tokens > 24 else min_new_tokens
        return min_new_tokens, max_new_tokens


_PIPELINE: SummaryPipeline | None = None


def get_pipeline() -> SummaryPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = SummaryPipeline()
    return _PIPELINE
