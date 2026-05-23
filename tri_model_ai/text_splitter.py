from __future__ import annotations

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
            def __init__(
                self,
                *,
                chunk_size: int,
                chunk_overlap: int,
                separators: list[str] | None = None,
            ) -> None:
                self.chunk_size = max(1, chunk_size)
                self.chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
                self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

            def split_text(self, text: str) -> list[str]:
                if not text.strip():
                    return [""]

                chunks: list[str] = []
                start = 0
                text_length = len(text)

                while start < text_length:
                    max_end = min(start + self.chunk_size, text_length)
                    end = max_end

                    if max_end < text_length:
                        window = text[start:max_end]
                        for separator in self.separators:
                            if not separator:
                                continue
                            split_at = window.rfind(separator)
                            if split_at > max(len(separator), self.chunk_size // 3):
                                end = start + split_at + len(separator)
                                break

                    chunk = text[start:end].strip()
                    if chunk:
                        chunks.append(chunk)

                    if end >= text_length:
                        break

                    start = max(end - self.chunk_overlap, start + 1)


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
