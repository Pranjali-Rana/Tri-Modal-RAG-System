from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


def strip_reference_artifacts(text: str) -> str:
    cleaned = re.sub(r"(?:\[\s*\d+\s*\]\s*)+", " ", text)
    cleaned = re.sub(
        r"\[\s*(?:citation|clarification|verification|dubious|needed|who\?|when\?|where\?)[^\]]*\]",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\[(?=\s|$)", "", cleaned)
    cleaned = re.sub(r"\](?=\s|$)", "", cleaned)
    return cleaned


def normalize_source_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = strip_reference_artifacts(normalized)
    lines = [line.rstrip() for line in normalized.split("\n")]
    compact_lines: list[str] = []
    blank_run = 0
    for line in lines:
        if line.strip():
            compact_lines.append(line)
            blank_run = 0
        else:
            blank_run += 1
            if blank_run <= 1:
                compact_lines.append("")
    cleaned = "\n".join(compact_lines).strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *([,.;:!?])", r"\1", cleaned)
    return cleaned


def extract_text_from_path(path: str) -> str:
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".log"}:
        return normalize_source_text(file_path.read_text(encoding="utf-8"))
    if suffix == ".pdf":
        return normalize_source_text(_extract_pdf_text(file_path.read_bytes()))
    if suffix == ".docx":
        return normalize_source_text(_extract_docx_text(file_path.read_bytes()))
    return normalize_source_text(file_path.read_text(encoding="utf-8"))


def extract_text_from_uploaded_file(filename: str, data: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".log"}:
        return normalize_source_text(data.decode("utf-8"))
    if suffix == ".pdf":
        return normalize_source_text(_extract_pdf_text(data))
    if suffix == ".docx":
        return normalize_source_text(_extract_docx_text(data))
    return normalize_source_text(data.decode("utf-8"))


def _extract_pdf_text(data: bytes) -> str:
    if PdfReader is None:
        raise ImportError("PDF parsing requires `pypdf`. Install dependencies from requirements.txt.")
    reader = PdfReader(BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(page.strip() for page in pages if page.strip())
    if not text.strip():
        raise ValueError("No readable text was found in the PDF.")
    return text


def _extract_docx_text(data: bytes) -> str:
    if Document is None:
        raise ImportError("DOCX parsing requires `python-docx`. Install dependencies from requirements.txt.")
    document = Document(BytesIO(data))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    text = "\n\n".join(paragraphs)
    if not text.strip():
        raise ValueError("No readable text was found in the DOCX file.")
    return text
