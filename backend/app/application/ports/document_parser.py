from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.domain.tables.evidence import (
    NormalizedTableEvidence,
)


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str
    metadata: dict


@dataclass(frozen=True)
class ParsedChunk:
    page_number: int
    chunk_index: int
    content: str
    section_title: str | None
    language: str
    metadata: dict


@dataclass(frozen=True)
class ParsedDocument:
    pages: tuple[ParsedPage, ...]
    chunks: tuple[ParsedChunk, ...]

    table_evidence: tuple[
        NormalizedTableEvidence,
        ...
    ] = ()


class DocumentParser(Protocol):
    def parse(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParsedDocument:
        ...
