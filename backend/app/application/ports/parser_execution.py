from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)


@dataclass(frozen=True)
class ParserExecutionResult:
    parsing_status: ParsingStatus
    failure_class: ParserFailureClass | None
    parsed_document: ParsedDocument | None
    lineage: dict[str, object]


class ParserExecutor(Protocol):
    def execute(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParserExecutionResult:
        ...
