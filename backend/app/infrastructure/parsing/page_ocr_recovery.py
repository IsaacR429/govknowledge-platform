from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
)
from app.application.ports.parser_execution import (
    ParserExecutionResult,
)
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)


class PageOcrRecovery:
    """Recover blank image-only PDF pages without replacing primary text."""

    def __init__(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        timeout_seconds: float,
        max_chunk_chars: int,
    ) -> None:
        if timeout_seconds <= 0 or max_chunk_chars < 1:
            raise ValueError("Invalid OCR recovery limits.")

        self._python = python_executable.resolve()
        self._worker = worker_script.resolve()
        self._timeout = timeout_seconds
        self._max_chunk_chars = max_chunk_chars

    @staticmethod
    def _required_pages(document: ParsedDocument) -> tuple[int, ...]:
        required = []

        for page in document.pages:
            if page.text.strip():
                continue

            native_chars = page.metadata.get("native_text_chars")
            image_count = page.metadata.get("image_count")

            if type(native_chars) is not int or type(image_count) is not int:
                raise RuntimeError(
                    "Blank page is missing integer OCR-routing metadata."
                )

            if native_chars == 0 and image_count > 0:
                required.append(page.page_number)

        return tuple(required)

    @staticmethod
    def _failure(
        primary: ParserExecutionResult,
        *,
        required_pages: tuple[int, ...],
        recovered_pages: list[int],
        failed_page: int,
        failure_class: ParserFailureClass,
        error_code: str,
        return_code: int | None = None,
    ) -> ParserExecutionResult:
        return ParserExecutionResult(
            parsing_status=ParsingStatus.FAILED,
            failure_class=failure_class,
            parsed_document=None,
            lineage={
                **primary.lineage,
                "ocr_recovery": {
                    "status": "FAILED",
                    "required_pages": list(required_pages),
                    "recovered_pages": recovered_pages.copy(),
                    "failed_page": failed_page,
                    "error_code": error_code,
                    "worker_return_code": return_code,
                },
            },
        )

    def _split_text(self, text: str) -> list[str]:
        """Split long OCR text without dropping words or exceeding the limit."""
        chunks = []
        remaining = text.strip()

        while remaining:
            if len(remaining) <= self._max_chunk_chars:
                chunks.append(remaining)
                break

            window = remaining[: self._max_chunk_chars]
            boundary = max(
                window.rfind(" "),
                window.rfind("\n"),
            )

            if boundary < self._max_chunk_chars // 2:
                boundary = self._max_chunk_chars

            chunks.append(remaining[:boundary].strip())
            remaining = remaining[boundary:].lstrip()

        return chunks

    def recover(
        self,
        file_path: Path,
        *,
        language: str,
        primary: ParserExecutionResult,
    ) -> ParserExecutionResult:
        if (
            primary.parsing_status is not ParsingStatus.SUCCESS
            or not isinstance(primary.parsed_document, ParsedDocument)
        ):
            raise ValueError("Page OCR requires a successful primary parse.")

        document = primary.parsed_document
        required_pages = self._required_pages(document)

        if not required_pages:
            return primary

        replacement_pages = {}
        added_chunks: list[ParsedChunk] = []
        recovered_pages: list[int] = []

        with tempfile.TemporaryDirectory(prefix="govknowledge-ocr-") as temp:
            for page_number in required_pages:
                result_path = Path(temp) / f"page-{page_number}.json"

                command = [
                    str(self._python),
                    str(self._worker),
                    "--pdf-path",
                    str(file_path.resolve()),
                    "--page-number",
                    str(page_number),
                    "--result-path",
                    str(result_path),
                ]

                try:
                    process = subprocess.run(
                        command,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=self._timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return self._failure(
                        primary,
                        required_pages=required_pages,
                        recovered_pages=recovered_pages,
                        failed_page=page_number,
                        failure_class=ParserFailureClass.TIMEOUT,
                        error_code="OCR_WORKER_TIMEOUT",
                    )
                except OSError:
                    return self._failure(
                        primary,
                        required_pages=required_pages,
                        recovered_pages=recovered_pages,
                        failed_page=page_number,
                        failure_class=ParserFailureClass.UNKNOWN,
                        error_code="OCR_WORKER_START_FAILED",
                    )

                if process.returncode != 0:
                    return self._failure(
                        primary,
                        required_pages=required_pages,
                        recovered_pages=recovered_pages,
                        failed_page=page_number,
                        failure_class=(
                            ParserFailureClass.NATIVE_CRASH
                            if process.returncode < 0
                            else ParserFailureClass.UNKNOWN
                        ),
                        error_code="OCR_WORKER_FAILED",
                        return_code=process.returncode,
                    )

                try:
                    payload = json.loads(
                        result_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    payload = None

                if (
                    not isinstance(payload, dict)
                    or payload.get("status") != "SUCCESS"
                    or payload.get("page_number") != page_number
                    or not isinstance(payload.get("text"), str)
                    or not payload["text"].strip()
                ):
                    return self._failure(
                        primary,
                        required_pages=required_pages,
                        recovered_pages=recovered_pages,
                        failed_page=page_number,
                        failure_class=ParserFailureClass.NORMALIZATION_FAILED,
                        error_code="OCR_RESULT_INVALID",
                        return_code=process.returncode,
                    )

                text = payload["text"].strip()
                original_page = next(
                    page
                    for page in document.pages
                    if page.page_number == page_number
                )

                metadata = {
                    **original_page.metadata,
                    "extraction_method": "rapidocr_page_recovery",
                    "ocr_requested": True,
                    "ocr_backend": "onnxruntime",
                    "ocr_model_family": "PP-OCRv6",
                    "ocr_text_chars": len(text),
                    "ocr_quality_verified": False,
                }

                replacement_pages[page_number] = replace(
                    original_page,
                    text=text,
                    metadata=metadata,
                )

                for chunk_text in self._split_text(text):
                    added_chunks.append(
                        ParsedChunk(
                            page_number=page_number,
                            chunk_index=0,  # Reassigned below.
                            content=chunk_text,
                            section_title=None,
                            language=language,
                            metadata=metadata.copy(),
                        )
                    )

                recovered_pages.append(page_number)

        # Keep all primary pages and their text. Replace only OCR-needed pages.
        pages = tuple(
            replacement_pages.get(page.page_number, page)
            for page in document.pages
        )

        # Keep the primary chunks' content and metadata. Reorder by PDF page
        # and assign contiguous chunk indexes after inserting OCR chunks.
        ordered_chunks = sorted(
            [*document.chunks, *added_chunks],
            key=lambda chunk: (chunk.page_number, chunk.chunk_index),
        )

        chunks = tuple(
            replace(chunk, chunk_index=index)
            for index, chunk in enumerate(ordered_chunks)
        )

        return ParserExecutionResult(
            parsing_status=ParsingStatus.SUCCESS,
            failure_class=None,
            parsed_document=ParsedDocument(
                pages=pages,
                chunks=chunks,
            ),
            lineage={
                **primary.lineage,
                "ocr_recovery": {
                    "status": "SUCCEEDED",
                    "required_pages": list(required_pages),
                    "recovered_pages": recovered_pages,
                    "engine": "rapidocr",
                    "backend": "onnxruntime",
                    "scope": "blank_image_only_pages",
                    "ocr_quality_verified": False,
                },
            },
        )
