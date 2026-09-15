from __future__ import annotations

import hashlib
from pathlib import Path

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.application.ports.parser_execution import (
    ParserExecutionResult,
    ParserExecutor,
)
from app.domain.enums.processing import (
    ParsingStatus,
)
from app.infrastructure.parsing.parser_worker import (
    DocumentWorkerRunResult,
    run_external_document_worker,
)


def _sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


class PyMuPDFParserExecutor(ParserExecutor):
    """
    Execute the PyMuPDF4LLM parser in an isolated external
    Python runtime.

    The parent GovKnowledge runtime does not import PyMuPDF.
    """

    def __init__(
        self,
        *,
        python_executable: Path,
        worker_script: Path,
        timeout_seconds: float,
        max_chunk_chars: int = 1500,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        if max_chunk_chars < 1:
            raise ValueError(
                "max_chunk_chars must be positive."
            )

        self._python_executable = (
            python_executable.resolve()
        )

        self._worker_script = (
            worker_script.resolve()
        )

        self._timeout_seconds = (
            timeout_seconds
        )

        self._max_chunk_chars = (
            max_chunk_chars
        )

    @staticmethod
    def _parser_config(
        result: DocumentWorkerRunResult,
        *,
        max_chunk_chars: int,
    ) -> dict[str, object]:
        config: dict[str, object] = {
            "max_chunk_chars":
                max_chunk_chars,
        }

        parsed = result.parsed_document

        if (
            isinstance(
                parsed,
                ParsedDocument,
            )
            and parsed.pages
        ):
            metadata = (
                parsed.pages[0].metadata
            )

            for key in (
                "pymupdf4llm_version",
                "pymupdf_version",
                "chunking_strategy",
                "ocr_requested",
            ):
                if key in metadata:
                    config[key] = (
                        metadata[key]
                    )

        return config

    def _lineage(
        self,
        *,
        result: DocumentWorkerRunResult,
        original_sha256: str,
    ) -> dict[str, object]:
        return {
            "execution_mode":
                "ISOLATED_WORKER",

            "parser_family":
                "pymupdf4llm",

            "timeout_seconds":
                self._timeout_seconds,

            "original_sha256":
                original_sha256,

            "return_code":
                result.return_code,

            "exception_type":
                result.exception_type,

            "parser_config":
                self._parser_config(
                    result,
                    max_chunk_chars=(
                        self._max_chunk_chars
                    ),
                ),

            "metrics": {
                "worker_spawn_ms":
                    result.worker_spawn_ms,

                "worker_setup_ms":
                    result.worker_setup_ms,

                "parser_execution_ms":
                    result.parser_execution_ms,

                "worker_teardown_ms":
                    result.worker_teardown_ms,

                "end_to_end_document_processing_ms":
                    (
                        result
                        .end_to_end_document_processing_ms
                    ),

                "worker_peak_rss_bytes":
                    result.worker_peak_rss_bytes,
            },
        }

    def execute(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParserExecutionResult:
        source = file_path.resolve()

        if not source.is_file():
            raise FileNotFoundError(
                source
            )

        if not language:
            raise ValueError(
                "language cannot be empty."
            )

        original_sha256 = (
            _sha256_file(
                source
            )
        )

        worker_result = (
            run_external_document_worker(
                python_executable=(
                    self._python_executable
                ),
                worker_script=(
                    self._worker_script
                ),
                file_path=source,
                language=language,
                timeout_seconds=(
                    self._timeout_seconds
                ),
                max_chunk_chars=(
                    self._max_chunk_chars
                ),
            )
        )

        if (
            worker_result.parsing_status
            is ParsingStatus.SUCCESS
        ):
            if (
                not isinstance(
                    worker_result.parsed_document,
                    ParsedDocument,
                )
            ):
                raise RuntimeError(
                    "Successful PyMuPDF worker "
                    "returned no ParsedDocument."
                )

            if (
                worker_result.failure_class
                is not None
            ):
                raise RuntimeError(
                    "Successful PyMuPDF worker "
                    "returned a failure class."
                )

        else:
            if (
                worker_result.parsed_document
                is not None
            ):
                raise RuntimeError(
                    "Failed PyMuPDF worker "
                    "returned a ParsedDocument."
                )

        return ParserExecutionResult(
            parsing_status=(
                worker_result.parsing_status
            ),
            failure_class=(
                worker_result.failure_class
            ),
            parsed_document=(
                worker_result.parsed_document
                if isinstance(
                    worker_result.parsed_document,
                    ParsedDocument,
                )
                else None
            ),
            lineage=self._lineage(
                result=worker_result,
                original_sha256=(
                    original_sha256
                ),
            ),
        )