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
from app.infrastructure.parsing.resilient_parser_worker import (
    run_resilient_document_worker,
)


class ResilientDoclingParserExecutor(
    ParserExecutor
):
    """
    GovKnowledge production-facing Docling execution adapter.

    Docling itself remains behind the existing DocumentParser
    adapter inside the isolated worker.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        artifacts_path: Path | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        self._timeout_seconds = (
            timeout_seconds
        )
        self._artifacts_path = (
            artifacts_path
        )

    def execute(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParserExecutionResult:
        execution = (
            run_resilient_document_worker(
                file_path=file_path,
                language=language,
                timeout_seconds=(
                    self._timeout_seconds
                ),
                artifacts_path=(
                    self._artifacts_path
                ),
            )
        )

        worker = execution.result
        parsed = worker.parsed_document

        if (
            worker.parsing_status
            is ParsingStatus.SUCCESS
            and not isinstance(
                parsed,
                ParsedDocument,
            )
        ):
            raise RuntimeError(
                "Successful parser worker returned "
                "no ParsedDocument."
            )

        if (
            worker.parsing_status
            is ParsingStatus.FAILED
            and parsed is not None
        ):
            raise RuntimeError(
                "Failed parser worker unexpectedly "
                "returned a ParsedDocument."
            )

        parser_configuration: dict[
            str,
            object,
        ] = {}

        if (
            isinstance(
                parsed,
                ParsedDocument,
            )
            and parsed.pages
        ):
            metadata = parsed.pages[0].metadata

            for key in (
                "parser",
                "docling_version",
                "docling_core_version",
                "docling_ibm_models_version",
                "docling_parse_version",
                "rapidocr_version",
                "onnxruntime_version",
                "max_chunk_chars",
                "artifact_bundle_id",
                "artifact_bundle_revision",
                "ocr_model_family",
                "chunking_strategy",
            ):
                value = metadata.get(key)

                if value is not None:
                    parser_configuration[
                        key
                    ] = value

        lineage: dict[str, object] = {
            "execution_mode": (
                "ISOLATED_WORKER"
            ),
            "parser_family": "docling",
            "timeout_seconds": (
                self._timeout_seconds
            ),
            "original_sha256": (
                execution.original_sha256
            ),
            "derived_sha256": (
                execution.derived_sha256
            ),
            "normalization_attempted": (
                execution.normalization_attempted
            ),
            "normalization_attempt_count": (
                execution
                .normalization_attempt_count
            ),
            "initial_failure_class": (
                execution
                .initial_failure_class.value
                if execution.initial_failure_class
                is not None
                else None
            ),
            "retry_status": (
                execution.retry_status.value
                if execution.retry_status
                is not None
                else None
            ),
            "source_page_count": (
                execution.source_page_count
            ),
            "derived_page_count": (
                execution.derived_page_count
            ),
            "return_code": (
                worker.return_code
            ),
            "exception_type": (
                worker.exception_type
            ),
            "metrics": {
                "worker_spawn_ms": (
                    worker.worker_spawn_ms
                ),
                "worker_setup_ms": (
                    worker.worker_setup_ms
                ),
                "parser_execution_ms": (
                    worker.parser_execution_ms
                ),
                "worker_teardown_ms": (
                    worker.worker_teardown_ms
                ),
                "end_to_end_document_processing_ms": (
                    execution
                    .end_to_end_document_processing_ms
                ),
                "worker_peak_rss_bytes": (
                    worker.worker_peak_rss_bytes
                ),
            },
        }

        if parser_configuration:
            lineage[
                "parser_configuration"
            ] = parser_configuration

        return ParserExecutionResult(
            parsing_status=(
                worker.parsing_status
            ),
            failure_class=(
                worker.failure_class
            ),
            parsed_document=(
                parsed
                if isinstance(
                    parsed,
                    ParsedDocument,
                )
                else None
            ),
            lineage=lineage,
        )
