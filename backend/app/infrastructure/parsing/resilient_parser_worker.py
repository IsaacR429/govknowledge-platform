from __future__ import annotations

import hashlib
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from app.infrastructure.parsing.pdf_structural_preflight import (
    preflight_pdf,
)

from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)
from app.infrastructure.parsing.parser_worker import (
    DocumentWorkerRunResult,
    run_document_worker,
)
from app.infrastructure.parsing.pdf_normalizer import (
    PdfNormalizationError,
    PostNormalizationMismatchError,
    normalize_pdf_copy,
)


@dataclass(frozen=True)
class ResilientDocumentWorkerResult:
    result: DocumentWorkerRunResult

    original_sha256: str
    derived_sha256: str | None

    normalization_attempted: bool
    normalization_attempt_count: int

    initial_failure_class: (
        ParserFailureClass | None
    )
    retry_status: ParsingStatus | None

    source_page_count: int
    derived_page_count: int | None

    end_to_end_document_processing_ms: float


def _sha256_file(path: Path) -> str:
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


def _independent_page_count(
    path: Path,
) -> int:
    return preflight_pdf(
        path
    ).page_count


def _terminal_normalization_failure(
    *,
    initial: DocumentWorkerRunResult,
    failure_class: ParserFailureClass,
    exc: BaseException,
    elapsed_ms: float,
) -> DocumentWorkerRunResult:
    return DocumentWorkerRunResult(
        parsing_status=ParsingStatus.FAILED,
        failure_class=failure_class,
        parsed_document=None,
        return_code=initial.return_code,
        worker_spawn_ms=(
            initial.worker_spawn_ms
        ),
        worker_setup_ms=(
            initial.worker_setup_ms
        ),
        parser_execution_ms=(
            initial.parser_execution_ms
        ),
        worker_teardown_ms=(
            initial.worker_teardown_ms
        ),
        end_to_end_document_processing_ms=(
            elapsed_ms
        ),
        worker_peak_rss_bytes=(
            initial.worker_peak_rss_bytes
        ),
        exception_type=(
            type(exc).__name__
        ),
        exception_message=str(exc)[:2000],
    )


def run_resilient_document_worker(
    *,
    file_path: Path,
    language: str,
    timeout_seconds: float,
    artifacts_path: Path | None = None,
) -> ResilientDocumentWorkerResult:
    """Parse once, with at most one eligible normalization retry."""

    started = time.perf_counter()

    source = file_path.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    original_sha256 = _sha256_file(
        source
    )

    source_page_count = (
        _independent_page_count(
            source
        )
    )

    initial = run_document_worker(
        file_path=source,
        language=language,
        timeout_seconds=timeout_seconds,
        artifacts_path=artifacts_path,
    )

    initial_failure = (
        initial.failure_class
    )

    if (
        initial.parsing_status
        is ParsingStatus.SUCCESS
        or initial.failure_class
        is not ParserFailureClass.UNRESOLVED_PAGE_GEOMETRY
    ):
        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        return ResilientDocumentWorkerResult(
            result=initial,
            original_sha256=(
                original_sha256
            ),
            derived_sha256=None,
            normalization_attempted=False,
            normalization_attempt_count=0,
            initial_failure_class=(
                initial_failure
            ),
            retry_status=None,
            source_page_count=(
                source_page_count
            ),
            derived_page_count=None,
            end_to_end_document_processing_ms=(
                elapsed_ms
            ),
        )

    # Exactly one normalization attempt.
    with tempfile.TemporaryDirectory(
        prefix="govknowledge-normalized-"
    ) as temporary_directory:
        derived = (
            Path(temporary_directory)
            / "normalized.pdf"
        )

        try:
            normalization = (
                normalize_pdf_copy(
                    source_path=source,
                    derived_path=derived,
                    expected_page_count=(
                        source_page_count
                    ),
                )
            )

        except PostNormalizationMismatchError as exc:
            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            terminal = (
                _terminal_normalization_failure(
                    initial=initial,
                    failure_class=(
                        ParserFailureClass
                        .POST_NORMALIZATION_MISMATCH
                    ),
                    exc=exc,
                    elapsed_ms=elapsed_ms,
                )
            )

            return ResilientDocumentWorkerResult(
                result=terminal,
                original_sha256=(
                    original_sha256
                ),
                derived_sha256=None,
                normalization_attempted=True,
                normalization_attempt_count=1,
                initial_failure_class=(
                    initial_failure
                ),
                retry_status=None,
                source_page_count=(
                    source_page_count
                ),
                derived_page_count=None,
                end_to_end_document_processing_ms=(
                    elapsed_ms
                ),
            )

        except PdfNormalizationError as exc:
            elapsed_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            terminal = (
                _terminal_normalization_failure(
                    initial=initial,
                    failure_class=(
                        ParserFailureClass
                        .NORMALIZATION_FAILED
                    ),
                    exc=exc,
                    elapsed_ms=elapsed_ms,
                )
            )

            return ResilientDocumentWorkerResult(
                result=terminal,
                original_sha256=(
                    original_sha256
                ),
                derived_sha256=None,
                normalization_attempted=True,
                normalization_attempt_count=1,
                initial_failure_class=(
                    initial_failure
                ),
                retry_status=None,
                source_page_count=(
                    source_page_count
                ),
                derived_page_count=None,
                end_to_end_document_processing_ms=(
                    elapsed_ms
                ),
            )

        retry = run_document_worker(
            file_path=derived,
            language=language,
            timeout_seconds=timeout_seconds,
            artifacts_path=artifacts_path,
        )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        # Critical invariant:
        # no call back into this function.
        # Therefore normalization can occur
        # no more than once.

        return ResilientDocumentWorkerResult(
            result=retry,
            original_sha256=(
                normalization.original_sha256
            ),
            derived_sha256=(
                normalization.derived_sha256
            ),
            normalization_attempted=True,
            normalization_attempt_count=1,
            initial_failure_class=(
                initial_failure
            ),
            retry_status=(
                retry.parsing_status
            ),
            source_page_count=(
                normalization.source_page_count
            ),
            derived_page_count=(
                normalization.derived_page_count
            ),
            end_to_end_document_processing_ms=(
                elapsed_ms
            ),
        )
