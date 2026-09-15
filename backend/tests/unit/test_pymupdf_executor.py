from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)
from app.infrastructure.parsing.parser_worker import (
    DocumentWorkerRunResult,
)
from app.infrastructure.parsing import (
    pymupdf_executor as module,
)
from app.infrastructure.parsing.pymupdf_executor import (
    PyMuPDFParserExecutor,
)


def success_worker_result() -> (
    DocumentWorkerRunResult
):
    metadata = {
        "parser": "pymupdf4llm",
        "pymupdf4llm_version": "1.28.2",
        "pymupdf_version": "1.28.2",
        "max_chunk_chars": 1500,
        "chunking_strategy":
            "paragraph_greedy_page_local",
        "page_number": 1,
        "ocr_requested": True,
        "native_text_chars": 5,
        "image_count": 0,
    }

    parsed = ParsedDocument(
        pages=(
            ParsedPage(
                page_number=1,
                text="hello",
                metadata=metadata.copy(),
            ),
        ),
        chunks=(
            ParsedChunk(
                page_number=1,
                chunk_index=0,
                content="hello",
                section_title=None,
                language="en",
                metadata=metadata.copy(),
            ),
        ),
    )

    return DocumentWorkerRunResult(
        parsing_status=(
            ParsingStatus.SUCCESS
        ),
        failure_class=None,
        parsed_document=parsed,
        return_code=0,
        worker_spawn_ms=1.0,
        worker_setup_ms=2.0,
        parser_execution_ms=3.0,
        worker_teardown_ms=4.0,
        end_to_end_document_processing_ms=(
            10.0
        ),
        worker_peak_rss_bytes=123456,
        exception_type=None,
        exception_message=None,
    )


def failed_worker_result() -> (
    DocumentWorkerRunResult
):
    return DocumentWorkerRunResult(
        parsing_status=(
            ParsingStatus.FAILED
        ),
        failure_class=(
            ParserFailureClass.TIMEOUT
        ),
        parsed_document=None,
        return_code=1,
        worker_spawn_ms=1.0,
        worker_setup_ms=None,
        parser_execution_ms=None,
        worker_teardown_ms=2.0,
        end_to_end_document_processing_ms=(
            501.0
        ),
        worker_peak_rss_bytes=654321,
        exception_type=None,
        exception_message=None,
    )


def make_executor(
    tmp_path: Path,
) -> PyMuPDFParserExecutor:
    python = tmp_path / "python.exe"
    python.write_bytes(b"python")

    worker = tmp_path / "worker.py"
    worker.write_text(
        "# worker",
        encoding="utf-8",
    )

    return PyMuPDFParserExecutor(
        python_executable=python,
        worker_script=worker,
        timeout_seconds=30.0,
        max_chunk_chars=1500,
    )


def test_success_maps_worker_result_and_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"document bytes")

    executor = make_executor(
        tmp_path
    )

    calls: list[
        dict[str, object]
    ] = []

    def fake_run(
        **kwargs: object,
    ) -> DocumentWorkerRunResult:
        calls.append(
            kwargs
        )

        return success_worker_result()

    monkeypatch.setattr(
        module,
        "run_external_document_worker",
        fake_run,
    )

    result = executor.execute(
        source,
        language="en",
    )

    assert (
        result.parsing_status
        is ParsingStatus.SUCCESS
    )

    assert result.failure_class is None
    assert result.parsed_document is not None

    assert len(calls) == 1

    call = calls[0]

    assert call["file_path"] == (
        source.resolve()
    )

    assert call["language"] == "en"
    assert call["timeout_seconds"] == 30.0
    assert call["max_chunk_chars"] == 1500

    lineage = result.lineage

    assert (
        lineage["execution_mode"]
        == "ISOLATED_WORKER"
    )

    assert (
        lineage["parser_family"]
        == "pymupdf4llm"
    )

    expected_hash = hashlib.sha256(
        b"document bytes"
    ).hexdigest()

    assert (
        lineage["original_sha256"]
        == expected_hash
    )

    config = lineage["parser_config"]

    assert (
        config["pymupdf4llm_version"]
        == "1.28.2"
    )

    assert (
        config["pymupdf_version"]
        == "1.28.2"
    )

    assert (
        config["chunking_strategy"]
        == "paragraph_greedy_page_local"
    )

    metrics = lineage["metrics"]

    assert (
        metrics["worker_peak_rss_bytes"]
        == 123456
    )

    assert (
        metrics[
            "parser_execution_ms"
        ]
        == 3.0
    )


def test_timeout_maps_to_failed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"document bytes")

    executor = make_executor(
        tmp_path
    )

    monkeypatch.setattr(
        module,
        "run_external_document_worker",
        lambda **kwargs: (
            failed_worker_result()
        ),
    )

    result = executor.execute(
        source,
        language="en",
    )

    assert (
        result.parsing_status
        is ParsingStatus.FAILED
    )

    assert (
        result.failure_class
        is ParserFailureClass.TIMEOUT
    )

    assert result.parsed_document is None

    assert (
        result.lineage["return_code"]
        == 1
    )

    assert (
        result.lineage["metrics"][
            "worker_peak_rss_bytes"
        ]
        == 654321
    )


def test_rejects_success_without_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"document bytes")

    executor = make_executor(
        tmp_path
    )

    invalid = DocumentWorkerRunResult(
        parsing_status=(
            ParsingStatus.SUCCESS
        ),
        failure_class=None,
        parsed_document=None,
        return_code=0,
        worker_spawn_ms=1.0,
        worker_setup_ms=1.0,
        parser_execution_ms=1.0,
        worker_teardown_ms=1.0,
        end_to_end_document_processing_ms=(
            4.0
        ),
        worker_peak_rss_bytes=100,
        exception_type=None,
        exception_message=None,
    )

    monkeypatch.setattr(
        module,
        "run_external_document_worker",
        lambda **kwargs: invalid,
    )

    with pytest.raises(
        RuntimeError,
        match="no ParsedDocument",
    ):
        executor.execute(
            source,
            language="en",
        )


def test_rejects_failed_result_with_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"document bytes")

    executor = make_executor(
        tmp_path
    )

    success = success_worker_result()

    invalid = DocumentWorkerRunResult(
        parsing_status=(
            ParsingStatus.FAILED
        ),
        failure_class=(
            ParserFailureClass.UNKNOWN
        ),
        parsed_document=(
            success.parsed_document
        ),
        return_code=1,
        worker_spawn_ms=1.0,
        worker_setup_ms=1.0,
        parser_execution_ms=1.0,
        worker_teardown_ms=1.0,
        end_to_end_document_processing_ms=(
            4.0
        ),
        worker_peak_rss_bytes=100,
        exception_type="ValueError",
        exception_message="bad",
    )

    monkeypatch.setattr(
        module,
        "run_external_document_worker",
        lambda **kwargs: invalid,
    )

    with pytest.raises(
        RuntimeError,
        match="returned a ParsedDocument",
    ):
        executor.execute(
            source,
            language="en",
        )