from __future__ import annotations

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
    modern_docling_executor as module,
)
from app.infrastructure.parsing.modern_docling_executor import (
    ModernDoclingParserExecutor,
)


def success_worker_result() -> (
    DocumentWorkerRunResult
):
    metadata = {
        "parser": "docling",
        "parser_variant": "modern_fallback",
        "docling_version": "2.126.0",
        "docling_core_version": "2.96.0",
        "docling_ibm_models_version": "4.0.2",
        "docling_parse_version": "7.19.1",
        "rapidocr_version": "3.9.2",
        "onnxruntime_version": "1.30.0",
        "torch_version": "2.14.0",
        "max_chunk_chars": 1500,
        "page_number": 1,
        "ocr_requested": True,
        "ocr_backend": "onnxruntime",
        "ocr_model_family": "PP-OCRv6",
        "chunking_strategy":
            "paragraph_greedy_char_limit",
        "ocr_det_model_sha256": "det-hash",
        "ocr_cls_model_sha256": "cls-hash",
        "ocr_rec_model_sha256": "rec-hash",
    }

    parsed = ParsedDocument(
        pages=(
            ParsedPage(
                page_number=1,
                text="OCR PAGE",
                metadata=metadata.copy(),
            ),
        ),
        chunks=(
            ParsedChunk(
                page_number=1,
                chunk_index=0,
                content="OCR PAGE",
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
) -> ModernDoclingParserExecutor:
    python = tmp_path / "python.exe"
    python.write_bytes(
        b"python"
    )

    worker = tmp_path / "worker.py"
    worker.write_text(
        "# worker",
        encoding="utf-8",
    )

    return ModernDoclingParserExecutor(
        python_executable=python,
        worker_script=worker,
        timeout_seconds=120.0,
        max_chunk_chars=1500,
    )


def test_success_maps_worker_result_and_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(
        b"document bytes"
    )

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
    assert call["timeout_seconds"] == 120.0
    assert call["max_chunk_chars"] == 1500

    lineage = result.lineage

    assert (
        lineage["execution_mode"]
        == "ISOLATED_WORKER"
    )

    assert (
        lineage["parser_family"]
        == "docling"
    )

    assert (
        lineage["parser_variant"]
        == "modern_fallback"
    )

    config = lineage["parser_config"]

    assert (
        config["docling_version"]
        == "2.126.0"
    )

    assert (
        config["docling_ibm_models_version"]
        == "4.0.2"
    )

    assert (
        config["ocr_model_family"]
        == "PP-OCRv6"
    )

    assert (
        config["ocr_backend"]
        == "onnxruntime"
    )

    assert (
        config["ocr_det_model_sha256"]
        == "det-hash"
    )

    assert (
        config["ocr_cls_model_sha256"]
        == "cls-hash"
    )

    assert (
        config["ocr_rec_model_sha256"]
        == "rec-hash"
    )

    metrics = lineage["metrics"]

    assert (
        metrics["worker_peak_rss_bytes"]
        == 123456
    )


def test_timeout_maps_to_failed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(
        b"document bytes"
    )

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


def test_rejects_success_without_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(
        b"document bytes"
    )

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
    source.write_bytes(
        b"document bytes"
    )

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