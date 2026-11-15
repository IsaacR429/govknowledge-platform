import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.ports.parser_execution import ParserExecutionResult
from app.domain.enums.processing import ParserFailureClass, ParsingStatus
from app.infrastructure.parsing.ocr_fallback_policy import requires_ocr_fallback
from app.infrastructure.parsing.page_ocr_recovery import PageOcrRecovery
from app.infrastructure.parsing.routed_parser_executor import RoutedParserExecutor


def make_primary_result():
    pages = (
        ParsedPage(1, "", {"native_text_chars": 0, "image_count": 1}),
        ParsedPage(2, "Original body text", {
            "native_text_chars": 18,
            "image_count": 0,
        }),
        ParsedPage(35, "", {"native_text_chars": 0, "image_count": 1}),
    )
    chunks = (
        ParsedChunk(
            page_number=2,
            chunk_index=0,
            content="Original body text",
            section_title=None,
            language="en",
            metadata={"source": "primary"},
        ),
    )
    return ParserExecutionResult(
        parsing_status=ParsingStatus.SUCCESS,
        failure_class=None,
        parsed_document=ParsedDocument(pages=pages, chunks=chunks),
        lineage={"parser_family": "pymupdf4llm"},
    )


def make_router(primary_result):
    primary = Mock()
    primary.execute.return_value = primary_result

    fallback = Mock()

    recovery = PageOcrRecovery(
        python_executable=Path("unused-python.exe"),
        worker_script=Path("unused-worker.py"),
        timeout_seconds=10,
        max_chunk_chars=1500,
    )

    router = RoutedParserExecutor(
        primary=primary,
        fallback=fallback,
        primary_name="pymupdf4llm",
        fallback_name="docling-modern",
        fallback_predicate=requires_ocr_fallback,
        ocr_recovery=recovery,
    )
    return router, fallback


def test_recovers_only_blank_image_pages_and_preserves_primary(
    monkeypatch, tmp_path
):
    def fake_run(command, **kwargs):
        page = int(command[command.index("--page-number") + 1])
        result_path = Path(command[command.index("--result-path") + 1])
        result_path.write_text(
            json.dumps({
                "status": "SUCCESS",
                "page_number": page,
                "text": f"Recovered cover {page}",
            }),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "app.infrastructure.parsing.page_ocr_recovery.subprocess.run",
        fake_run,
    )

    router, fallback = make_router(make_primary_result())
    result = router.execute(tmp_path / "example.pdf", language="en")

    assert result.parsing_status is ParsingStatus.SUCCESS
    assert result.failure_class is None
    assert result.parsed_document is not None

    document = result.parsed_document
    assert [(p.page_number, p.text) for p in document.pages] == [
        (1, "Recovered cover 1"),
        (2, "Original body text"),
        (35, "Recovered cover 35"),
    ]
    assert [c.content for c in document.chunks] == [
        "Recovered cover 1",
        "Original body text",
        "Recovered cover 35",
    ]
    assert [c.chunk_index for c in document.chunks] == [0, 1, 2]
    assert document.chunks[1].metadata == {"source": "primary"}
    assert result.lineage["ocr_recovery"]["recovered_pages"] == [1, 35]
    fallback.execute.assert_not_called()


def test_ocr_native_crash_fails_without_discarding_evidence(
    monkeypatch, tmp_path
):
    def fake_crash(command, **kwargs):
        return subprocess.CompletedProcess(command, -1073740791, "", "")

    monkeypatch.setattr(
        "app.infrastructure.parsing.page_ocr_recovery.subprocess.run",
        fake_crash,
    )

    router, fallback = make_router(make_primary_result())
    result = router.execute(tmp_path / "example.pdf", language="en")

    assert result.parsing_status is ParsingStatus.FAILED
    assert result.failure_class is ParserFailureClass.NATIVE_CRASH
    assert result.parsed_document is None
    assert result.lineage["ocr_recovery"]["failed_page"] == 1
    assert result.lineage["ocr_recovery"]["required_pages"] == [1, 35]
    fallback.execute.assert_not_called()
