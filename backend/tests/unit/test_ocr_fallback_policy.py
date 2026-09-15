from __future__ import annotations

from pathlib import Path

import pytest

from app.application.ports.document_parser import (
    ParsedDocument,
    ParsedPage,
)
from app.application.ports.parser_execution import (
    ParserExecutionResult,
)
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)
from app.infrastructure.parsing.ocr_fallback_policy import (
    requires_ocr_fallback,
)


SOURCE = Path("document.pdf")


def success_result(
    *pages: ParsedPage,
) -> ParserExecutionResult:
    return ParserExecutionResult(
        parsing_status=(
            ParsingStatus.SUCCESS
        ),
        failure_class=None,
        parsed_document=ParsedDocument(
            pages=tuple(pages),
            chunks=(),
        ),
        lineage={},
    )


def page(
    *,
    page_number: int,
    text: str,
    native_text_chars: int,
    image_count: int,
) -> ParsedPage:
    return ParsedPage(
        page_number=page_number,
        text=text,
        metadata={
            "parser": "pymupdf4llm",
            "native_text_chars":
                native_text_chars,
            "image_count":
                image_count,
        },
    )


def test_requests_fallback_for_raster_only_blank_page() -> None:
    result = success_result(
        page(
            page_number=1,
            text="",
            native_text_chars=0,
            image_count=1,
        )
    )

    assert (
        requires_ocr_fallback(
            SOURCE,
            result,
        )
        is True
    )


def test_does_not_route_legitimate_blank_page_without_image() -> None:
    result = success_result(
        page(
            page_number=1,
            text="",
            native_text_chars=0,
            image_count=0,
        )
    )

    assert (
        requires_ocr_fallback(
            SOURCE,
            result,
        )
        is False
    )


def test_does_not_route_page_with_native_text_signal() -> None:
    result = success_result(
        page(
            page_number=1,
            text="",
            native_text_chars=20,
            image_count=1,
        )
    )

    assert (
        requires_ocr_fallback(
            SOURCE,
            result,
        )
        is False
    )


def test_does_not_route_nonempty_page_with_image() -> None:
    result = success_result(
        page(
            page_number=1,
            text="native text",
            native_text_chars=11,
            image_count=4,
        )
    )

    assert (
        requires_ocr_fallback(
            SOURCE,
            result,
        )
        is False
    )


def test_any_qualifying_page_requests_fallback() -> None:
    result = success_result(
        page(
            page_number=1,
            text="native page",
            native_text_chars=11,
            image_count=0,
        ),
        page(
            page_number=2,
            text="",
            native_text_chars=0,
            image_count=1,
        ),
        page(
            page_number=3,
            text="native page",
            native_text_chars=11,
            image_count=0,
        ),
    )

    assert (
        requires_ocr_fallback(
            SOURCE,
            result,
        )
        is True
    )


def test_failed_primary_is_not_policy_routed() -> None:
    result = ParserExecutionResult(
        parsing_status=(
            ParsingStatus.FAILED
        ),
        failure_class=(
            ParserFailureClass.UNKNOWN
        ),
        parsed_document=None,
        lineage={},
    )

    assert (
        requires_ocr_fallback(
            SOURCE,
            result,
        )
        is False
    )


def test_blank_page_with_missing_signal_fails_loudly() -> None:
    result = success_result(
        ParsedPage(
            page_number=1,
            text="",
            metadata={
                "parser":
                    "pymupdf4llm",
            },
        )
    )

    with pytest.raises(
        RuntimeError,
        match="OCR-routing metadata",
    ):
        requires_ocr_fallback(
            SOURCE,
            result,
        )