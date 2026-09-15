from __future__ import annotations

from pathlib import Path

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.application.ports.parser_execution import (
    ParserExecutionResult,
)
from app.domain.enums.processing import (
    ParsingStatus,
)


def requires_ocr_fallback(
    file_path: Path,
    result: ParserExecutionResult,
) -> bool:
    """
    Request the single Modern Docling fallback when a
    successful primary result contains a raster-only page
    that PyMuPDF4LLM did not recover as text.

    Qualified routing signal:

        page.text is empty
        AND native_text_chars == 0
        AND image_count > 0

    Primary failures are handled by RoutedParserExecutor
    before this predicate is called.
    """

    _ = file_path

    if (
        result.parsing_status
        is not ParsingStatus.SUCCESS
    ):
        return False

    parsed = result.parsed_document

    if not isinstance(
        parsed,
        ParsedDocument,
    ):
        raise RuntimeError(
            "Successful primary parser result "
            "has no ParsedDocument."
        )

    for page in parsed.pages:
        if page.text.strip():
            continue

        native_text_chars = (
            page.metadata.get(
                "native_text_chars"
            )
        )

        image_count = (
            page.metadata.get(
                "image_count"
            )
        )

        if (
            type(native_text_chars)
            is not int
            or type(image_count)
            is not int
        ):
            raise RuntimeError(
                "Blank primary parser page is missing "
                "integer OCR-routing metadata."
            )

        if (
            native_text_chars == 0
            and image_count > 0
        ):
            return True

    return False