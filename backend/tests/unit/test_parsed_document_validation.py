import pytest

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.services.parsed_document_validation import (
    validate_parsed_document,
)
from app.domain.exceptions.document_processing import (
    ParsedDocumentValidationError,
)


def page(
    number,
    text="Page text",
):
    return ParsedPage(
        page_number=number,
        text=text,
        metadata={},
    )


def chunk(
    *,
    page_number=1,
    chunk_index=0,
    content="Evidence text",
    language="fr",
):
    return ParsedChunk(
        page_number=page_number,
        chunk_index=chunk_index,
        content=content,
        section_title=None,
        language=language,
        metadata={},
    )


def validate(
    parsed,
):
    validate_parsed_document(
        parsed,
        expected_language="fr",
    )


def test_valid_document_passes():
    validate(
        ParsedDocument(
            pages=(
                page(1),
                page(2, text=""),
            ),
            chunks=(
                chunk(
                    page_number=1,
                    chunk_index=0,
                ),
            ),
        )
    )


def test_no_pages_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="no pages",
    ):
        validate(
            ParsedDocument(
                pages=(),
                chunks=(),
            )
        )


def test_nonpositive_page_number_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="positive",
    ):
        validate(
            ParsedDocument(
                pages=(page(0),),
                chunks=(),
            )
        )


def test_duplicate_page_number_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="Duplicate page",
    ):
        validate(
            ParsedDocument(
                pages=(
                    page(1),
                    page(1),
                ),
                chunks=(),
            )
        )


def test_negative_chunk_index_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="nonnegative",
    ):
        validate(
            ParsedDocument(
                pages=(page(1),),
                chunks=(
                    chunk(
                        chunk_index=-1
                    ),
                ),
            )
        )


def test_duplicate_chunk_index_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="Duplicate chunk",
    ):
        validate(
            ParsedDocument(
                pages=(
                    page(1),
                    page(2),
                ),
                chunks=(
                    chunk(
                        page_number=1,
                        chunk_index=0,
                    ),
                    chunk(
                        page_number=2,
                        chunk_index=0,
                    ),
                ),
            )
        )


def test_unknown_chunk_page_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="unknown page",
    ):
        validate(
            ParsedDocument(
                pages=(page(1),),
                chunks=(
                    chunk(
                        page_number=2
                    ),
                ),
            )
        )


def test_empty_chunk_content_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="cannot be empty",
    ):
        validate(
            ParsedDocument(
                pages=(page(1),),
                chunks=(
                    chunk(
                        content="   "
                    ),
                ),
            )
        )


def test_wrong_chunk_language_is_rejected():
    with pytest.raises(
        ParsedDocumentValidationError,
        match="canonical document language",
    ):
        validate(
            ParsedDocument(
                pages=(page(1),),
                chunks=(
                    chunk(
                        language="en"
                    ),
                ),
            )
        )
