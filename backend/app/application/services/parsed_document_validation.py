from __future__ import annotations

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.domain.exceptions.document_processing import (
    ParsedDocumentValidationError,
)


def validate_parsed_document(
    parsed: ParsedDocument,
    *,
    expected_language: str,
) -> None:
    if not parsed.pages:
        raise ParsedDocumentValidationError(
            "Parsed document contains no pages."
        )

    page_numbers: set[int] = set()

    for page in parsed.pages:
        if page.page_number < 1:
            raise ParsedDocumentValidationError(
                "Page number must be positive."
            )

        if page.page_number in page_numbers:
            raise ParsedDocumentValidationError(
                "Duplicate page number."
            )

        page_numbers.add(
            page.page_number
        )

    chunk_indices: set[int] = set()

    for chunk in parsed.chunks:
        if chunk.chunk_index < 0:
            raise ParsedDocumentValidationError(
                "Chunk index must be nonnegative."
            )

        if (
            chunk.chunk_index
            in chunk_indices
        ):
            raise ParsedDocumentValidationError(
                "Duplicate chunk index."
            )

        chunk_indices.add(
            chunk.chunk_index
        )

        if (
            chunk.page_number
            not in page_numbers
        ):
            raise ParsedDocumentValidationError(
                "Chunk references an unknown page."
            )

        if not chunk.content.strip():
            raise ParsedDocumentValidationError(
                "Chunk content cannot be empty."
            )

        if (
            chunk.language
            != expected_language
        ):
            raise ParsedDocumentValidationError(
                "Chunk language does not match "
                "the canonical document language."
            )
