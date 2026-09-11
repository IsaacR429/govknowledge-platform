from __future__ import annotations

from app.application.ports.citation_source_repository import (
    CitationSourceRepository,
)
from app.application.services.citation_mapping import (
    Citation,
)


class InvalidCitationError(Exception):
    pass


class CitationValidator:
    def __init__(
        self,
        *,
        repository: CitationSourceRepository,
    ) -> None:
        self._repository = repository

    def validate(
        self,
        citation: Citation,
    ) -> Citation:
        source = (
            self._repository.get_by_chunk_id(
                citation.chunk_id
            )
        )

        if source is None:
            raise InvalidCitationError(
                "Citation source is invalid."
            )

        if (
            source.document_id
            != citation.document_id
        ):
            self._invalid()

        if (
            source.document_version_id
            != citation.document_version_id
        ):
            self._invalid()

        if (
            source.page_id
            != citation.page_id
        ):
            self._invalid()

        if (
            source.page_number
            != citation.page_number
        ):
            self._invalid()

        if citation.start_char < 0:
            self._invalid()

        if (
            citation.end_char
            <= citation.start_char
        ):
            self._invalid()

        if (
            citation.end_char
            > len(source.content)
        ):
            self._invalid()

        canonical_text = source.content[
            citation.start_char:
            citation.end_char
        ]

        if (
            canonical_text
            != citation.cited_text
        ):
            self._invalid()

        return citation

    @staticmethod
    def _invalid() -> None:
        raise InvalidCitationError(
            "Citation source is invalid."
        )
