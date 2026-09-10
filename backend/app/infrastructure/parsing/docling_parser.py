from importlib.metadata import version
from pathlib import Path

from docling.document_converter import DocumentConverter

from app.application.ports.document_parser import (
    DocumentParser,
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)


class DoclingDocumentParser(DocumentParser):
    """Docling adapter for GovKnowledge.

    Docling objects remain inside infrastructure.
    GovKnowledge receives only its own canonical parser types.

    P0 rule:
    - page numbers are 1-based
    - a chunk belongs to exactly one page
    - chunks never cross page boundaries
    """

    def __init__(
        self,
        *,
        converter: DocumentConverter | None = None,
        max_chunk_chars: int = 3000,
    ) -> None:
        if max_chunk_chars < 1:
            raise ValueError(
                "max_chunk_chars must be greater than zero."
            )

        self._converter = (
            converter
            if converter is not None
            else DocumentConverter()
        )

        self._max_chunk_chars = max_chunk_chars

        self._docling_version = version(
            "docling"
        )

        self._docling_core_version = version(
            "docling-core"
        )

    def parse(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParsedDocument:
        path = file_path.resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        conversion = self._converter.convert(
            path,
            raises_on_error=True,
        )

        document = conversion.document

        page_numbers = sorted(
            int(page_number)
            for page_number in document.pages.keys()
        )

        if not page_numbers:
            raise ValueError(
                "Docling produced no pages."
            )

        pages: list[ParsedPage] = []
        chunks: list[ParsedChunk] = []

        chunk_index = 0

        for page_number in page_numbers:
            page_text = document.export_to_text(
                page_no=page_number,
                traverse_pictures=True,
            ).strip()

            parser_metadata = {
                "parser": "docling",
                "docling_version": (
                    self._docling_version
                ),
                "docling_core_version": (
                    self._docling_core_version
                ),
                "page_number": page_number,
            }

            pages.append(
                ParsedPage(
                    page_number=page_number,
                    text=page_text,
                    metadata=parser_metadata.copy(),
                )
            )

            for chunk_text in self._split_page(
                page_text
            ):
                chunks.append(
                    ParsedChunk(
                        page_number=page_number,
                        chunk_index=chunk_index,
                        content=chunk_text,
                        section_title=None,
                        language=language,
                        metadata=(
                            parser_metadata.copy()
                        ),
                    )
                )

                chunk_index += 1

        return ParsedDocument(
            pages=tuple(pages),
            chunks=tuple(chunks),
        )

    def _split_page(
        self,
        text: str,
    ) -> tuple[str, ...]:
        text = text.strip()

        if not text:
            return ()

        paragraphs = [
            paragraph.strip()
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

        chunks: list[str] = []

        current_parts: list[str] = []
        current_length = 0

        def flush() -> None:
            nonlocal current_parts
            nonlocal current_length

            if current_parts:
                chunks.append(
                    "\n\n".join(
                        current_parts
                    )
                )

            current_parts = []
            current_length = 0

        for paragraph in paragraphs:
            if (
                len(paragraph)
                > self._max_chunk_chars
            ):
                flush()

                for start in range(
                    0,
                    len(paragraph),
                    self._max_chunk_chars,
                ):
                    piece = paragraph[
                        start:
                        start + self._max_chunk_chars
                    ].strip()

                    if piece:
                        chunks.append(piece)

                continue

            separator_length = (
                2 if current_parts else 0
            )

            proposed_length = (
                current_length
                + separator_length
                + len(paragraph)
            )

            if (
                current_parts
                and proposed_length
                > self._max_chunk_chars
            ):
                flush()

            if current_parts:
                current_length += 2

            current_parts.append(
                paragraph
            )

            current_length += len(
                paragraph
            )

        flush()

        return tuple(chunks)
