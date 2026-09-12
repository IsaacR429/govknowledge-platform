from importlib.metadata import version
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)

from app.application.ports.document_parser import (
    DocumentParser,
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.infrastructure.parsing.docling_artifacts import (
    OCR_BACKEND,
    P0_MAX_CHUNK_CHARS,
    load_docling_artifact_bundle,
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
        artifacts_path: Path | None = None,
        max_chunk_chars: int = P0_MAX_CHUNK_CHARS,
    ) -> None:
        if max_chunk_chars < 1:
            raise ValueError(
                "max_chunk_chars must be greater than zero."
            )

        self._artifact_bundle_id: str | None = None
        self._artifact_bundle_revision: str | None = None
        self._ocr_model_family: str | None = None
        self._chunking_strategy: str | None = None

        if converter is not None:
            self._converter = converter
        else:
            bundle = load_docling_artifact_bundle(
                artifacts_path
            )

            if (
                max_chunk_chars
                != bundle.max_chunk_chars
            ):
                raise ValueError(
                    "Production Docling parser "
                    "chunk size must match the "
                    "frozen P0 bundle."
                )

            pipeline_options = PdfPipelineOptions(
                artifacts_path=bundle.root,
                do_ocr=True,
                ocr_options=RapidOcrOptions(
                    backend=OCR_BACKEND,
                    det_model_path=str(
                        bundle.ocr_det_model_path
                    ),
                    cls_model_path=str(
                        bundle.ocr_cls_model_path
                    ),
                    rec_model_path=str(
                        bundle.ocr_rec_model_path
                    ),
                    rapidocr_params={
                        "Rec.rec_keys_path": None,
                    },
                ),
            )

            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=(
                            pipeline_options
                        ),
                    )
                }
            )

            self._artifact_bundle_id = (
                bundle.bundle_id
            )
            self._artifact_bundle_revision = (
                bundle.bundle_revision
            )
            self._ocr_model_family = (
                bundle.ocr_model_family
            )
            self._chunking_strategy = (
                bundle.chunking_strategy
            )

        self._max_chunk_chars = max_chunk_chars

        self._docling_version = version(
            "docling"
        )

        self._docling_core_version = version(
            "docling-core"
        )

        self._docling_ibm_models_version = version(
            "docling-ibm-models"
        )

        self._docling_parse_version = version(
            "docling-parse"
        )

        self._rapidocr_version = version(
            "rapidocr"
        )

        self._onnxruntime_version = version(
            "onnxruntime"
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
                "docling_ibm_models_version": (
                    self._docling_ibm_models_version
                ),
                "docling_parse_version": (
                    self._docling_parse_version
                ),
                "rapidocr_version": (
                    self._rapidocr_version
                ),
                "onnxruntime_version": (
                    self._onnxruntime_version
                ),
                "max_chunk_chars": (
                    self._max_chunk_chars
                ),
                "page_number": page_number,
            }

            if (
                self._artifact_bundle_id
                is not None
            ):
                parser_metadata[
                    "artifact_bundle_id"
                ] = self._artifact_bundle_id

                parser_metadata[
                    "artifact_bundle_revision"
                ] = (
                    self._artifact_bundle_revision
                )

                parser_metadata[
                    "ocr_model_family"
                ] = self._ocr_model_family

                parser_metadata[
                    "chunking_strategy"
                ] = self._chunking_strategy

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

                remaining = paragraph

                while remaining:
                    if (
                        len(remaining)
                        <= self._max_chunk_chars
                    ):
                        piece = remaining.strip()

                        if piece:
                            chunks.append(piece)

                        break

                    window = remaining[
                        :self._max_chunk_chars
                    ]

                    # Prefer ending after a complete sentence.
                    sentence_boundaries = (
                        window.rfind(". "),
                        window.rfind("? "),
                        window.rfind("! "),
                        window.rfind(".\n"),
                        window.rfind("?\n"),
                        window.rfind("!\n"),
                    )

                    boundary = max(
                        sentence_boundaries
                    )

                    if boundary >= (
                        self._max_chunk_chars // 2
                    ):
                        # Keep the punctuation in this chunk.
                        boundary += 1
                    else:
                        # Otherwise fall back to the last
                        # whitespace before the hard limit.
                        whitespace_boundaries = (
                            window.rfind(" "),
                            window.rfind("\n"),
                            window.rfind("\t"),
                        )

                        boundary = max(
                            whitespace_boundaries
                        )

                        if boundary < 1:
                            # Pathological case: one token itself
                            # exceeds the configured limit.
                            boundary = (
                                self._max_chunk_chars
                            )

                    piece = remaining[
                        :boundary
                    ].strip()

                    if piece:
                        chunks.append(piece)

                    remaining = remaining[
                        boundary:
                    ].strip()

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
