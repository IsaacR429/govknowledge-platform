import pytest

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.application.services.table_evidence_guard import (
    reject_unapproved_table_evidence,
)


def make_document(chunk_metadata, page_metadata=None):
    return ParsedDocument(
        pages=(
            ParsedPage(
                page_number=19,
                text="Original page text",
                metadata=page_metadata or {},
            ),
        ),
        chunks=(
            ParsedChunk(
                page_number=19,
                chunk_index=0,
                content="Example evidence",
                section_title=None,
                language="en",
                metadata=chunk_metadata,
            ),
        ),
    )


def test_ordinary_text_and_ocr_chunks_are_allowed():
    reject_unapproved_table_evidence(
        make_document({"parser": "pymupdf4llm"})
    )
    reject_unapproved_table_evidence(
        make_document({
            "extraction_method": "rapidocr_page_recovery",
        })
    )


def test_page_metadata_candidates_are_not_searchable_chunks():
    reject_unapproved_table_evidence(
        make_document(
            {"parser": "pymupdf4llm"},
            page_metadata={"table_candidates": [{"marker_count": 65}]},
        )
    )


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "evidence_type": "vector_marker_table",
            "validation_status": "candidate",
        },
        {
            "evidence_type": "vector_marker_table",
            "validation_status": "verified",
        },
        {
            "extraction_method": "vector_marker_table_candidate",
        },
        {
            "table_id": "page-19-table-1",
            "marked_activities": ["Creation"],
        },
    ],
)
def test_unapproved_table_chunks_are_rejected(metadata):
    with pytest.raises(
        ValueError,
        match="Unapproved table evidence",
    ):
        reject_unapproved_table_evidence(
            make_document(metadata)
        )
