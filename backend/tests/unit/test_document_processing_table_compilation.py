from unittest.mock import MagicMock, call
from uuid import uuid4

import pytest

from app.application.ports.document_parser import (
    ParsedChunk,
    ParsedDocument,
    ParsedPage,
)
from app.domain.enums.processing import (
    ProcessingStatus,
)
from app.domain.enums.table import (
    TableCellTextStatus,
    TableCellValueSource,
    TableEvidenceLayer,
    TableRowRole,
    TableSpanKind,
)
from app.domain.models.table import (
    CanonicalTable,
    CanonicalTableCell,
    CanonicalTableRow,
    TableBBox,
    TableSourceSpan,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.document_processing import (
    PostgresDocumentProcessingRepository,
)
from app.infrastructure.database.repositories.table_evidence import (
    PostgresTableEvidenceRepository,
)


def span(*, page: int, ref: str, text: str | None = None) -> TableSourceSpan:
    return TableSourceSpan(
        page_number=page,
        kind=TableSpanKind.TEXT if text is not None else TableSpanKind.GEOMETRY,
        source_ref=ref,
        bbox=TableBBox(x0=0.0, y0=0.0, x1=10.0, y1=10.0),
        text=text,
    )


def test_replace_version_atoms_commit_flag():
    mock_db = MagicMock()
    mock_db.scalar.return_value = uuid4()
    mock_db.execute.return_value.all.return_value = []

    repo = PostgresTableEvidenceRepository(mock_db)

    # commit=False should not commit
    repo.replace_version_atoms(
        version_id=uuid4(),
        atoms=(),
        commit=False,
    )
    assert not mock_db.commit.called

    # commit=True should commit
    repo.replace_version_atoms(
        version_id=uuid4(),
        atoms=(),
        commit=True,
    )
    assert mock_db.commit.called


def test_complete_processing_wires_compilation_and_stages_atoms(monkeypatch):
    mock_db = MagicMock()
    doc_id = uuid4()
    ver_id = uuid4()
    job_id = uuid4()

    mock_version = MagicMock(spec=DocumentVersion)
    mock_version.id = ver_id
    mock_version.document_id = doc_id
    mock_version.processing_status = ProcessingStatus.PROCESSING.value
    mock_version.storage_path = "path/to/doc.pdf"
    mock_version.file_hash = "abc"

    mock_job = MagicMock(spec=DocumentProcessingJob)
    mock_job.id = job_id
    mock_job.document_version_id = ver_id
    mock_job.status = ProcessingStatus.PROCESSING.value

    mock_document = MagicMock(spec=Document)
    mock_document.id = doc_id
    mock_document.language = "fr"

    def mock_get(model, pk):
        if model is DocumentVersion:
            return mock_version
        if model is DocumentProcessingJob:
            return mock_job
        if model is Document:
            return mock_document
        return None

    mock_db.get.side_effect = mock_get
    mock_db.scalar.return_value = None  # no existing pages

    # Mock compilation service and builder to verify invocation
    mock_compile = MagicMock()
    mock_compile.tables = (MagicMock(spec=CanonicalTable),)
    mock_compile.sequences = (mock_compile.tables,)

    mock_compilation_service = MagicMock()
    mock_compilation_service.compile.return_value = mock_compile

    mock_atom_builder = MagicMock()
    mock_atoms = (MagicMock(),)
    mock_atom_builder.build_sequence.return_value = mock_atoms

    mock_table_repo = MagicMock()

    monkeypatch.setattr(
        "app.infrastructure.database.repositories.document_processing.TableEvidenceCompilationService",
        lambda: mock_compilation_service,
    )
    monkeypatch.setattr(
        "app.infrastructure.database.repositories.document_processing.TableEvidenceAtomBuilder",
        lambda: mock_atom_builder,
    )
    monkeypatch.setattr(
        "app.infrastructure.database.repositories.document_processing.PostgresTableEvidenceRepository",
        lambda db: mock_table_repo,
    )

    parsed = ParsedDocument(
        pages=(
            ParsedPage(page_number=1, text="Page 1", metadata={}),
        ),
        chunks=(
            ParsedChunk(
                page_number=1,
                chunk_index=0,
                content="Chunk 1",
                section_title=None,
                language="fr",
                metadata={},
            ),
        ),
        table_evidence=(),
    )

    repo = PostgresDocumentProcessingRepository(mock_db)
    repo.complete_processing(
        document_id=doc_id,
        version_id=ver_id,
        processing_job_id=job_id,
        parsed=parsed,
    )

    # 1. Compilation service was called with physical and structural tables
    assert mock_compilation_service.compile.called
    compile_kwargs = mock_compilation_service.compile.call_args.kwargs
    assert compile_kwargs["physical_tables"] == ()
    assert compile_kwargs["structural_tables"] == ()

    # 2. flush was called before staging table atoms
    assert mock_db.flush.called

    # 3. atom builder was called with compilation tables and TableDocumentContext
    assert mock_atom_builder.build_sequence.called
    build_kwargs = mock_atom_builder.build_sequence.call_args.kwargs
    assert build_kwargs["tables"] == mock_compile.tables
    context = build_kwargs["context"]
    assert context.document_id == doc_id
    assert context.document_version_id == ver_id
    assert context.language == "fr"

    # 4. PostgresTableEvidenceRepository.replace_version_atoms was called with commit=False
    mock_table_repo.replace_version_atoms.assert_called_once_with(
        version_id=ver_id,
        atoms=mock_atoms,
        commit=False,
    )

    # 5. Final commit happened on the db session
    assert mock_db.commit.called
