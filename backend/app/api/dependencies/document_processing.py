"""API composition for document processing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, status

from app.application.ports.parser_execution import ParserExecutor
from app.application.services.audit import AuditService
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.core.config import get_settings
from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
)
from app.infrastructure.database.repositories.document_processing import (
    PostgresDocumentProcessingRepository,
)
from app.infrastructure.database.session import SessionLocal
from app.infrastructure.parsing.modern_docling_executor import (
    ModernDoclingParserExecutor,
)
from app.infrastructure.parsing.ocr_fallback_policy import (
    requires_ocr_fallback,
)
from app.infrastructure.parsing.pymupdf_executor import (
    PyMuPDFParserExecutor,
)
from app.infrastructure.parsing.routed_parser_executor import (
    RoutedParserExecutor,
)
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)

logger = logging.getLogger(__name__)

PARSER_TIMEOUT_SECONDS = 600.0
MAX_CHUNK_CHARS = 1500


def _required_file_from_env(name: str) -> Path:
    raw = os.environ.get(name)
    if not raw:
        raise RuntimeError(f"Missing parser configuration: {name}")

    path = Path(raw).resolve()
    if not path.is_file():
        raise RuntimeError(
            f"Configured parser executable does not exist: {name}"
        )

    return path


def build_routed_parser_executor() -> ParserExecutor:
    """Validate both isolated runtimes before ingestion begins."""
    backend_root = Path(__file__).resolve().parents[3]

    pymupdf_python = _required_file_from_env(
        "GOVKNOWLEDGE_PYMUPDF_PYTHON"
    )
    docling_python = _required_file_from_env(
        "GOVKNOWLEDGE_MODERN_DOCLING_PYTHON"
    )

    artifact_raw = os.environ.get(
        "GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT"
    )
    if (
        not artifact_raw
        or not Path(artifact_raw).resolve().is_dir()
    ):
        raise RuntimeError(
            "GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT "
            "is not configured."
        )

    pymupdf_worker = (
        backend_root
        / "app/infrastructure/parsing/pymupdf_worker_entry.py"
    )
    docling_worker = (
        backend_root
        / "app/infrastructure/parsing/modern_docling_worker_entry.py"
    )

    if (
        not pymupdf_worker.is_file()
        or not docling_worker.is_file()
    ):
        raise RuntimeError(
            "Configured parser worker script does not exist."
        )

    return RoutedParserExecutor(
        primary=PyMuPDFParserExecutor(
            python_executable=pymupdf_python,
            worker_script=pymupdf_worker,
            timeout_seconds=PARSER_TIMEOUT_SECONDS,
            max_chunk_chars=MAX_CHUNK_CHARS,
        ),
        fallback=ModernDoclingParserExecutor(
            python_executable=docling_python,
            worker_script=docling_worker,
            timeout_seconds=PARSER_TIMEOUT_SECONDS,
            max_chunk_chars=MAX_CHUNK_CHARS,
        ),
        primary_name="pymupdf4llm",
        fallback_name="docling-modern",
        fallback_predicate=requires_ocr_fallback,
    )


def get_routed_parser_executor() -> ParserExecutor:
    """Return HTTP 503 before creating an upload if setup is missing."""
    try:
        return build_routed_parser_executor()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.error(
            "Document parser setup unavailable: %s",
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document processing is not configured.",
        ) from exc


def run_document_processing_job(
    processing_job_id: UUID,
    parser_executor: ParserExecutor,
) -> None:
    """Run after HTTP 201, without borrowing the request's DB session."""
    db = SessionLocal()

    try:
        audit = AuditService(
            repository=PostgresAuditRepository(db)
        )

        service = DocumentProcessingService(
            repository=PostgresDocumentProcessingRepository(db),
            storage=LocalFileStorage(
                Path(get_settings().document_storage_root)
            ),
            audit=audit,
            parser_executor=parser_executor,
        )

        service.process(
            ProcessDocumentRequest(
                processing_job_id=processing_job_id
            )
        )

    except Exception:
        db.rollback()
        logger.exception(
            "Background document processing failed: job=%s",
            processing_job_id,
        )

    finally:
        db.close()