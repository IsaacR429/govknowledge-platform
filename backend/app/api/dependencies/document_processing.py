"""API composition for document processing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import update

from app.application.ports.parser_execution import ParserExecutor
from app.application.services.audit import AuditService
from app.application.services.document_indexing import DocumentIndexingService
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.core.config import get_settings
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
)
from app.infrastructure.database.repositories.chunk_embeddings import (
    PostgresChunkEmbeddingRepository,
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
from app.infrastructure.parsing.page_ocr_recovery import PageOcrRecovery
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


def _required_file_from_env(name: str, default: Path) -> Path:
    configured = os.environ.get(name)
    path = (
        Path(configured) if configured else default
    ).expanduser().resolve()

    if not path.is_file():
        raise RuntimeError(
            f"Parser executable missing ({name}): {path}"
        )

    return path


def build_routed_parser_executor() -> ParserExecutor:
    """Validate both isolated runtimes before ingestion begins."""
    backend_root = Path(__file__).resolve().parents[3]
    platform_root = backend_root.parent

    pymupdf_python = _required_file_from_env(
        "GOVKNOWLEDGE_PYMUPDF_PYTHON",
        platform_root
        / ".runtime/venvs/pymupdf4llm/Scripts/python.exe",
    )

    docling_python = _required_file_from_env(
        "GOVKNOWLEDGE_MODERN_DOCLING_PYTHON",
        platform_root
        / ".runtime/venvs/docling-modern/Scripts/python.exe",
    )

    artifact_root = Path(
        os.environ.get("GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT")
        or platform_root / ".runtime/artifacts/docling-p0"
    ).expanduser().resolve()

    if not artifact_root.is_dir():
        raise RuntimeError(
            f"Docling artifact directory missing: {artifact_root}"
        )

    pymupdf_worker = (
        backend_root
        / "app/infrastructure/parsing/pymupdf_worker_entry.py"
    )
    docling_worker = (
        backend_root
        / "app/infrastructure/parsing/modern_docling_worker_entry.py"
    )
    ocr_worker = (
        backend_root
        / "app/infrastructure/parsing/rapidocr_page_worker_entry.py"
    )

    if (
        not pymupdf_worker.is_file()
        or not docling_worker.is_file()
        or not ocr_worker.is_file()
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
        ocr_recovery=PageOcrRecovery(
            python_executable=docling_python,
            worker_script=ocr_worker,
            timeout_seconds=120.0,
            max_chunk_chars=MAX_CHUNK_CHARS,
        ),
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


def _set_indexing_status(
    processing_job_id: UUID,
    *,
    status: str,
    error_code: str | None = None,
) -> None:
    """Commit indexing state independently of embedding transactions."""
    with SessionLocal() as status_db:
        job = status_db.get(DocumentProcessingJob, processing_job_id)
        if job is None:
            raise RuntimeError("Document processing job not found.")

        job.indexing_status = status
        job.indexing_error_code = error_code

        if status == "INDEXING":
            job.indexing_attempt_number += 1

        status_db.commit()


def _claim_indexing_retry(processing_job_id: UUID) -> UUID | None:
    """Atomically claim a failed indexing job; return its version ID."""
    with SessionLocal() as db:
        version_id = db.scalar(
            update(DocumentProcessingJob)
            .where(
                DocumentProcessingJob.id == processing_job_id,
                DocumentProcessingJob.status == "SUCCEEDED",
                DocumentProcessingJob.parsing_status == "SUCCESS",
                DocumentProcessingJob.indexing_status == "FAILED",
            )
            .values(
                indexing_status="INDEXING",
                indexing_error_code=None,
                indexing_attempt_number=(
                    DocumentProcessingJob.indexing_attempt_number + 1
                ),
            )
            .returning(DocumentProcessingJob.document_version_id)
        )

        if version_id is None:
            db.rollback()
            return None

        db.commit()
        return version_id


def run_document_indexing_retry(
    processing_job_id: UUID,
    version_id: UUID,
) -> None:
    """Retry indexing already-parsed chunks; never reparse the PDF."""
    db = SessionLocal()
    try:
        from app.api.dependencies.qa import get_qa_embedding_provider

        indexer = DocumentIndexingService(
            repository=PostgresChunkEmbeddingRepository(db),
            embedding_provider=get_qa_embedding_provider(),
        )
        result = indexer.index_version(version_id=version_id)

        _set_indexing_status(
            processing_job_id,
            status="SUCCEEDED",
        )
        logger.info(
            "Document indexing retry completed: "
            "job=%s version=%s new_embeddings=%s",
            processing_job_id,
            version_id,
            result.indexed_count,
        )

    except Exception:
        db.rollback()

        try:
            _set_indexing_status(
                processing_job_id,
                status="FAILED",
                error_code="INDEXING_FAILED",
            )
        except Exception:
            logger.exception(
                "Could not persist indexing retry failure: job=%s",
                processing_job_id,
            )

        logger.exception(
            "Document indexing retry failed: job=%s version=%s",
            processing_job_id,
            version_id,
        )

    finally:
        db.close()


def run_document_processing_job(
    processing_job_id: UUID,
    parser_executor: ParserExecutor,
    actor_user_id: UUID | None = None,
    correlation_id: UUID | None = None,
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

        result = service.process(
            ProcessDocumentRequest(
                processing_job_id=processing_job_id,
                actor_user_id=actor_user_id,
                correlation_id=correlation_id,
            )
        )

        # Parsing has succeeded. Indexing has its own lifecycle.
        _set_indexing_status(
            processing_job_id,
            status="PENDING",
        )

        if result.chunk_count == 0:
            _set_indexing_status(
                processing_job_id,
                status="NO_CHUNKS",
            )
            logger.warning(
                "Document processed without searchable chunks: "
                "job=%s version=%s",
                processing_job_id,
                result.version_id,
            )
        else:
            try:
                _set_indexing_status(
                    processing_job_id,
                    status="INDEXING",
                )

                # Load the same cached embedding model used by Q&A,
                # but only after parsing has succeeded.
                from app.api.dependencies.qa import get_qa_embedding_provider

                indexer = DocumentIndexingService(
                    repository=PostgresChunkEmbeddingRepository(db),
                    embedding_provider=get_qa_embedding_provider(),
                )
                index_result = indexer.index_version(
                    version_id=result.version_id
                )

            except Exception:
                db.rollback()

                # Do not persist raw exception text: it could contain
                # provider details or other sensitive information.
                try:
                    _set_indexing_status(
                        processing_job_id,
                        status="FAILED",
                        error_code="INDEXING_FAILED",
                    )
                except Exception:
                    logger.exception(
                        "Could not persist indexing failure: job=%s",
                        processing_job_id,
                    )

                raise  # Preserve the original error for server logging.

            _set_indexing_status(
                processing_job_id,
                status="SUCCEEDED",
            )
            logger.info(
                "Document indexing completed: "
                "job=%s version=%s new_embeddings=%s",
                processing_job_id,
                result.version_id,
                index_result.indexed_count,
            )

    except Exception:
        db.rollback()
        logger.exception(
            "Background document processing failed: job=%s",
            processing_job_id,
        )

    finally:
        db.close()