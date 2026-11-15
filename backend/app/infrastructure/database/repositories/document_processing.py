from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.application.ports.document_processing_repository import (
    DocumentProcessingRepository,
    ProcessingJobAttempt,
    ProcessingTarget,
)
from app.application.services.table_evidence_atoms import (
    TableDocumentContext,
    TableEvidenceAtomBuilder,
)
from app.application.services.table_evidence_compilation import (
    TableEvidenceCompilationService,
)
from app.application.services.table_evidence_guard import (
    UnapprovedTableEvidenceError,
    reject_unapproved_table_evidence,
)
from app.domain.enums.table import (
    TableEvidenceLayer,
)
from app.infrastructure.database.repositories.table_evidence import (
    PostgresTableEvidenceRepository,
)
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
    ProcessingStatus,
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
from app.infrastructure.parsing.reviewed_table_approval import (
    validate_reviewed_table_evidence,
)


class PostgresDocumentProcessingRepository(
    DocumentProcessingRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def _load(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
    ) -> tuple[
        DocumentVersion,
        DocumentProcessingJob,
    ]:
        version = self._db.get(
            DocumentVersion,
            version_id,
        )

        job = self._db.get(
            DocumentProcessingJob,
            processing_job_id,
        )

        if version is None:
            raise ValueError(
                "Document version does not exist."
            )

        if job is None:
            raise ValueError(
                "Processing job does not exist."
            )

        if (
            job.document_version_id
            != version.id
        ):
            raise ValueError(
                "Processing job does not belong "
                "to document version."
            )

        return version, job

    def create_retry_job(
        self,
        *,
        version_id: UUID,
    ) -> ProcessingJobAttempt:
        # Lock the version while allocating the next
        # attempt number. This prevents two callers
        # from creating the same retry concurrently.
        version = self._db.scalar(
            select(
                DocumentVersion
            )
            .where(
                DocumentVersion.id
                == version_id
            )
            .with_for_update()
        )

        if version is None:
            raise ValueError(
                "Document version does not exist."
            )

        if (
            version.processing_status
            != ProcessingStatus.FAILED.value
        ):
            raise ValueError(
                "Only a FAILED document version "
                "can be retried."
            )

        active_job = self._db.scalar(
            select(
                DocumentProcessingJob.id
            )
            .where(
                DocumentProcessingJob.document_version_id
                == version_id,
                DocumentProcessingJob.status.in_(
                    (
                        ProcessingStatus.PENDING.value,
                        ProcessingStatus.PROCESSING.value,
                    )
                ),
            )
            .limit(1)
        )

        if active_job is not None:
            raise ValueError(
                "Document version already has "
                "a pending or active processing job."
            )

        latest_attempt = self._db.scalar(
            select(
                func.max(
                    DocumentProcessingJob.attempt_number
                )
            ).where(
                DocumentProcessingJob.document_version_id
                == version_id
            )
        )

        if latest_attempt is None:
            raise ValueError(
                "Document version has no previous "
                "processing attempt."
            )

        next_attempt = latest_attempt + 1
        processing_job_id = uuid4()

        job = DocumentProcessingJob(
            id=processing_job_id,
            document_version_id=version_id,
            status=ProcessingStatus.PENDING.value,
            attempt_number=next_attempt,
        )

        self._db.add(job)

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        return ProcessingJobAttempt(
            processing_job_id=processing_job_id,
            version_id=version_id,
            attempt_number=next_attempt,
        )

    def mark_processing(
        self,
        *,
        processing_job_id: UUID,
    ) -> ProcessingTarget:
        # Lock the job so two workers cannot
        # start the same PENDING attempt.
        job = self._db.scalar(
            select(
                DocumentProcessingJob
            )
            .where(
                DocumentProcessingJob.id
                == processing_job_id
            )
            .with_for_update()
        )

        if job is None:
            raise ValueError(
                "Processing job does not exist."
            )

        version = self._db.scalar(
            select(
                DocumentVersion
            )
            .where(
                DocumentVersion.id
                == job.document_version_id
            )
            .with_for_update()
        )

        if version is None:
            raise ValueError(
                "Document version does not exist."
            )

        document = self._db.get(
            Document,
            version.document_id,
        )

        if document is None:
            raise ValueError(
                "Document does not exist."
            )

        if (
            job.status
            != ProcessingStatus.PENDING.value
        ):
            raise ValueError(
                "Processing job must be PENDING "
                f"before starting; got {job.status}."
            )

        allowed_version_statuses = {
            ProcessingStatus.PENDING.value,
            ProcessingStatus.FAILED.value,
        }

        if (
            version.processing_status
            not in allowed_version_statuses
        ):
            raise ValueError(
                "Document version cannot start "
                "processing from status "
                f"{version.processing_status}."
            )

        # Capture the authoritative target before
        # commit. Nothing below is supplied by
        # the processing caller.
        target = ProcessingTarget(
            processing_job_id=job.id,
            version_id=version.id,
            document_id=version.document_id,
            storage_path=version.storage_path,
            language=document.language,
        )

        now = datetime.now(
            timezone.utc
        )

        version.processing_status = (
            ProcessingStatus.PROCESSING.value
        )

        job.status = (
            ProcessingStatus.PROCESSING.value
        )

        job.started_at = now
        job.completed_at = None
        job.error_code = None
        job.error_message = None

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        return target

    def record_parser_outcome(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
        parsing_status: ParsingStatus,
        failure_class: ParserFailureClass | None,
        parser_lineage: dict[str, object],
    ) -> None:
        version, job = self._load(
            version_id=version_id,
            processing_job_id=processing_job_id,
        )

        if (
            version.processing_status
            != ProcessingStatus.PROCESSING.value
            or job.status
            != ProcessingStatus.PROCESSING.value
        ):
            raise ValueError(
                "Parser outcome can only be "
                "recorded while processing."
            )

        if (
            parsing_status is ParsingStatus.SUCCESS
            and failure_class is not None
        ):
            raise ValueError(
                "Successful parsing cannot have "
                "a failure class."
            )

        if (
            parsing_status is ParsingStatus.FAILED
            and failure_class is None
        ):
            raise ValueError(
                "Failed parsing requires "
                "a failure class."
            )

        job.parsing_status = (
            parsing_status.value
        )

        job.failure_class = (
            failure_class.value
            if failure_class is not None
            else None
        )

        job.parser_lineage_json = dict(
            parser_lineage
        )

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

    def complete_processing(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        processing_job_id: UUID,
        parsed: ParsedDocument,
        resolve_stored_file: Callable[[str], Path] | None = None,
    ) -> None:
        version, job = self._load(
            version_id=version_id,
            processing_job_id=processing_job_id,
        )

        if version.document_id != document_id:
            raise ValueError(
                "Document version does not belong "
                "to document."
            )

        if (
            version.processing_status
            != ProcessingStatus.PROCESSING.value
            or job.status
            != ProcessingStatus.PROCESSING.value
        ):
            raise ValueError(
                "Document processing can only "
                "complete from PROCESSING state."
            )

        existing_page = self._db.scalar(
            select(
                DocumentPage.id
            ).where(
                DocumentPage.document_version_id
                == version_id
            ).limit(1)
        )

        if existing_page is not None:
            raise ValueError(
                "Document version already contains "
                "parsed pages."
            )

        # Normal documents pass the existing guard unchanged.
        # Table evidence requires a complete source-bound validation.
        try:
            reject_unapproved_table_evidence(parsed)
        except UnapprovedTableEvidenceError:
            if resolve_stored_file is None:
                raise

            # Resolve the path from the database version, not from
            # a path supplied by the request or parser.
            stored_pdf_path = resolve_stored_file(version.storage_path)

            approved_indexes = validate_reviewed_table_evidence(
                parsed=parsed,
                stored_pdf_path=stored_pdf_path,
                version_file_hash=version.file_hash,
            )

            if len(approved_indexes) != 39:
                raise ValueError(
                    "Reviewed table approval did not cover all 39 chunks."
                )

        physical_table_evidence = tuple(
            table
            for table in parsed.table_evidence
            if (
                table.layer
                == TableEvidenceLayer.PHYSICAL
            )
        )
        structural_table_evidence = tuple(
            table
            for table in parsed.table_evidence
            if (
                table.layer
                == TableEvidenceLayer.STRUCTURAL
            )
        )
        table_compilation = (
            TableEvidenceCompilationService()
            .compile(
                physical_tables=(
                    physical_table_evidence
                ),
                structural_tables=(
                    structural_table_evidence
                ),
            )
        )
        document = self._db.get(
            Document,
            document_id,
        )
        if document is None:
            raise ValueError(
                "Document does not exist."
            )

        # Only now may page/chunk persistence begin.

        page_rows: dict[
            int,
            DocumentPage,
        ] = {}

        try:
            for parsed_page in parsed.pages:
                page = DocumentPage(
                    document_version_id=version_id,
                    page_number=(
                        parsed_page.page_number
                    ),
                    extracted_text=(
                        parsed_page.text
                    ),
                    metadata_json=(
                        parsed_page.metadata
                    ),
                )

                self._db.add(page)

                page_rows[
                    parsed_page.page_number
                ] = page

            # We need database-generated page UUIDs
            # before creating chunk foreign keys.
            self._db.flush()

            table_context = (
                TableDocumentContext(
                    document_id=document_id,
                    document_version_id=(
                        version_id
                    ),
                    page_ids=tuple(
                        (
                            page_number,
                            page.id,
                        )
                        for (
                            page_number,
                            page,
                        )
                        in sorted(
                            page_rows.items()
                        )
                    ),
                    language=(
                        document.language
                    ),
                )
            )
            table_atom_builder = (
                TableEvidenceAtomBuilder()
            )

            table_sequences = (
                table_compilation.sequences
                or tuple(
                    (table,)
                    for table
                    in table_compilation.tables
                )
            )

            table_atoms = tuple(
                atom
                for sequence in table_sequences
                for atom in (
                    table_atom_builder
                    .build_sequence(
                        tables=sequence,
                        context=table_context,
                    )
                )
            )
            (
                PostgresTableEvidenceRepository(
                    self._db
                )
                .replace_version_atoms(
                    version_id=version_id,
                    atoms=table_atoms,
                    commit=False,
                )
            )

            for parsed_chunk in parsed.chunks:
                page = page_rows.get(
                    parsed_chunk.page_number
                )

                if page is None:
                    raise ValueError(
                        "Chunk references a page "
                        "that was not parsed."
                    )

                chunk = DocumentChunk(
                    document_id=document_id,
                    document_version_id=version_id,
                    page_id=page.id,
                    page_start=(
                        parsed_chunk.page_number
                    ),
                    page_end=(
                        parsed_chunk.page_number
                    ),
                    chunk_index=(
                        parsed_chunk.chunk_index
                    ),
                    section_title=(
                        parsed_chunk.section_title
                    ),
                    content=(
                        parsed_chunk.content
                    ),
                    language=(
                        parsed_chunk.language
                    ),
                    metadata_json=(
                        parsed_chunk.metadata
                    ),
                )

                self._db.add(chunk)

            version.processing_status = (
                ProcessingStatus.SUCCEEDED.value
            )

            job.status = (
                ProcessingStatus.SUCCEEDED.value
            )

            job.completed_at = datetime.now(
                timezone.utc
            )

            self._db.commit()

        except Exception:
            self._db.rollback()
            raise

    def fail_processing(
        self,
        *,
        version_id: UUID,
        processing_job_id: UUID,
        error_code: str,
        error_message: str,
    ) -> None:
        version, job = self._load(
            version_id=version_id,
            processing_job_id=processing_job_id,
        )

        if (
            version.processing_status
            != ProcessingStatus.PROCESSING.value
            or job.status
            != ProcessingStatus.PROCESSING.value
        ):
            raise ValueError(
                "Document processing can only "
                "fail from PROCESSING state."
            )

        version.processing_status = (
            ProcessingStatus.FAILED.value
        )

        job.status = (
            ProcessingStatus.FAILED.value
        )

        job.completed_at = datetime.now(
            timezone.utc
        )

        job.error_code = error_code

        # Avoid dumping enormous exception payloads
        # into PostgreSQL.
        job.error_message = (
            error_message[:4000]
        )

        self._db.commit()
