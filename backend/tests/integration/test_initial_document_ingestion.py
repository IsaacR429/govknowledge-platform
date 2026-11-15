from hashlib import sha256
from io import BytesIO
from uuid import uuid4

import pytest
from sqlalchemy import delete

from app.application.services.document_ingestion import (
    CreateDocumentRequest,
    DocumentIngestionService,
)
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.documents import (
    Document,
    DocumentVersion,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.document_ingestion import (
    PostgresDocumentIngestionRepository,
)
from app.infrastructure.database.session import SessionLocal
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


def test_initial_document_upload_persists_complete_lineage(
    tmp_path,
):
    content = (
        b"%PDF-1.4\n"
        b"GovKnowledge ingestion integration proof\n"
        b"%%EOF\n"
    )

    expected_hash = sha256(
        content
    ).hexdigest()

    storage = LocalFileStorage(
        tmp_path / "documents"
    )

    db = SessionLocal()

    result = None
    version = None
    storage_path = None

    try:
        from app.application.services.audit import AuditService
        from app.infrastructure.database.models.audit import AuditEventRecord
        from app.infrastructure.database.repositories.audit import (
            PostgresAuditRepository,
        )

        repository = PostgresDocumentIngestionRepository(db)

        audit_service = AuditService(
            repository=PostgresAuditRepository(db)
        )

        service = DocumentIngestionService(
            repository=repository,
            storage=storage,
            audit=audit_service,
        )

        unique_suffix = str(
            uuid4()
        )[:8]

        request = CreateDocumentRequest(
            title=(
                "Integration Test Policy "
                f"{unique_suffix}"
            ),
            description=(
                "Temporary GovKnowledge "
                "integration-test document."
            ),
            document_type="POLICY",
            issuing_organization=(
                "GovKnowledge Test Authority"
            ),
            jurisdiction="Burkina Faso",
            language="fr",
            publication_date=None,
            source_url=None,
            confidentiality_level="INTERNAL",
            original_filename="policy.pdf",
        )

        actor_user_id = uuid4()
        correlation_id = uuid4()

        result = service.create_initial_upload(
            request=request,
            source=BytesIO(content),
            actor_user_id=actor_user_id,
            correlation_id=correlation_id,
        )

        document = db.get(
            Document,
            result.document_id,
        )

        version = db.get(
            DocumentVersion,
            result.version_id,
        )

        job = db.get(
            DocumentProcessingJob,
            result.processing_job_id,
        )

        audit_record = db.query(AuditEventRecord).filter(
            AuditEventRecord.document_id == result.document_id,
            AuditEventRecord.event_type == "DOCUMENT_INGESTED",
        ).one_or_none()

        assert document is not None
        assert version is not None
        assert job is not None
        assert audit_record is not None
        assert audit_record.actor_user_id == actor_user_id
        assert audit_record.correlation_id == correlation_id

        assert document.id == result.document_id
        assert document.status == "ACTIVE"

        assert version.document_id == document.id
        assert version.version_number == 1
        assert version.original_filename == "policy.pdf"
        assert version.file_hash == expected_hash
        assert (
            version.processing_status
            == ProcessingStatus.PENDING.value
        )

        assert (
            job.document_version_id
            == version.id
        )
        assert (
            job.status
            == ProcessingStatus.PENDING.value
        )
        assert job.attempt_number == 1

        assert result.file_hash == expected_hash
        assert result.size_bytes == len(content)

        storage_path = storage.resolve(
            version.storage_path
        )

        assert storage_path.exists()
        assert storage_path.read_bytes() == content

    finally:
        if result is not None and version is not None:
            storage.delete(
                version.storage_path
            )

        if result is not None:
            db.rollback()

            db.execute(
                delete(
                    AuditEventRecord
                ).where(
                    AuditEventRecord.document_id
                    == result.document_id
                )
            )

            db.execute(
                delete(
                    DocumentProcessingJob
                ).where(
                    DocumentProcessingJob.id
                    == result.processing_job_id
                )
            )

            db.execute(
                delete(
                    DocumentVersion
                ).where(
                    DocumentVersion.id
                    == result.version_id
                )
            )

            db.execute(
                delete(
                    Document
                ).where(
                    Document.id
                    == result.document_id
                )
            )

            db.commit()

        db.close()


def test_ingestion_audit_failure_rolls_back_and_deletes_file(
    tmp_path,
    monkeypatch,
):
    from app.application.services.audit import (
        AuditPersistenceError,
        AuditService,
    )
    from app.infrastructure.database.models.audit import (
        AuditEventRecord,
    )
    from app.infrastructure.database.repositories.audit import (
        PostgresAuditRepository,
    )

    content = (
        b"%PDF-1.4\n"
        b"GovKnowledge atomic ingestion failure proof\n"
        b"%%EOF\n"
    )

    storage_root = tmp_path / "documents"

    storage = LocalFileStorage(
        storage_root
    )

    db = SessionLocal()

    document_id = uuid4()
    version_id = uuid4()
    processing_job_id = uuid4()

    # DocumentIngestionService generates these IDs
    # internally. Fix them so we can prove that
    # none survive the failed transaction.
    generated_ids = iter(
        (
            document_id,
            version_id,
            processing_job_id,
        )
    )

    monkeypatch.setattr(
        "app.application.services.document_ingestion.uuid4",
        lambda: next(generated_ids),
    )

    repository = (
        PostgresDocumentIngestionRepository(db)
    )

    audit_service = AuditService(
        repository=PostgresAuditRepository(db)
    )

    service = DocumentIngestionService(
        repository=repository,
        storage=storage,
        audit=audit_service,
    )

    request = CreateDocumentRequest(
        title=(
            "G30 atomic ingestion "
            f"failure {str(uuid4())[:8]}"
        ),
        description=(
            "Temporary rollback acceptance test."
        ),
        document_type="POLICY",
        issuing_organization=(
            "GovKnowledge Test Authority"
        ),
        jurisdiction="Burkina Faso",
        language="fr",
        publication_date=None,
        source_url=None,
        confidentiality_level="INTERNAL",
        original_filename="failure-proof.pdf",
    )

    actor_user_id = uuid4()
    correlation_id = uuid4()

    original_commit = db.commit

    def fail_commit():
        raise RuntimeError(
            "SIMULATED_AUDIT_COMMIT_FAILURE"
        )

    # Ingestion now only flushes. Therefore this
    # commit belongs to DOCUMENT_INGESTED audit
    # persistence.
    monkeypatch.setattr(
        db,
        "commit",
        fail_commit,
    )

    try:
        with pytest.raises(
            AuditPersistenceError,
            match=(
                "Audit event could not be persisted."
            ),
        ):
            service.create_initial_upload(
                request=request,
                source=BytesIO(content),
                actor_user_id=actor_user_id,
                correlation_id=correlation_id,
            )

        # Restore normal DB behavior before checking
        # what actually survived the rollback.
        monkeypatch.setattr(
            db,
            "commit",
            original_commit,
        )

        assert (
            db.get(Document, document_id)
            is None
        )

        assert (
            db.get(DocumentVersion, version_id)
            is None
        )

        assert (
            db.get(
                DocumentProcessingJob,
                processing_job_id,
            )
            is None
        )

        audit_record = (
            db.query(AuditEventRecord)
            .filter(
                AuditEventRecord.document_id
                == document_id
            )
            .one_or_none()
        )

        assert audit_record is None

        # The immutable-original file was written
        # before DB persistence, so the service must
        # explicitly clean it up after rollback.
        stored_files = [
            path
            for path in storage_root.rglob("*")
            if path.is_file()
        ]

        assert stored_files == []

    finally:
        db.rollback()
        db.close()
