from hashlib import sha256
from io import BytesIO
from uuid import uuid4

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
        repository = (
            PostgresDocumentIngestionRepository(
                db
            )
        )

        service = DocumentIngestionService(
            repository=repository,
            storage=storage,
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

        result = service.create_initial_upload(
            request=request,
            source=BytesIO(content),
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

        assert document is not None
        assert version is not None
        assert job is not None

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
