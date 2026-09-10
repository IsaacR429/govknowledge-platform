from app.application.ports.document_ingestion_repository import (
    DocumentIngestionRepository,
    InitialDocumentUpload,
)
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.documents import (
    Document,
    DocumentVersion,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from sqlalchemy.orm import Session


class PostgresDocumentIngestionRepository(
    DocumentIngestionRepository
):
    def __init__(
        self,
        db: Session,
    ) -> None:
        self._db = db

    def create_initial_upload(
        self,
        upload: InitialDocumentUpload,
    ) -> None:
        document = Document(
            id=upload.document_id,
            title=upload.title,
            description=upload.description,
            document_type=upload.document_type,
            issuing_organization=(
                upload.issuing_organization
            ),
            jurisdiction=upload.jurisdiction,
            language=upload.language,
            publication_date=(
                upload.publication_date
            ),
            source_url=upload.source_url,
            status="ACTIVE",
            confidentiality_level=(
                upload.confidentiality_level
            ),
        )

        version = DocumentVersion(
            id=upload.version_id,
            document_id=upload.document_id,
            version_number=1,
            original_filename=(
                upload.original_filename
            ),
            storage_path=upload.storage_path,
            file_hash=upload.file_hash,
            processing_status=(
                ProcessingStatus.PENDING.value
            ),
        )

        job = DocumentProcessingJob(
            id=upload.processing_job_id,
            document_version_id=upload.version_id,
            status=ProcessingStatus.PENDING.value,
            attempt_number=1,
        )

        self._db.add(document)
        self._db.add(version)
        self._db.flush()
        self._db.add(job)

        try:
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise
