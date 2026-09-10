from hashlib import sha256
from pathlib import Path
import sys

from sqlalchemy import func, select

from app.application.services.document_ingestion import (
    CreateDocumentRequest,
    DocumentIngestionService,
)
from app.application.services.document_processing import (
    DocumentProcessingService,
    ProcessDocumentRequest,
)
from app.core.config import get_settings
from app.domain.enums.processing import ProcessingStatus
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.document_ingestion import (
    PostgresDocumentIngestionRepository,
)
from app.infrastructure.database.repositories.document_processing import (
    PostgresDocumentProcessingRepository,
)
from app.infrastructure.database.session import SessionLocal
from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


EXPECTED_PAGE_COUNT = 54
EXPECTED_CHUNK_COUNT = 107


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m "
            "scripts.prove_real_processing_pipeline "
            "<pdf-path>"
        )

    source_pdf = Path(
        sys.argv[1]
    ).resolve()

    if not source_pdf.is_file():
        raise SystemExit(
            f"PDF not found: {source_pdf}"
        )

    settings = get_settings()

    storage = LocalFileStorage(
        Path(settings.document_storage_root)
    )

    source_hash = sha256(
        source_pdf.read_bytes()
    ).hexdigest()

    db = SessionLocal()

    try:
        ingestion_repository = (
            PostgresDocumentIngestionRepository(
                db
            )
        )

        ingestion_service = (
            DocumentIngestionService(
                repository=ingestion_repository,
                storage=storage,
            )
        )

        with source_pdf.open("rb") as source:
            created = (
                ingestion_service.create_initial_upload(
                    request=CreateDocumentRequest(
                        title=(
                            "UN Charter - "
                            "GovKnowledge E2E Proof"
                        ),
                        description=(
                            "Real document processing "
                            "pipeline validation."
                        ),
                        document_type="TREATY",
                        issuing_organization=(
                            "United Nations"
                        ),
                        jurisdiction=(
                            "International"
                        ),
                        language="en",
                        publication_date=None,
                        source_url=None,
                        confidentiality_level=(
                            "PUBLIC"
                        ),
                        original_filename=(
                            source_pdf.name
                        ),
                    ),
                    source=source,
                )
            )

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        job = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        if version is None or job is None:
            raise RuntimeError(
                "Initial ingestion records "
                "could not be loaded."
            )

        print()
        print(
            "=== Initial ingestion ==="
        )
        print(
            "Document ID:",
            created.document_id,
        )
        print(
            "Version ID:",
            created.version_id,
        )
        print(
            "Processing job ID:",
            created.processing_job_id,
        )
        print(
            "Initial status:",
            version.processing_status,
        )
        print(
            "Stored path:",
            version.storage_path,
        )

        assert (
            version.processing_status
            == ProcessingStatus.PENDING.value
        )

        assert (
            job.status
            == ProcessingStatus.PENDING.value
        )

        assert version.file_hash == source_hash
        assert created.file_hash == source_hash

        stored_path = storage.resolve(
            version.storage_path
        )

        assert stored_path.is_file()

        assert sha256(
            stored_path.read_bytes()
        ).hexdigest() == source_hash

        processing_repository = (
            PostgresDocumentProcessingRepository(
                db
            )
        )

        processing_service = (
            DocumentProcessingService(
                repository=processing_repository,
                storage=storage,
                parser=DoclingDocumentParser(
                    max_chunk_chars=1500
                ),
            )
        )

        print()
        print(
            "Starting real Docling processing..."
        )

        processed = processing_service.process(
            ProcessDocumentRequest(
                document_id=created.document_id,
                version_id=created.version_id,
                processing_job_id=(
                    created.processing_job_id
                ),
                storage_path=(
                    version.storage_path
                ),
                language="en",
            )
        )

        # Refresh state after repository commits.
        db.expire_all()

        document = db.get(
            Document,
            created.document_id,
        )

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        job = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        assert document is not None
        assert version is not None
        assert job is not None

        page_count = db.scalar(
            select(
                func.count(
                    DocumentPage.id
                )
            ).where(
                DocumentPage.document_version_id
                == created.version_id
            )
        )

        chunk_count = db.scalar(
            select(
                func.count(
                    DocumentChunk.id
                )
            ).where(
                DocumentChunk.document_version_id
                == created.version_id
            )
        )

        page_numbers = list(
            db.scalars(
                select(
                    DocumentPage.page_number
                )
                .where(
                    DocumentPage.document_version_id
                    == created.version_id
                )
                .order_by(
                    DocumentPage.page_number
                )
            )
        )

        chunks = list(
            db.scalars(
                select(
                    DocumentChunk
                )
                .where(
                    DocumentChunk.document_version_id
                    == created.version_id
                )
                .order_by(
                    DocumentChunk.chunk_index
                )
            )
        )

        pages_by_id = {
            page.id: page
            for page in db.scalars(
                select(
                    DocumentPage
                ).where(
                    DocumentPage.document_version_id
                    == created.version_id
                )
            )
        }

        assert page_count == EXPECTED_PAGE_COUNT
        assert chunk_count == EXPECTED_CHUNK_COUNT

        assert processed.page_count == page_count
        assert processed.chunk_count == chunk_count

        assert page_numbers == list(
            range(
                1,
                EXPECTED_PAGE_COUNT + 1,
            )
        )

        assert (
            version.processing_status
            == ProcessingStatus.SUCCEEDED.value
        )

        assert (
            job.status
            == ProcessingStatus.SUCCEEDED.value
        )

        assert job.started_at is not None
        assert job.completed_at is not None
        assert job.error_code is None
        assert job.error_message is None

        assert len(chunks) == EXPECTED_CHUNK_COUNT

        for expected_index, chunk in enumerate(
            chunks
        ):
            assert (
                chunk.chunk_index
                == expected_index
            )

            assert (
                chunk.document_id
                == created.document_id
            )

            assert (
                chunk.document_version_id
                == created.version_id
            )

            page = pages_by_id.get(
                chunk.page_id
            )

            assert page is not None

            assert (
                chunk.page_start
                == page.page_number
            )

            assert (
                chunk.page_end
                == page.page_number
            )

            assert (
                chunk.metadata_json[
                    "page_number"
                ]
                == page.page_number
            )

        print()
        print(
            "=== PostgreSQL proof ==="
        )
        print(
            "Pages persisted:",
            page_count,
        )
        print(
            "Chunks persisted:",
            chunk_count,
        )
        print(
            "Version status:",
            version.processing_status,
        )
        print(
            "Job status:",
            job.status,
        )
        print(
            "Page range:",
            f"{page_numbers[0]}-{page_numbers[-1]}",
        )
        print(
            "File SHA-256:",
            source_hash,
        )
        print()
        print(
            "Canonical lineage:",
            "PASS",
        )
        print(
            "Real processing pipeline:",
            "PASS",
        )

        print()
        print(
            "Records intentionally kept "
            "for independent SQL inspection."
        )
        print(
            "Document ID:",
            created.document_id,
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
