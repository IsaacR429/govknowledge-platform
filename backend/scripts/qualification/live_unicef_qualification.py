from hashlib import sha256
import json
import os
from pathlib import Path
import sys

from sqlalchemy import func, select

from app.application.services.audit import AuditService
from app.application.services.document_indexing import (
    DocumentIndexingService,
)
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
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import (
    ChunkEmbedding,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
)
from app.infrastructure.database.repositories.chunk_embeddings import (
    PostgresChunkEmbeddingRepository,
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
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
    SentenceTransformerEmbeddingProvider,
)
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


EXPECTED_PAGE_COUNT = 26

EXPECTED_DOCLING_BUNDLE_SHA256 = (
    "64253811295ecd8b2b20ed3d3b9f58a0"
    "ebefe90b1882df56bf23d5632b6f6d4d"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m "
            "scripts.live_unicef_qualification "
            "<unicef-pdf-path>"
        )

    source_pdf = Path(sys.argv[1]).resolve()

    if not source_pdf.is_file():
        raise SystemExit(
            f"PDF not found: {source_pdf}"
        )

    print()
    print("=== GovKnowledge live qualification ===")
    print("Document:", source_pdf)

    # --------------------------------------------------
    # 1. Verify frozen production parser bundle
    # --------------------------------------------------

    artifact_path_raw = os.environ.get(
        "GOVKNOWLEDGE_DOCLING_ARTIFACTS_PATH"
    )

    if not artifact_path_raw:
        raise RuntimeError(
            "GOVKNOWLEDGE_DOCLING_ARTIFACTS_PATH "
            "is not configured."
        )

    artifact_path = Path(
        artifact_path_raw
    ).resolve()

    manifest_path = (
        artifact_path / "manifest.json"
    )

    if not manifest_path.is_file():
        raise RuntimeError(
            f"Docling manifest missing: {manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    bundle_hash = manifest.get(
        "content_sha256"
    )

    if (
        bundle_hash
        != EXPECTED_DOCLING_BUNDLE_SHA256
    ):
        raise RuntimeError(
            "Unexpected Docling artifact bundle: "
            f"{bundle_hash}"
        )

    print(
        "Docling bundle:",
        bundle_hash,
    )

    # --------------------------------------------------
    # 2. Load embedding model BEFORE DB mutations
    # --------------------------------------------------

    print()
    print(
        "Loading pinned multilingual E5 model..."
    )

    embedding_provider = (
        SentenceTransformerEmbeddingProvider()
    )

    assert (
        embedding_provider.model_name
        == DEFAULT_MODEL_NAME
    )

    assert (
        embedding_provider.model_revision
        == DEFAULT_MODEL_REVISION
    )

    assert (
        embedding_provider.dimension
        == DEFAULT_DIMENSION
    )

    print(
        "Embedding model:",
        embedding_provider.model_name,
    )
    print(
        "Embedding revision:",
        embedding_provider.model_revision,
    )
    print(
        "Embedding dimension:",
        embedding_provider.dimension,
    )

    # --------------------------------------------------
    # 3. Production service wiring
    # --------------------------------------------------

    settings = get_settings()

    storage = LocalFileStorage(
        Path(
            settings.document_storage_root
        )
    )

    source_hash = sha256(
        source_pdf.read_bytes()
    ).hexdigest()

    db = SessionLocal()

    try:
        ingestion_service = (
            DocumentIngestionService(
                repository=(
                    PostgresDocumentIngestionRepository(
                        db
                    )
                ),
                storage=storage,
            )
        )

        audit_service = AuditService(
            repository=(
                PostgresAuditRepository(
                    db
                )
            )
        )

        processing_service = (
            DocumentProcessingService(
                repository=(
                    PostgresDocumentProcessingRepository(
                        db
                    )
                ),
                storage=storage,
                parser=DoclingDocumentParser(),
                audit=audit_service,
            )
        )

        indexing_service = (
            DocumentIndexingService(
                repository=(
                    PostgresChunkEmbeddingRepository(
                        db
                    )
                ),
                embedding_provider=(
                    embedding_provider
                ),
            )
        )

        # --------------------------------------------------
        # 4. Ingest UNICEF PDF
        # --------------------------------------------------

        with source_pdf.open("rb") as source:
            created = (
                ingestion_service.create_initial_upload(
                    request=CreateDocumentRequest(
                        title=(
                            "UNICEF Emergency Procedures"
                        ),
                        description=(
                            "UNICEF simplified emergency "
                            "procedures qualification document."
                        ),
                        document_type=(
                            "PROCEDURE"
                        ),
                        issuing_organization=(
                            "UNICEF"
                        ),
                        jurisdiction=(
                            "International"
                        ),
                        language="fr",
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

        print()
        print("=== Initial ingestion ===")
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
            "File SHA-256:",
            created.file_hash,
        )

        assert (
            created.file_hash
            == source_hash
        )

        # --------------------------------------------------
        # 5. Process through real Docling pipeline
        # --------------------------------------------------

        print()
        print(
            "Starting production document processing..."
        )

        processed = processing_service.process(
            ProcessDocumentRequest(
                processing_job_id=(
                    created.processing_job_id
                )
            )
        )

        db.expire_all()

        version = db.get(
            DocumentVersion,
            created.version_id,
        )

        job = db.get(
            DocumentProcessingJob,
            created.processing_job_id,
        )

        assert version is not None
        assert job is not None

        assert (
            version.processing_status
            == ProcessingStatus.SUCCEEDED.value
        )

        assert (
            job.status
            == ProcessingStatus.SUCCEEDED.value
        )

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

        assert (
            page_count
            == EXPECTED_PAGE_COUNT
        )

        assert chunk_count is not None
        assert chunk_count > 0

        assert (
            processed.page_count
            == page_count
        )

        assert (
            processed.chunk_count
            == chunk_count
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

        assert page_numbers == list(
            range(
                1,
                EXPECTED_PAGE_COUNT + 1,
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

        assert len(chunks) == chunk_count

        for expected_index, chunk in enumerate(
            chunks
        ):
            assert (
                chunk.chunk_index
                == expected_index
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
        print("=== Processing result ===")
        print(
            "Pages:",
            page_count,
        )
        print(
            "Chunks:",
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
            "Page lineage:",
            "PASS",
        )

        # --------------------------------------------------
        # 6. Index real chunks with pinned E5
        # --------------------------------------------------

        print()
        print(
            "Starting production embedding indexing..."
        )

        indexed = indexing_service.index_version(
            version_id=created.version_id
        )

        db.expire_all()

        embedding_count = db.scalar(
            select(
                func.count(
                    ChunkEmbedding.id
                )
            )
            .join(
                DocumentChunk,
                ChunkEmbedding.chunk_id
                == DocumentChunk.id,
            )
            .where(
                DocumentChunk.document_version_id
                == created.version_id
            )
        )

        assert embedding_count is not None

        assert (
            indexed.indexed_count
            == chunk_count
        )

        assert (
            embedding_count
            == chunk_count
        )

        assert (
            indexed.model_name
            == DEFAULT_MODEL_NAME
        )

        assert (
            indexed.model_revision
            == DEFAULT_MODEL_REVISION
        )

        print()
        print("=== Embedding result ===")
        print(
            "Indexed chunks:",
            indexed.indexed_count,
        )
        print(
            "Embedding rows:",
            embedding_count,
        )
        print(
            "Model:",
            indexed.model_name,
        )
        print(
            "Revision:",
            indexed.model_revision,
        )
        print(
            "Dimension:",
            embedding_provider.dimension,
        )

        print()
        print("==============================")
        print(
            "UNICEF LIVE QUALIFICATION: PASS"
        )
        print("==============================")
        print()
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
            "Records intentionally retained "
            "for retrieval/security testing."
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
