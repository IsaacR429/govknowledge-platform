from __future__ import annotations

from datetime import date
from hashlib import sha256
import math
import os
from pathlib import Path
import secrets
import time

from pwdlib import PasswordHash
from sqlalchemy import func, select

# Qualification must use only frozen local model artifacts.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault(
    "GOVKNOWLEDGE_DOCLING_ARTIFACTS_PATH",
    str(
        Path.home()
        / ".govknowledge"
        / "artifacts"
        / "docling-p0"
    ),
)

from app.api.dependencies.qa import (
    get_qa_embedding_provider,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.audit import (
    AuditService,
)
from app.application.services.authorization import (
    AuthorizationService,
)
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
from app.domain.enums.permission import (
    PermissionAction,
)
from app.domain.enums.processing import (
    ProcessingStatus,
)
from app.infrastructure.database.models.authentication import (
    LocalCredential,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import (
    ChunkEmbedding,
)
from app.infrastructure.database.models.identity import (
    User,
)
from app.infrastructure.database.models.permissions import (
    DocumentPermission,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.repositories.audit import (
    PostgresAuditRepository,
)
from app.infrastructure.database.repositories.authorization import (
    PostgresAuthorizationRepository,
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
from app.infrastructure.database.session import (
    SessionLocal,
)
from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


EXPECTED_PAGE_COUNT = 34

AUTHORIZATION_TARGET_MS = 500.0

WARMUP_RUNS = 20
MEASURED_RUNS = 200


def file_sha256(
    path: Path,
) -> str:
    digest = sha256()

    with path.open("rb") as source:
        while True:
            block = source.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def percentile(
    values: list[float],
    percent: float,
) -> float:
    if not values:
        raise ValueError(
            "values cannot be empty"
        )

    ordered = sorted(values)

    position = (
        (len(ordered) - 1)
        * percent
        / 100.0
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def summarize(
    name: str,
    values: list[float],
    target_ms: float,
) -> float:
    ordered = sorted(values)

    minimum = ordered[0]
    maximum = ordered[-1]

    median = percentile(
        values,
        50.0,
    )

    p95 = percentile(
        values,
        95.0,
    )

    p99 = percentile(
        values,
        99.0,
    )

    print()
    print(
        f"===== {name} ====="
    )
    print(
        "Measured requests:",
        len(values),
    )
    print(
        f"Minimum: {minimum:.3f} ms"
    )
    print(
        f"Median:  {median:.3f} ms"
    )
    print(
        f"P95:     {p95:.3f} ms"
    )
    print(
        f"P99:     {p99:.3f} ms"
    )
    print(
        f"Maximum: {maximum:.3f} ms"
    )
    print(
        f"Target:  P95 <= "
        f"{target_ms:.3f} ms"
    )

    status = (
        "PASS"
        if p95 <= target_ms
        else "FAIL"
    )

    print(
        "Result:",
        status,
    )

    return p95


def measure(
    operation,
    *,
    warmup_runs: int,
    measured_runs: int,
) -> list[float]:
    for _ in range(
        warmup_runs
    ):
        operation()

    measurements = []

    for _ in range(
        measured_runs
    ):
        started = (
            time.perf_counter_ns()
        )

        operation()

        finished = (
            time.perf_counter_ns()
        )

        elapsed_ms = (
            finished - started
        ) / 1_000_000.0

        measurements.append(
            elapsed_ms
        )

    return measurements


def main() -> None:
    repo_root = (
        Path(__file__)
        .resolve()
        .parents[3]
    )

    source_pdf = (
        repo_root
        / "documents"
        / "qualification"
        / "Carta_NU.pdf"
    )

    if not source_pdf.exists():
        raise FileNotFoundError(
            source_pdf
        )

    source_hash = file_sha256(
        source_pdf
    )

    print(
        "===== GOVKNOWLEDGE PERFORMANCE QUALIFICATION ====="
    )
    print(
        "Source:",
        source_pdf,
    )
    print(
        "Source SHA-256:",
        source_hash,
    )
    print(
        "Warm-up runs:",
        WARMUP_RUNS,
    )
    print(
        "Measured runs:",
        MEASURED_RUNS,
    )

    settings = get_settings()

    storage = LocalFileStorage(
        Path(
            settings.document_storage_root
        )
    )

    db = SessionLocal()

    try:
        audit = AuditService(
            repository=(
                PostgresAuditRepository(
                    db
                )
            )
        )

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

        processing_service = (
            DocumentProcessingService(
                repository=(
                    PostgresDocumentProcessingRepository(
                        db
                    )
                ),
                storage=storage,
                parser=(
                    DoclingDocumentParser()
                ),
                audit=audit,
            )
        )

        print()
        print(
            "===== COLD / SETUP ====="
        )

        setup_started = (
            time.perf_counter()
        )

        with source_pdf.open(
            "rb"
        ) as source:
            created = (
                ingestion_service
                .create_initial_upload(
                    request=(
                        CreateDocumentRequest(
                            title=(
                                "Charte des "
                                "Nations Unies"
                            ),
                            description=(
                                "GovKnowledge "
                                "performance "
                                "qualification "
                                "document."
                            ),
                            document_type=(
                                "POLICY"
                            ),
                            issuing_organization=(
                                "Nations Unies"
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
                        )
                    ),
                    source=source,
                )
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
            "File SHA-256:",
            created.file_hash,
        )

        assert (
            created.file_hash
            == source_hash
        )

        document = db.get(
            Document,
            created.document_id,
        )

        assert document is not None
        assert (
            document.status
            == "ACTIVE"
        )

        processing_started = (
            time.perf_counter()
        )

        processed = (
            processing_service.process(
                ProcessDocumentRequest(
                    processing_job_id=(
                        created.processing_job_id
                    )
                )
            )
        )

        processing_seconds = (
            time.perf_counter()
            - processing_started
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

        print(
            "Pages:",
            page_count,
        )
        print(
            "Chunks:",
            chunk_count,
        )
        print(
            "Document processing:",
            f"{processing_seconds:.3f} s",
        )

        print()
        print(
            "Loading frozen E5 "
            "embedding provider..."
        )

        embedding_load_started = (
            time.perf_counter()
        )

        embedding_provider = (
            get_qa_embedding_provider()
        )

        embedding_load_seconds = (
            time.perf_counter()
            - embedding_load_started
        )

        print(
            "Embedding model:",
            DEFAULT_MODEL_NAME,
        )
        print(
            "Embedding revision:",
            DEFAULT_MODEL_REVISION,
        )
        print(
            "Cold model load:",
            f"{embedding_load_seconds:.3f} s",
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

        indexing_started = (
            time.perf_counter()
        )

        indexed = (
            indexing_service
            .index_version(
                version_id=(
                    created.version_id
                )
            )
        )

        indexing_seconds = (
            time.perf_counter()
            - indexing_started
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

        print(
            "Indexed chunks:",
            indexed.indexed_count,
        )
        print(
            "Indexing time:",
            f"{indexing_seconds:.3f} s",
        )

        username = (
            "govknowledge-performance-"
            f"{created.document_id.hex[:8]}"
        )

        user = User(
            username=username,
            display_name=(
                "GovKnowledge "
                "Performance User"
            ),
            email=None,
            department=(
                "Qualification"
            ),
            language_preference="fr",
            is_active=True,
        )

        db.add(user)
        db.flush()

        qualification_password = (
            secrets.token_urlsafe(
                24
            )
        )

        password_hasher = (
            PasswordHash.recommended()
        )

        credential = LocalCredential(
            user_id=user.id,
            password_hash=(
                password_hasher.hash(
                    qualification_password
                )
            ),
        )

        db.add(credential)

        permission = (
            DocumentPermission(
                user_id=user.id,
                document_id=(
                    created.document_id
                ),
                action=(
                    PermissionAction
                    .READ
                    .value
                ),
            )
        )

        db.add(permission)
        db.commit()

        identity = CurrentIdentity(
            user_id=user.id,
            username=user.username,
            display_name=(
                user.display_name
            ),
            email=user.email,
            department=user.department,
            language_preference=(
                user.language_preference
            ),
            roles=(),
        )

        authorization = (
            AuthorizationService(
                PostgresAuthorizationRepository(
                    db
                )
            )
        )

        # Correctness before timing.
        scope = (
            authorization
            .retrieval_scope(
                identity
            )
        )

        assert (
            scope.document_ids
            == frozenset(
                {
                    created.document_id,
                }
            )
        )

        authorization.require_document_read(
            identity,
            created.document_id,
        )

        total_setup_seconds = (
            time.perf_counter()
            - setup_started
        )

        print()
        print(
            "Setup completed:",
            f"{total_setup_seconds:.3f} s",
        )

        print()
        print(
            "===== AUTHORIZATION "
            "STEADY-STATE ====="
        )

        scope_timings = measure(
            lambda: (
                authorization
                .retrieval_scope(
                    identity
                )
            ),
            warmup_runs=WARMUP_RUNS,
            measured_runs=MEASURED_RUNS,
        )

        read_timings = measure(
            lambda: (
                authorization
                .require_document_read(
                    identity,
                    created.document_id,
                )
            ),
            warmup_runs=WARMUP_RUNS,
            measured_runs=MEASURED_RUNS,
        )

        scope_p95 = summarize(
            (
                "AUTHORIZATION "
                "RETRIEVAL SCOPE"
            ),
            scope_timings,
            AUTHORIZATION_TARGET_MS,
        )

        read_p95 = summarize(
            (
                "AUTHORIZATION "
                "DOCUMENT READ"
            ),
            read_timings,
            AUTHORIZATION_TARGET_MS,
        )

        worst_p95 = max(
            scope_p95,
            read_p95,
        )

        print()
        print(
            "===== AUTHORIZATION GATE ====="
        )
        print(
            "Worst P95:",
            f"{worst_p95:.3f} ms",
        )
        print(
            "Required:",
            (
                f"<= "
                f"{AUTHORIZATION_TARGET_MS:.0f} ms"
            ),
        )

        if (
            worst_p95
            <= AUTHORIZATION_TARGET_MS
        ):
            print(
                "AUTHORIZATION PERFORMANCE: PASS"
            )
        else:
            print(
                "AUTHORIZATION PERFORMANCE: FAIL"
            )
            raise AssertionError(
                "Authorization P95 exceeded "
                "qualification target."
            )

    finally:
        db.close()


if __name__ == "__main__":
    main()
