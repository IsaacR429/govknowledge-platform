from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import (
    MetaData,
    Table,
    func,
    inspect as sa_inspect,
    select,
)

from app.api.dependencies.qa import (
    get_qa_embedding_provider,
)
from app.application.services.audit import (
    AuditService,
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
from app.infrastructure.database.models.documents import (
    DocumentChunk,
    DocumentPage,
    DocumentVersion,
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
from app.infrastructure.database.session import (
    SessionLocal,
)
from app.infrastructure.parsing.modern_docling_executor import (
    ModernDoclingParserExecutor,
)
from app.infrastructure.parsing.pymupdf_executor import (
    PyMuPDFParserExecutor,
)
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)
from app.core.config import get_settings


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if is_dataclass(value):
        return json_safe(
            asdict(value)
        )

    if hasattr(value, "value"):
        try:
            return json_safe(
                value.value
            )
        except Exception:
            pass

    return str(value)


def build_executor(
    *,
    candidate: str,
    backend_root: Path,
    timeout_seconds: float,
    max_chunk_chars: int,
):
    if candidate == "pymupdf4llm":

        runtime_raw = os.environ.get(
            "GOVKNOWLEDGE_PYMUPDF_PYTHON"
        )

        if not runtime_raw:
            raise RuntimeError(
                "GOVKNOWLEDGE_PYMUPDF_PYTHON "
                "is not configured."
            )

        runtime = Path(
            runtime_raw
        ).resolve()

        worker = (
            backend_root
            / "app"
            / "infrastructure"
            / "parsing"
            / "pymupdf_worker_entry.py"
        )

        if not runtime.is_file():
            raise RuntimeError(
                f"PyMuPDF runtime missing: {runtime}"
            )

        if not worker.is_file():
            raise RuntimeError(
                f"PyMuPDF worker missing: {worker}"
            )

        return PyMuPDFParserExecutor(
            python_executable=runtime,
            worker_script=worker,
            timeout_seconds=(
                timeout_seconds
            ),
            max_chunk_chars=(
                max_chunk_chars
            ),
        )

    if candidate == "modern_docling":

        runtime_raw = os.environ.get(
            "GOVKNOWLEDGE_MODERN_DOCLING_PYTHON"
        )

        artifact_raw = os.environ.get(
            "GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT"
        )

        if not runtime_raw:
            raise RuntimeError(
                "GOVKNOWLEDGE_MODERN_DOCLING_PYTHON "
                "is not configured."
            )

        if not artifact_raw:
            raise RuntimeError(
                "GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT "
                "is not configured."
            )

        runtime = Path(
            runtime_raw
        ).resolve()

        artifact_root = Path(
            artifact_raw
        ).resolve()

        worker = (
            backend_root
            / "app"
            / "infrastructure"
            / "parsing"
            / "modern_docling_worker_entry.py"
        )

        if not runtime.is_file():
            raise RuntimeError(
                "Modern Docling runtime missing: "
                f"{runtime}"
            )

        if not artifact_root.is_dir():
            raise RuntimeError(
                "Modern Docling artifact root "
                f"missing: {artifact_root}"
            )

        if not worker.is_file():
            raise RuntimeError(
                "Modern Docling worker missing: "
                f"{worker}"
            )

        # Child worker inherits this.
        os.environ[
            "GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT"
        ] = str(
            artifact_root
        )

        return ModernDoclingParserExecutor(
            python_executable=runtime,
            worker_script=worker,
            timeout_seconds=(
                timeout_seconds
            ),
            max_chunk_chars=(
                max_chunk_chars
            ),
        )

    raise RuntimeError(
        "Candidate must be "
        "'pymupdf4llm' or "
        "'modern_docling'."
    )


def reflected_embedding_count(
    db,
    *,
    version_id,
) -> dict[str, Any]:

    bind = db.get_bind()

    metadata = MetaData()

    chunks = Table(
        "document_chunks",
        metadata,
        autoload_with=bind,
    )

    embeddings = Table(
        "chunk_embeddings",
        metadata,
        autoload_with=bind,
    )

    joined = embeddings.join(
        chunks,
        embeddings.c.chunk_id
        == chunks.c.id,
    )

    total_rows = db.scalar(
        select(
            func.count()
        )
        .select_from(joined)
        .where(
            chunks.c.document_version_id
            == version_id
        )
    )

    model_rows = db.execute(
        select(
            embeddings.c.model_name,
            embeddings.c.model_revision,
            func.count().label(
                "row_count"
            ),
        )
        .select_from(joined)
        .where(
            chunks.c.document_version_id
            == version_id
        )
        .group_by(
            embeddings.c.model_name,
            embeddings.c.model_revision,
        )
        .order_by(
            embeddings.c.model_name,
            embeddings.c.model_revision,
        )
    ).all()

    return {
        "table": "chunk_embeddings",
        "joined_via": (
            "chunk_embeddings.chunk_id "
            "-> document_chunks.id"
        ),
        "total_rows": int(
            total_rows or 0
        ),
        "models": [
            {
                "model_name": row.model_name,
                "model_revision": (
                    row.model_revision
                ),
                "row_count": int(
                    row.row_count
                ),
            }
            for row in model_rows
        ],
    }


def inspect_persisted_result(
    db,
    *,
    document_id,
    version_id,
    job_id,
    expected_page_count: int,
    max_chunk_chars: int,
) -> dict[str, Any]:

    version = db.get(
        DocumentVersion,
        version_id,
    )

    job = db.get(
        DocumentProcessingJob,
        job_id,
    )

    pages = list(
        db.scalars(
            select(
                DocumentPage
            )
            .where(
                DocumentPage.document_version_id
                == version_id
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
                == version_id
            )
            .order_by(
                DocumentChunk.chunk_index
            )
        )
    )

    page_numbers = [
        page.page_number
        for page in pages
    ]

    expected_sequence = list(
        range(
            1,
            expected_page_count + 1,
        )
    )

    chunk_lengths = [
        len(
            chunk.content or ""
        )
        for chunk in chunks
    ]

    empty_pages = [
        page.page_number
        for page in pages
        if not (
            page.extracted_text or ""
        ).strip()
    ]

    embedding_info = (
        reflected_embedding_count(
            db,
            version_id=version_id,
        )
    )

    return {
        "document_id": str(
            document_id
        ),
        "version_id": str(
            version_id
        ),
        "processing_job_id": str(
            job_id
        ),
        "version_status": (
            getattr(
                version,
                "processing_status",
                None,
            )
            if version is not None
            else None
        ),
        "job_status": (
            getattr(
                job,
                "status",
                None,
            )
            if job is not None
            else None
        ),
        "parsing_status": (
            getattr(
                job,
                "parsing_status",
                None,
            )
            if job is not None
            else None
        ),
        "failure_class": (
            getattr(
                job,
                "failure_class",
                None,
            )
            if job is not None
            else None
        ),
        "parser_lineage": (
            getattr(
                job,
                "parser_lineage_json",
                None,
            )
            if job is not None
            else None
        ),
        "page_count": len(
            pages
        ),
        "expected_page_count": (
            expected_page_count
        ),
        "page_count_matches_expected": (
            len(pages)
            == expected_page_count
        ),
        "page_number_sequence_valid": (
            page_numbers
            == expected_sequence
        ),
        "page_numbers": (
            page_numbers
        ),
        "nonempty_page_count": (
            len(pages)
            - len(empty_pages)
        ),
        "empty_page_numbers": (
            empty_pages
        ),
        "combined_page_text_chars": sum(
            len(
                page.extracted_text or ""
            )
            for page in pages
        ),
        "chunk_count": len(
            chunks
        ),
        "max_chunk_chars_observed": (
            max(
                chunk_lengths,
                default=0,
            )
        ),
        "chunks_over_limit": sum(
            1
            for length
            in chunk_lengths
            if length
            > max_chunk_chars
        ),
        "chunk_limit": (
            max_chunk_chars
        ),
        "combined_chunk_text_chars": sum(
            chunk_lengths
        ),
        "embedding_rows": (
            embedding_info
        ),
    }


REQUIRED_CASE_FIELDS = (
    "input_id",
    "label",
    "case_kind",
    "path",
    "sha256",
    "expected_page_count",
    "language",
)


def validate_manifest(
    manifest: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    str,
    float,
    int,
]:
    """
    Validate a qualification manifest without
    assuming fixed corpus size or fixed case IDs.
    """

    if not isinstance(manifest, dict):
        raise RuntimeError(
            "Manifest root must be an object."
        )

    suite_id = manifest.get(
        "suite_id"
    )

    if (
        not isinstance(suite_id, str)
        or not suite_id.strip()
    ):
        raise RuntimeError(
            "Manifest suite_id must be "
            "a non-empty string."
        )

    cases = manifest.get(
        "cases"
    )

    if (
        not isinstance(cases, list)
        or not cases
    ):
        raise RuntimeError(
            "Manifest cases must be "
            "a non-empty list."
        )

    seen_ids: set[str] = set()

    for position, case in enumerate(
        cases,
        start=1,
    ):
        if not isinstance(case, dict):
            raise RuntimeError(
                f"Manifest case {position} "
                "must be an object."
            )

        missing = [
            field
            for field in REQUIRED_CASE_FIELDS
            if field not in case
        ]

        if missing:
            raise RuntimeError(
                f"Manifest case {position} "
                f"is missing fields: {missing}"
            )

        input_id = case[
            "input_id"
        ]

        if (
            not isinstance(input_id, str)
            or not input_id.strip()
        ):
            raise RuntimeError(
                f"Manifest case {position} "
                "has invalid input_id."
            )

        if input_id in seen_ids:
            raise RuntimeError(
                "Manifest contains duplicate "
                f"input_id: {input_id}"
            )

        seen_ids.add(
            input_id
        )

        for field in (
            "label",
            "case_kind",
            "path",
            "language",
        ):
            value = case[field]

            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise RuntimeError(
                    f"{input_id}: "
                    f"{field} must be a "
                    "non-empty string."
                )

        sha256_value = case[
            "sha256"
        ]

        if (
            not isinstance(
                sha256_value,
                str,
            )
            or len(sha256_value) != 64
            or any(
                character
                not in "0123456789abcdefABCDEF"
                for character
                in sha256_value
            )
        ):
            raise RuntimeError(
                f"{input_id}: invalid SHA-256."
            )

        page_count = case[
            "expected_page_count"
        ]

        if (
            isinstance(page_count, bool)
            or not isinstance(
                page_count,
                int,
            )
            or page_count <= 0
        ):
            raise RuntimeError(
                f"{input_id}: "
                "expected_page_count must "
                "be a positive integer."
            )

    timeout_raw = manifest.get(
        "parser_timeout_seconds"
    )

    try:
        timeout_seconds = float(
            timeout_raw
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "parser_timeout_seconds must "
            "be numeric."
        ) from exc

    if timeout_seconds <= 0:
        raise RuntimeError(
            "parser_timeout_seconds must "
            "be greater than zero."
        )

    max_chunk_chars = manifest.get(
        "max_chunk_chars"
    )

    if (
        isinstance(max_chunk_chars, bool)
        or not isinstance(
            max_chunk_chars,
            int,
        )
        or max_chunk_chars <= 0
    ):
        raise RuntimeError(
            "max_chunk_chars must be "
            "a positive integer."
        )

    return (
        cases,
        suite_id.strip(),
        timeout_seconds,
        max_chunk_chars,
    )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--candidate",
        required=True,
        choices=(
            "pymupdf4llm",
            "modern_docling",
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    manifest_path = Path(
        args.manifest
    ).resolve()

    output_path = Path(
        args.output
    ).resolve()

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    (
        cases,
        suite_id,
        timeout_seconds,
        max_chunk_chars,
    ) = validate_manifest(
        manifest
    )

    backend_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    executor = build_executor(
        candidate=args.candidate,
        backend_root=backend_root,
        timeout_seconds=(
            timeout_seconds
        ),
        max_chunk_chars=(
            max_chunk_chars
        ),
    )

    settings = get_settings()

    storage = LocalFileStorage(
        Path(
            settings.document_storage_root
        )
    )

    # Load the real pinned embedding provider
    # once for the entire candidate suite.
    embedding_provider = (
        get_qa_embedding_provider()
    )

    suite_start = time.perf_counter()

    results = []

    print("=" * 88)
    print(
        "GOVKNOWLEDGE — "
        "FULL CORPUS APPLICATION E2E"
    )
    print("=" * 88)
    print(
        "Candidate:",
        args.candidate,
    )
    print(
        "Cases:",
        len(cases),
    )
    print(
        "Parser timeout:",
        timeout_seconds,
    )
    print(
        "Chunk limit:",
        max_chunk_chars,
    )

    for position, case in enumerate(
        cases,
        start=1,
    ):

        case_start = (
            time.perf_counter()
        )

        path = Path(
            case["path"]
        ).resolve()

        print()
        print("=" * 88)
        print(
            f"[{position:02d}/"
            f"{len(cases):02d}] "
            f"{case['input_id']} — "
            f"{case['label']}"
        )
        print("=" * 88)

        source_before = (
            sha256_path(path)
        )

        result: dict[str, Any] = {
            "input_id": (
                case["input_id"]
            ),
            "label": case["label"],
            "case_kind": (
                case["case_kind"]
            ),
            "path": str(path),
            "language": (
                case["language"]
            ),
            "expected_page_count": (
                case[
                    "expected_page_count"
                ]
            ),
            "source_sha256_before": (
                source_before
            ),
            "source_sha256_expected": (
                case["sha256"]
            ),
            "source_hash_matches_manifest": (
                source_before
                == case["sha256"]
            ),
            "candidate": (
                args.candidate
            ),
        }

        if not result[
            "source_hash_matches_manifest"
        ]:
            result[
                "case_outcome"
            ] = "SOURCE_HASH_MISMATCH"

            results.append(
                result
            )

            print(
                "STOP CASE: source hash mismatch"
            )

            continue

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
                    audit=audit,
                    parser_executor=(
                        executor
                    ),
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

            with path.open(
                "rb"
            ) as source:

                created = (
                    ingestion_service
                    .create_initial_upload(
                        request=(
                            CreateDocumentRequest(
                                title=(
                                    "Full Corpus "
                                    f"{case['input_id']} "
                                    f"{case['label']} "
                                    f"[{args.candidate}]"
                                ),
                                description=(
                                    "GovKnowledge "
                                    "full-corpus "
                                    "pre-UI parser "
                                    "qualification."
                                ),
                                document_type=(
                                    "POLICY"
                                ),
                                issuing_organization=(
                                    "GovKnowledge "
                                    "Qualification"
                                ),
                                jurisdiction=(
                                    "Qualification"
                                ),
                                language=(
                                    case[
                                        "language"
                                    ]
                                ),
                                publication_date=None,
                                source_url=None,
                                confidentiality_level=(
                                    "PUBLIC"
                                ),
                                original_filename=(
                                    path.name
                                ),
                            )
                        ),
                        source=source,
                    )
                )

            result[
                "document_id"
            ] = str(
                created.document_id
            )

            result[
                "version_id"
            ] = str(
                created.version_id
            )

            result[
                "processing_job_id"
            ] = str(
                created.processing_job_id
            )

            processing_start = (
                time.perf_counter()
            )

            processing_exception = None

            try:
                processed = (
                    processing_service
                    .process(
                        ProcessDocumentRequest(
                            processing_job_id=(
                                created
                                .processing_job_id
                            )
                        )
                    )
                )

                result[
                    "processing_result"
                ] = json_safe(
                    processed
                )

            except Exception as exc:
                processing_exception = (
                    exc
                )

                result[
                    "processing_exception"
                ] = {
                    "type": (
                        type(exc).__name__
                    ),
                    "message": str(
                        exc
                    ),
                }

                # Restore session usability.
                db.rollback()

            result[
                "processing_wall_clock_ms"
            ] = (
                (
                    time.perf_counter()
                    - processing_start
                )
                * 1000.0
            )

            # Read committed parser outcome from
            # a fresh transaction/session state.
            db.expire_all()

            persisted = (
                inspect_persisted_result(
                    db,
                    document_id=(
                        created.document_id
                    ),
                    version_id=(
                        created.version_id
                    ),
                    job_id=(
                        created.processing_job_id
                    ),
                    expected_page_count=(
                        case[
                            "expected_page_count"
                        ]
                    ),
                    max_chunk_chars=(
                        max_chunk_chars
                    ),
                )
            )

            result.update(
                persisted
            )

            # Index only if document processing
            # actually produced chunks.
            if (
                processing_exception
                is None
                and persisted[
                    "chunk_count"
                ]
                > 0
            ):
                indexing_start = (
                    time.perf_counter()
                )

                try:
                    indexed = (
                        indexing_service
                        .index_version(
                            version_id=(
                                created
                                .version_id
                            )
                        )
                    )

                    result[
                        "indexing_result"
                    ] = json_safe(
                        indexed
                    )

                    result[
                        "indexing_exception"
                    ] = None

                except Exception as exc:
                    result[
                        "indexing_exception"
                    ] = {
                        "type": (
                            type(exc).__name__
                        ),
                        "message": str(
                            exc
                        ),
                    }

                    db.rollback()

                result[
                    "indexing_wall_clock_ms"
                ] = (
                    (
                        time.perf_counter()
                        - indexing_start
                    )
                    * 1000.0
                )

                db.expire_all()

                result[
                    "embedding_rows"
                ] = (
                    reflected_embedding_count(
                        db,
                        version_id=(
                            created.version_id
                        ),
                    )
                )

            source_after = (
                sha256_path(path)
            )

            result[
                "source_sha256_after"
            ] = (
                source_after
            )

            result[
                "source_unchanged"
            ] = (
                source_before
                == source_after
            )

            result[
                "case_outcome"
            ] = (
                "PROCESSING_SUCCEEDED"
                if processing_exception
                is None
                else "PROCESSING_FAILED"
            )

        except Exception as exc:
            try:
                db.rollback()
            except Exception:
                pass

            result[
                "harness_exception"
            ] = {
                "type": (
                    type(exc).__name__
                ),
                "message": str(
                    exc
                ),
            }

            result[
                "case_outcome"
            ] = (
                "HARNESS_ERROR"
            )

        finally:
            db.close()

        result[
            "case_wall_clock_ms"
        ] = (
            (
                time.perf_counter()
                - case_start
            )
            * 1000.0
        )

        results.append(
            result
        )

        print(
            "Outcome:",
            result.get(
                "case_outcome"
            ),
        )

        print(
            "Pages:",
            result.get(
                "page_count"
            ),
            "/",
            case[
                "expected_page_count"
            ],
        )

        print(
            "Chunks:",
            result.get(
                "chunk_count"
            ),
        )

        print(
            "Max chunk chars:",
            result.get(
                "max_chunk_chars_observed"
            ),
        )

        print(
            "Chunks > limit:",
            result.get(
                "chunks_over_limit"
            ),
        )

        print(
            "Failure class:",
            json_safe(
                result.get(
                    "failure_class"
                )
            ),
        )

        # Persist after every case so a later
        # timeout/crash cannot erase earlier data.
        interim = {
            "schema_version": 1,
            "suite_id": suite_id,
            "candidate": (
                args.candidate
            ),
            "manifest": str(
                manifest_path
            ),
            "started_at_utc": None,
            "completed": False,
            "results": results,
        }

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path.write_text(
            json.dumps(
                json_safe(
                    interim
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    manifest_input_ids = [
        case["input_id"]
        for case in cases
    ]

    result_input_ids = [
        item.get("input_id")
        for item in results
    ]

    if len(results) != len(cases):
        raise RuntimeError(
            "Qualification result count "
            "does not match manifest "
            f"case count: "
            f"{len(results)} != "
            f"{len(cases)}"
        )

    if (
        result_input_ids
        != manifest_input_ids
    ):
        raise RuntimeError(
            "Qualification result IDs/order "
            "do not exactly match manifest. "
            f"manifest={manifest_input_ids}, "
            f"results={result_input_ids}"
        )

    suite_ms = (
        (
            time.perf_counter()
            - suite_start
        )
        * 1000.0
    )

    summary = {
        "case_count": len(
            results
        ),
        "processing_succeeded": sum(
            1
            for item in results
            if item.get(
                "case_outcome"
            )
            == "PROCESSING_SUCCEEDED"
        ),
        "processing_failed": sum(
            1
            for item in results
            if item.get(
                "case_outcome"
            )
            == "PROCESSING_FAILED"
        ),
        "harness_errors": sum(
            1
            for item in results
            if item.get(
                "case_outcome"
            )
            == "HARNESS_ERROR"
        ),
        "page_count_matches": sum(
            1
            for item in results
            if item.get(
                "page_count_matches_expected"
            )
            is True
        ),
        "chunk_limit_violations": sum(
            int(
                item.get(
                    "chunks_over_limit",
                    0,
                )
                or 0
            )
            for item in results
        ),
        "source_changed_cases": sum(
            1
            for item in results
            if item.get(
                "source_unchanged"
            )
            is False
        ),
        "suite_wall_clock_ms": (
            suite_ms
        ),
    }

    final_payload = {
        "schema_version": 1,
        "suite_id": suite_id,
        "candidate": (
            args.candidate
        ),
        "manifest": str(
            manifest_path
        ),
        "completed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "completed": True,
        "summary": summary,
        "results": results,
    }

    output_path.write_text(
        json.dumps(
            json_safe(
                final_payload
            ),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 88)
    print(
        "FULL CORPUS APPLICATION E2E "
        "— COMPLETE"
    )
    print("=" * 88)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print(
        "Result:",
        output_path,
    )

    if summary["harness_errors"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
