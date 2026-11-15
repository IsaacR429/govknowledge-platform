from datetime import date, datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies.administration import (
    require_admin_identity,
)
from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.authorization import (
    get_authorization_service,
)
from app.api.dependencies.document_processing import (
    _claim_indexing_retry,
    get_routed_parser_executor,
    run_document_indexing_retry,
    run_document_processing_job,
)
from app.api.dependencies.documents import (
    get_document_ingestion_service,
)
from app.api.schemas.documents import (
    AccessibleDocumentDetailResponse,
    AccessibleDocumentItem,
    DocumentArchiveResponse,
    DocumentDetailResponse,
    DocumentInventoryItem,
    DocumentMetadataPatch,
    DocumentMetadataUpdateResponse,
    DocumentUploadResponse,
    DocumentVersionSummary,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.domain.enums.audit import (
    AuditEventType,
    AuditOutcome,
)
from app.domain.exceptions.authorization import (
    AuthorizationDeniedError,
)
from app.infrastructure.database.models.audit import (
    AuditEventRecord,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.parser_execution import (
    ParserExecutor,
)
from app.application.services.document_ingestion import (
    CreateDocumentRequest,
    DocumentIngestionService,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentChunk,
    DocumentVersion,
)
from app.infrastructure.database.models.embeddings import ChunkEmbedding
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.session import get_db
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_REVISION,
)


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


def _required_text(
    *,
    field_name: str,
    value: str,
) -> str:
    cleaned = value.strip()

    if not cleaned:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"{field_name} cannot be blank.",
        )

    return cleaned


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None

    cleaned = value.strip()
    return cleaned or None


def _audit_json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()

    if isinstance(value, UUID):
        return str(value)

    return value


def _version_summary(
    version: DocumentVersion,
) -> DocumentVersionSummary:
    return DocumentVersionSummary(
        version_id=version.id,
        version_number=version.version_number,
        original_filename=version.original_filename,
        processing_status=version.processing_status,
        effective_date=version.effective_date,
        superseded_at=version.superseded_at,
        created_at=version.created_at,
    )


def _document_detail(
    document: Document,
    versions: list[DocumentVersion],
) -> DocumentDetailResponse:
    ordered_versions = sorted(
        versions,
        key=lambda item: item.version_number,
        reverse=True,
    )

    latest_version = (
        _version_summary(ordered_versions[0])
        if ordered_versions
        else None
    )

    return DocumentDetailResponse(
        document_id=document.id,
        title=document.title,
        description=document.description,
        document_type=document.document_type,
        issuing_organization=document.issuing_organization,
        jurisdiction=document.jurisdiction,
        language=document.language,
        publication_date=document.publication_date,
        source_url=document.source_url,
        status=document.status,
        confidentiality_level=document.confidentiality_level,
        created_at=document.created_at,
        updated_at=document.updated_at,
        latest_version=latest_version,
        versions=[
            _version_summary(version)
            for version in ordered_versions
        ],
    )


def _commit_document_audit(
    db: Session,
    *,
    event_type: AuditEventType,
    actor_id: UUID,
    document_id: UUID,
    metadata: dict,
) -> None:
    db.add(
        AuditEventRecord(
            event_type=event_type.value,
            outcome=AuditOutcome.SUCCEEDED.value,
            actor_user_id=actor_id,
            resource_type="DOCUMENT",
            resource_id=document_id,
            document_id=document_id,
            metadata_json=metadata,
        )
    )

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    background_tasks: BackgroundTasks,
    file: Annotated[
        UploadFile,
        File(),
    ],
    title: Annotated[
        str,
        Form(),
    ],
    issuing_organization: Annotated[
        str,
        Form(),
    ],
    jurisdiction: Annotated[
        str,
        Form(),
    ],
    language: Annotated[
        str,
        Form(),
    ],
    confidentiality_level: Annotated[
        str,
        Form(),
    ],
    identity: Annotated[
        CurrentIdentity,
        Depends(require_admin_identity),
    ],
    ingestion_service: Annotated[
        DocumentIngestionService,
        Depends(
            get_document_ingestion_service
        ),
    ],
    parser_executor: Annotated[
        ParserExecutor,
        Depends(get_routed_parser_executor),
    ],
    description: Annotated[
        str | None,
        Form(),
    ] = None,
    document_type: Annotated[
        str | None,
        Form(),
    ] = None,
    publication_date: Annotated[
        date | None,
        Form(),
    ] = None,
    source_url: Annotated[
        str | None,
        Form(),
    ] = None,
) -> DocumentUploadResponse:
    original_filename = Path(
        file.filename or ""
    ).name

    if not original_filename:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail="Uploaded file must have a filename.",
        )

    if not original_filename.lower().endswith(
        ".pdf"
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
            ),
            detail="Only PDF documents are supported.",
        )

    file.file.seek(0)

    correlation_id = uuid4()

    result = (
        ingestion_service.create_initial_upload(
            request=CreateDocumentRequest(
                title=_required_text(
                    field_name="title",
                    value=title,
                ),
                description=(
                    description.strip()
                    if description
                    else None
                ),
                document_type=(
                    document_type.strip()
                    if document_type
                    else None
                ),
                issuing_organization=(
                    _required_text(
                        field_name=(
                            "issuing_organization"
                        ),
                        value=(
                            issuing_organization
                        ),
                    )
                ),
                jurisdiction=_required_text(
                    field_name="jurisdiction",
                    value=jurisdiction,
                ),
                language=_required_text(
                    field_name="language",
                    value=language,
                ),
                publication_date=(
                    publication_date
                ),
                source_url=(
                    source_url.strip()
                    if source_url
                    else None
                ),
                confidentiality_level=(
                    _required_text(
                        field_name=(
                            "confidentiality_level"
                        ),
                        value=(
                            confidentiality_level
                        ),
                    )
                ),
                original_filename=(
                    original_filename
                ),
            ),
            source=file.file,
            actor_user_id=identity.user_id,
            correlation_id=correlation_id,
            content_type=(
                file.content_type
                or "application/pdf"
            ),
        )
    )

    background_tasks.add_task(
        run_document_processing_job,
        result.processing_job_id,
        parser_executor,
        identity.user_id,
        correlation_id,
    )

    return DocumentUploadResponse(
        document_id=result.document_id,
        version_id=result.version_id,
        processing_job_id=(
            result.processing_job_id
        ),
        file_hash=result.file_hash,
        size_bytes=result.size_bytes,
    )


@router.get("/jobs/{processing_job_id}")
def read_document_processing_job(
    processing_job_id: UUID,
    identity: Annotated[
        CurrentIdentity,
        Depends(require_admin_identity),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    del identity

    job = db.get(
        DocumentProcessingJob,
        processing_job_id,
    )
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Processing job not found.",
        )

    version = db.get(
        DocumentVersion,
        job.document_version_id,
    )
    if version is None:
        raise HTTPException(
            status_code=404,
            detail="Processing job not found.",
        )

    document = db.get(Document, version.document_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Processing job not found.",
        )

    chunk_count = db.scalar(
        select(func.count(DocumentChunk.id)).where(
            DocumentChunk.document_version_id == version.id
        )
    ) or 0

    indexed_count = db.scalar(
        select(func.count(ChunkEmbedding.id))
        .join(DocumentChunk, ChunkEmbedding.chunk_id == DocumentChunk.id)
        .where(
            DocumentChunk.document_version_id == version.id,
            ChunkEmbedding.model_name == DEFAULT_MODEL_NAME,
            ChunkEmbedding.model_revision == DEFAULT_MODEL_REVISION,
        )
    ) or 0

    searchable = (
        func.coalesce(
            DocumentChunk.metadata_json.op("->>")(
                "evidence_type"
            ),
            "",
        ) != "vector_marker_table"
    )

    searchable_chunk_count = db.scalar(
        select(func.count(DocumentChunk.id)).where(
            DocumentChunk.document_version_id == version.id,
            searchable,
        )
    ) or 0

    searchable_indexed_count = db.scalar(
        select(func.count(ChunkEmbedding.id))
        .join(
            DocumentChunk,
            ChunkEmbedding.chunk_id == DocumentChunk.id,
        )
        .where(
            DocumentChunk.document_version_id == version.id,
            ChunkEmbedding.model_name == DEFAULT_MODEL_NAME,
            ChunkEmbedding.model_revision == DEFAULT_MODEL_REVISION,
            searchable,
        )
    ) or 0

    if document.status != "ACTIVE":
        readiness = "ARCHIVED"
    elif version.processing_status != "SUCCEEDED":
        readiness = version.processing_status
    elif searchable_chunk_count == 0:
        readiness = "NO_SEARCHABLE_CONTENT"
    elif searchable_indexed_count < searchable_chunk_count:
        readiness = "INDEXING_INCOMPLETE"
    else:
        readiness = "SEARCH_READY"

    return {
        "processing_job_id": job.id,
        "version_id": version.id,
        "document_id": version.document_id,
        "processing_status": job.status,
        "version_processing_status": version.processing_status,
        "attempt_number": job.attempt_number,
        "parsing_status": job.parsing_status,
        "failure_class": job.failure_class,
        "error_code": job.error_code,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "parser_lineage": job.parser_lineage_json,
        "indexing_status": job.indexing_status,
        "indexing_attempt_number": job.indexing_attempt_number,
        "indexing_error_code": job.indexing_error_code,
        "readiness_status": readiness,
        "chunk_count": chunk_count,
        "indexed_count": indexed_count,
        "searchable_chunk_count": searchable_chunk_count,
        "searchable_indexed_count": searchable_indexed_count,
        "ready_for_authorized_users": readiness == "SEARCH_READY",
        "requires_document_read_permission": True,
    }


@router.post(
    "/jobs/{processing_job_id}/retry-indexing",
    status_code=status.HTTP_202_ACCEPTED,
)
def retry_document_indexing(
    processing_job_id: UUID,
    background_tasks: BackgroundTasks,
    identity: Annotated[
        CurrentIdentity,
        Depends(require_admin_identity),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    del identity

    job = db.get(DocumentProcessingJob, processing_job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Processing job not found.",
        )

    version = db.get(DocumentVersion, job.document_version_id)
    document = (
        db.get(Document, version.document_id)
        if version is not None
        else None
    )
    if version is None or document is None:
        raise HTTPException(
            status_code=404,
            detail="Processing job not found.",
        )

    if (
        document.status != "ACTIVE"
        or version.processing_status != "SUCCEEDED"
    ):
        raise HTTPException(
            status_code=409,
            detail="Document version is not eligible for indexing retry.",
        )

    # Atomically change FAILED -> INDEXING. Only one request can claim it.
    version_id = _claim_indexing_retry(processing_job_id)
    if version_id is None:
        raise HTTPException(
            status_code=409,
            detail="Indexing job is not in a retryable FAILED state.",
        )

    background_tasks.add_task(
        run_document_indexing_retry,
        processing_job_id,
        version_id,
    )

    return {
        "processing_job_id": processing_job_id,
        "version_id": version_id,
        "indexing_status": "INDEXING",
        "retry_accepted": True,
    }


@router.get(
    "",
    response_model=list[DocumentInventoryItem],
)
def list_documents(
    identity: Annotated[
        CurrentIdentity,
        Depends(require_admin_identity),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> list[DocumentInventoryItem]:
    del identity

    documents = db.scalars(
        select(Document).order_by(
            Document.created_at.desc()
        )
    ).all()

    if not documents:
        return []

    document_ids = [
        document.id
        for document in documents
    ]

    versions = db.scalars(
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id.in_(
                document_ids
            )
        )
        .order_by(
            DocumentVersion.document_id,
            DocumentVersion.version_number.desc(),
        )
    ).all()

    latest_by_document: dict[UUID, DocumentVersion] = {}

    for version in versions:
        latest_by_document.setdefault(
            version.document_id,
            version,
        )

    return [
        DocumentInventoryItem(
            document_id=document.id,
            title=document.title,
            description=document.description,
            document_type=document.document_type,
            issuing_organization=(
                document.issuing_organization
            ),
            jurisdiction=document.jurisdiction,
            language=document.language,
            publication_date=(
                document.publication_date
            ),
            source_url=document.source_url,
            status=document.status,
            confidentiality_level=(
                document.confidentiality_level
            ),
            created_at=document.created_at,
            updated_at=document.updated_at,
            latest_version=(
                _version_summary(
                    latest_by_document[document.id]
                )
                if document.id in latest_by_document
                else None
            ),
        )
        for document in documents
    ]


@router.get(
    "/accessible",
    response_model=list[AccessibleDocumentItem],
)
def list_accessible_documents(
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
    authorization: Annotated[
        AuthorizationService,
        Depends(get_authorization_service),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> list[AccessibleDocumentItem]:
    scope = authorization.retrieval_scope(
        identity
    )

    if not scope.document_ids:
        return []

    documents = db.scalars(
        select(Document)
        .where(
            Document.id.in_(
                tuple(scope.document_ids)
            ),
            Document.status == "ACTIVE",
        )
        .order_by(
            Document.created_at.desc()
        )
    ).all()

    return [
        AccessibleDocumentItem(
            document_id=document.id,
            title=document.title,
            description=document.description,
            document_type=document.document_type,
            issuing_organization=(
                document.issuing_organization
            ),
            jurisdiction=document.jurisdiction,
            language=document.language,
            publication_date=(
                document.publication_date
            ),
            source_url=document.source_url,
        )
        for document in documents
    ]


@router.get(
    "/accessible/{document_id}",
    response_model=AccessibleDocumentDetailResponse,
)
def read_accessible_document(
    document_id: UUID,
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
    authorization: Annotated[
        AuthorizationService,
        Depends(get_authorization_service),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> AccessibleDocumentDetailResponse:
    try:
        authorization.require_document_read(
            identity,
            document_id,
        )
    except AuthorizationDeniedError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        ) from exc

    document = db.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.status == "ACTIVE",
        )
    )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return AccessibleDocumentDetailResponse(
        document_id=document.id,
        title=document.title,
        description=document.description,
        document_type=document.document_type,
        issuing_organization=(
            document.issuing_organization
        ),
        jurisdiction=document.jurisdiction,
        language=document.language,
        publication_date=document.publication_date,
        source_url=document.source_url,
    )


@router.get(
    "/{document_id}",
    response_model=DocumentDetailResponse,
)
def read_document(
    document_id: UUID,
    identity: Annotated[
        CurrentIdentity,
        Depends(require_admin_identity),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentDetailResponse:
    del identity

    document = db.get(
        Document,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    versions = db.scalars(
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id
            == document.id
        )
        .order_by(
            DocumentVersion.version_number.desc()
        )
    ).all()

    return _document_detail(
        document,
        list(versions),
    )


@router.patch(
    "/{document_id}",
    response_model=DocumentMetadataUpdateResponse,
)
def update_document_metadata(
    document_id: UUID,
    payload: DocumentMetadataPatch,
    identity: Annotated[
        CurrentIdentity,
        Depends(require_admin_identity),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentMetadataUpdateResponse:
    document = db.get(
        Document,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    updates = payload.model_dump(
        exclude_unset=True
    )

    required_text_fields = {
        "title",
        "issuing_organization",
        "jurisdiction",
        "language",
        "confidentiality_level",
    }

    optional_text_fields = {
        "description",
        "document_type",
        "source_url",
    }

    normalized_updates = {}

    for field_name, value in updates.items():
        if field_name in required_text_fields:
            if value is None:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"{field_name} cannot be null."
                    ),
                )

            normalized_updates[field_name] = (
                _required_text(
                    field_name=field_name,
                    value=value,
                )
            )

        elif field_name in optional_text_fields:
            normalized_updates[field_name] = (
                _optional_text(value)
            )

        else:
            normalized_updates[field_name] = value

    changed = {}

    for field_name, new_value in (
        normalized_updates.items()
    ):
        old_value = getattr(
            document,
            field_name,
        )

        if old_value == new_value:
            continue

        changed[field_name] = {
            "old": _audit_json_value(old_value),
            "new": _audit_json_value(new_value),
        }

        setattr(
            document,
            field_name,
            new_value,
        )

    if changed:
        _commit_document_audit(
            db,
            event_type=(
                AuditEventType.DOCUMENT_METADATA_UPDATED
            ),
            actor_id=identity.user_id,
            document_id=document.id,
            metadata={
                "changed_fields": changed,
            },
        )

        db.refresh(document)

    versions = db.scalars(
        select(DocumentVersion)
        .where(
            DocumentVersion.document_id
            == document.id
        )
        .order_by(
            DocumentVersion.version_number.desc()
        )
    ).all()

    return DocumentMetadataUpdateResponse(
        changed=bool(changed),
        changed_fields=list(changed.keys()),
        document=_document_detail(
            document,
            list(versions),
        ),
    )


@router.post(
    "/{document_id}/archive",
    response_model=DocumentArchiveResponse,
)
def archive_document(
    document_id: UUID,
    identity: Annotated[
        CurrentIdentity,
        Depends(require_admin_identity),
    ],
    db: Annotated[Session, Depends(get_db)],
) -> DocumentArchiveResponse:
    document = db.get(
        Document,
        document_id,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    if document.status == "ARCHIVED":
        return DocumentArchiveResponse(
            document_id=document.id,
            status=document.status,
            changed=False,
        )

    if document.status != "ACTIVE":
        raise HTTPException(
            status_code=409,
            detail=(
                "Document is not in an "
                "archivable state."
            ),
        )

    previous_status = document.status
    document.status = "ARCHIVED"

    _commit_document_audit(
        db,
        event_type=(
            AuditEventType.DOCUMENT_ARCHIVED
        ),
        actor_id=identity.user_id,
        document_id=document.id,
        metadata={
            "previous_status": previous_status,
            "new_status": "ARCHIVED",
        },
    )

    db.refresh(document)

    return DocumentArchiveResponse(
        document_id=document.id,
        status=document.status,
        changed=True,
    )
