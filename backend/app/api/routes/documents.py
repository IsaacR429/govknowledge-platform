from datetime import date
from pathlib import Path
from typing import Annotated
from uuid import UUID

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
from sqlalchemy.orm import Session

from app.api.dependencies.administration import (
    require_admin_identity,
)
from app.api.dependencies.document_processing import (
    get_routed_parser_executor,
    run_document_processing_job,
)
from app.api.dependencies.documents import (
    get_document_ingestion_service,
)
from app.api.schemas.documents import (
    DocumentUploadResponse,
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
    DocumentVersion,
)
from app.infrastructure.database.models.processing import (
    DocumentProcessingJob,
)
from app.infrastructure.database.session import get_db


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
    del identity

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
        )
    )

    background_tasks.add_task(
        run_document_processing_job,
        result.processing_job_id,
        parser_executor,
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
    }
