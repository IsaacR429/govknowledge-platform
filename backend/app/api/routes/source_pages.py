from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Response,
    status,
)
from sqlalchemy.orm import Session

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.source_pages import (
    get_authorized_source_page_service,
    get_source_page_storage,
)
from app.api.schemas.source_pages import (
    SourcePageResponse,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorized_source_page import (
    AuthorizedSourcePageService,
)
from app.core.config import get_settings
from app.domain.exceptions.authorization import (
    AuthorizationDeniedError,
)
from app.domain.exceptions.source_page import (
    SourcePageNotFoundError,
)
from app.infrastructure.database.models.documents import (
    DocumentVersion,
)
from app.infrastructure.database.session import get_db
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


router = APIRouter(
    tags=["sources"],
)


@router.get(
    "/source-pages/{page_id}",
    response_model=SourcePageResponse,
)
def get_source_page(
    page_id: UUID,
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
    service: Annotated[
        AuthorizedSourcePageService,
        Depends(
            get_authorized_source_page_service
        ),
    ],
    correlation_id: UUID | None = None,
) -> SourcePageResponse:
    try:
        page = service.get(
            identity=identity,
            page_id=page_id,
            correlation_id=correlation_id,
        )

    except (
        SourcePageNotFoundError,
        AuthorizationDeniedError,
    ) as exc:
        # Do not reveal whether an inaccessible
        # page actually exists.
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail=(
                "Source page not found."
            ),
        ) from exc

    return SourcePageResponse(
        page_id=page.page_id,
        document_id=page.document_id,
        document_version_id=(
            page.document_version_id
        ),
        page_number=page.page_number,
        extracted_text=(
            page.extracted_text
        ),
        metadata=dict(page.metadata),
        document_title=(
            page.document_title
        ),
        issuing_organization=(
            page.issuing_organization
        ),
        jurisdiction=page.jurisdiction,
        document_language=(
            page.document_language
        ),
        publication_date=(
            page.publication_date
        ),
        source_url=page.source_url,
        version_number=(
            page.version_number
        ),
        original_filename=(
            page.original_filename
        ),
    )


@router.get(
    "/source-pages/{page_id}/original.pdf",
    response_class=Response,
    responses={
        200: {
            "description": "Authorized original source page as PDF.",
            "content": {
                "application/pdf": {
                    "schema": {
                        "type": "string",
                        "format": "binary",
                    }
                }
            },
        }
    },
)
def get_original_source_page(
    page_id: UUID,
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
    service: Annotated[
        AuthorizedSourcePageService,
        Depends(get_authorized_source_page_service),
    ],
    db: Session = Depends(get_db),
    storage: LocalFileStorage = Depends(get_source_page_storage),
    correlation_id: UUID | None = None,
) -> Response:
    try:
        # Check READ permission before opening the original file.
        page = service.authorize(
            identity=identity,
            page_id=page_id,
            correlation_id=correlation_id,
        )
    except (
        SourcePageNotFoundError,
        AuthorizationDeniedError,
    ) as exc:
        raise HTTPException(
            status_code=404,
            detail="Source page not found.",
        ) from exc

    version = db.get(
        DocumentVersion,
        page.document_version_id,
    )

    if (
        version is None
        or version.document_id != page.document_id
    ):
        raise HTTPException(
            status_code=503,
            detail="Original source page is unavailable.",
        )

    try:
        original_path = storage.resolve(
            version.storage_path
        )
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Original source page is unavailable.",
        ) from exc

    backend_root = Path(__file__).resolve().parents[3]
    platform_root = backend_root.parent

    configured_python = os.environ.get(
        "GOVKNOWLEDGE_PYMUPDF_PYTHON"
    )

    worker_python = (
        Path(configured_python).expanduser().resolve()
        if configured_python
        else platform_root
        / ".runtime/venvs/pymupdf4llm/Scripts/python.exe"
    )

    worker_script = (
        backend_root
        / "app/infrastructure/source_pages/"
          "single_page_pdf_worker.py"
    )

    if (
        not worker_python.is_file()
        or not worker_script.is_file()
    ):
        raise HTTPException(
            status_code=503,
            detail="Original source page is unavailable.",
        )

    # The worker verifies the original's SHA-256 and extracts
    # only the requested physical page.
    try:
        result = subprocess.run(
            [
                str(worker_python),
                "-I",
                str(worker_script),
                str(original_path),
                str(version.file_hash),
                str(page.page_number),
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise HTTPException(
            status_code=503,
            detail="Original source page is unavailable.",
        ) from exc

    # Refuse worker errors, empty output, and unexpectedly large
    # responses. Do not return worker stderr to the caller.
    if (
        result.returncode != 0
        or not result.stdout.startswith(b"%PDF-")
        or len(result.stdout) > 25 * 1024 * 1024
    ):
        raise HTTPException(
            status_code=503,
            detail="Original source page is unavailable.",
        )

    response = Response(
        content=result.stdout,
        media_type="application/pdf",
        headers={
            "Content-Disposition": (
                f'inline; filename="source-page-{page.page_number}.pdf"'
            ),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )

    # Recheck permission after potentially lengthy PDF extraction.
    # A denied recheck is indistinguishable from a missing page.
    try:
        service.authorize(
            identity=identity,
            page_id=page_id,
            correlation_id=correlation_id,
        )
    except (
        SourcePageNotFoundError,
        AuthorizationDeniedError,
    ) as exc:
        raise HTTPException(
            status_code=404,
            detail="Source page not found.",
        ) from exc

    # If audit persistence fails, this exception propagates and
    # the prepared PDF response is never returned.
    service.record_view(
        identity=identity,
        page=page,
        correlation_id=correlation_id,
    )

    return response
