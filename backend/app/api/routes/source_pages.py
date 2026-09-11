from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
)

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.source_pages import (
    get_authorized_source_page_service,
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
from app.domain.exceptions.authorization import (
    AuthorizationDeniedError,
)
from app.domain.exceptions.source_page import (
    SourcePageNotFoundError,
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
) -> SourcePageResponse:
    try:
        page = service.get(
            identity=identity,
            page_id=page_id,
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
