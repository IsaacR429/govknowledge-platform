from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    Query,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies.authentication import (
    get_current_identity,
)
from app.api.dependencies.qa import (
    get_authorized_search_service,
)
from app.api.schemas.search import (
    SearchResponse,
    SearchResultItem,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.services.authorized_search import (
    AuthorizedSearchService,
)
from app.infrastructure.database.models.documents import (
    Document,
    DocumentVersion,
)
from app.infrastructure.database.session import (
    get_db,
)


router = APIRouter(
    prefix="/search",
    tags=["search"],
)


@router.get(
    "",
    response_model=SearchResponse,
)
def search_documents(
    identity: Annotated[
        CurrentIdentity,
        Depends(get_current_identity),
    ],
    search: Annotated[
        AuthorizedSearchService,
        Depends(get_authorized_search_service),
    ],
    db: Annotated[
        Session,
        Depends(get_db),
    ],
    q: Annotated[
        str,
        Query(
            min_length=1,
            max_length=4000,
        ),
    ],
    top_k: Annotated[
        int,
        Query(
            ge=1,
            le=20,
        ),
    ] = 5,
) -> SearchResponse:
    result = search.search(
        identity=identity,
        query=q,
        top_k=top_k,
    )

    if not result.results:
        return SearchResponse(
            query=result.query,
            result_count=0,
            results=[],
        )

    document_ids = {
        item.document_id
        for item in result.results
    }

    version_ids = {
        item.document_version_id
        for item in result.results
    }

    documents = {
        document.id: document
        for document in db.scalars(
            select(Document).where(
                Document.id.in_(
                    document_ids
                )
            )
        ).all()
    }

    versions = {
        version.id: version
        for version in db.scalars(
            select(DocumentVersion).where(
                DocumentVersion.id.in_(
                    version_ids
                )
            )
        ).all()
    }

    response_items = []

    for item in result.results:
        document = documents.get(
            item.document_id
        )
        version = versions.get(
            item.document_version_id
        )

        # Retrieved chunks should never reference
        # missing provenance records.
        if document is None or version is None:
            raise RuntimeError(
                "Search result provenance "
                "could not be resolved."
            )

        response_items.append(
            SearchResultItem(
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                document_version_id=(
                    item.document_version_id
                ),
                version_number=(
                    version.version_number
                ),
                page_id=item.page_id,
                page_number=item.page_number,
                page_end=item.page_end,
                chunk_index=item.chunk_index,
                document_title=document.title,
                issuing_organization=(
                    document.issuing_organization
                ),
                jurisdiction=(
                    document.jurisdiction
                ),
                section_title=(
                    item.section_title
                ),
                language=item.language,
                content=item.content,
                score=item.score,
            )
        )

    return SearchResponse(
        query=result.query,
        result_count=len(response_items),
        results=response_items,
    )
