from __future__ import annotations

from uuid import UUID

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.source_page_repository import (
    SourcePage,
    SourcePageRepository,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.domain.exceptions.source_page import (
    SourcePageNotFoundError,
)


class AuthorizedSourcePageService:
    def __init__(
        self,
        *,
        repository: SourcePageRepository,
        authorization: AuthorizationService,
    ) -> None:
        self._repository = repository
        self._authorization = (
            authorization
        )

    def get(
        self,
        *,
        identity: CurrentIdentity,
        page_id: UUID,
    ) -> SourcePage:
        page = (
            self._repository
            .get_by_id(page_id)
        )

        if page is None:
            raise SourcePageNotFoundError(
                page_id
            )

        # Possession of page_id is never
        # treated as authorization.
        self._authorization.require_document_read(
            identity,
            page.document_id,
        )

        return page
