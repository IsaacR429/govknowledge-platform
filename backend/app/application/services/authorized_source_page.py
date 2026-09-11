from __future__ import annotations

from uuid import UUID

from app.application.ports.audit import (
    AuditEvent,
)
from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.source_page_repository import (
    SourcePage,
    SourcePageRepository,
)
from app.application.services.audit import (
    AuditPersistenceError,
    AuditService,
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
from app.domain.exceptions.source_page import (
    SourcePageNotFoundError,
)


class AuthorizedSourcePageService:
    def __init__(
        self,
        *,
        repository: SourcePageRepository,
        authorization: AuthorizationService,
        audit: AuditService,
    ) -> None:
        self._repository = repository
        self._authorization = (
            authorization
        )
        self._audit = audit

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
            # Do not invent resource/document
            # lineage for a nonexistent page.
            raise SourcePageNotFoundError(
                page_id
            )

        try:
            # Possession of page_id is never
            # treated as authorization.
            self._authorization.require_document_read(
                identity,
                page.document_id,
            )

        except AuthorizationDeniedError:
            try:
                self._audit.record(
                    AuditEvent(
                        event_type=(
                            AuditEventType
                            .SOURCE_PAGE_DENIED
                        ),
                        outcome=(
                            AuditOutcome.DENIED
                        ),
                        actor_user_id=(
                            identity.user_id
                        ),
                        resource_type=(
                            "source_page"
                        ),
                        resource_id=page.page_id,
                        document_id=(
                            page.document_id
                        ),
                        error_code=(
                            "DOCUMENT_READ_DENIED"
                        ),
                        metadata={
                            "permission": "READ",
                        },
                    )
                )

            except AuditPersistenceError:
                # Access is already denied.
                # Preserve the authorization
                # result and anti-enumeration
                # behavior even if audit storage
                # is temporarily unavailable.
                pass

            raise

        # Successful disclosure requires a
        # successfully persisted access event.
        # AuditPersistenceError intentionally
        # propagates here: no audit, no page.
        self._audit.record(
            AuditEvent(
                event_type=(
                    AuditEventType
                    .SOURCE_PAGE_VIEWED
                ),
                outcome=(
                    AuditOutcome.SUCCEEDED
                ),
                actor_user_id=(
                    identity.user_id
                ),
                resource_type="source_page",
                resource_id=page.page_id,
                document_id=page.document_id,
                metadata={
                    "permission": "READ",
                },
            )
        )

        return page
