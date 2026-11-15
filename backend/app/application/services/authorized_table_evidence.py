from __future__ import annotations

from app.application.ports.identity_provider import (
    CurrentIdentity,
)
from app.application.ports.table_evidence_repository import (
    AuthorizedTableVocabulary,
    TableEvidenceRepository,
)
from app.application.services.authorization import (
    AuthorizationService,
)
from app.application.services.table_evidence_atoms import (
    LabeledRowLookupRequest,
    TableEvidenceLookupService,
    TableLookupRequest,
    TableLookupSelection,
)


class AuthorizedTableEvidenceError(
    RuntimeError
):
    pass


class AuthorizedTableEvidenceService:
    """Authorization-first typed table evidence access.

    Security sequence for exact cell lookup:

        identity
          ↓
        AuthorizationService.retrieval_scope()
          ↓
        persistence query constrained by scope
          ↓
        defense-in-depth result validation
          ↓
        deterministic typed lookup
          ↓
        exact READ reauthorization before disclosure

    Structural vocabulary follows the same
    authorization-first principle, but does not
    disclose table cell values.
    """

    def __init__(
        self,
        *,
        authorization: AuthorizationService,
        repository: TableEvidenceRepository,
        lookup: (
            TableEvidenceLookupService
            | None
        ) = None,
    ) -> None:

        self._authorization = (
            authorization
        )

        self._repository = (
            repository
        )

        self._lookup = (
            lookup
            or TableEvidenceLookupService()
        )

    def vocabulary(
        self,
        *,
        identity: CurrentIdentity,
    ) -> AuthorizedTableVocabulary:
        """
        Return structural table vocabulary visible to
        this identity.

        No authorization means no persistence query.
        """

        # --------------------------------------------
        # Authorization BEFORE persistence.
        # --------------------------------------------

        scope = (
            self._authorization
            .retrieval_scope(
                identity
            )
        )

        # --------------------------------------------
        # No permission => no corpus query.
        # --------------------------------------------

        if not scope.document_ids:
            return (
                AuthorizedTableVocabulary
                .empty()
            )

        vocabulary = (
            self._repository
            .list_authorized_vocabulary(
                scope=scope
            )
        )

        # --------------------------------------------
        # Defense in depth around repository contract.
        # --------------------------------------------

        if not isinstance(
            vocabulary,
            AuthorizedTableVocabulary,
        ):
            raise (
                AuthorizedTableEvidenceError(
                    "Table evidence repository "
                    "returned invalid vocabulary."
                )
            )

        return vocabulary

    def lookup(
        self,
        *,
        identity: CurrentIdentity,
        request: (
            TableLookupRequest
            | LabeledRowLookupRequest
        ),
    ) -> (
        TableLookupSelection
        | None
    ):

        # --------------------------------------------
        # Authorization happens BEFORE persistence.
        # --------------------------------------------

        scope = (
            self._authorization
            .retrieval_scope(
                identity
            )
        )

        # No permissions => no corpus/database lookup.
        if not scope.document_ids:
            return None

        logical_key = (
            request.logical_key
        )

        rows = (
            self._repository
            .find_authorized_by_logical_key(
                scope=scope,
                logical_key=(
                    logical_key
                ),
            )
        )

        # --------------------------------------------
        # Defense in depth.
        #
        # SQL/repository filtering is the real
        # authorization control. This detects a broken
        # repository implementation rather than trying
        # to compensate for one.
        # --------------------------------------------

        for row in rows:

            if not scope.allows(
                row.document_id
            ):
                raise (
                    AuthorizedTableEvidenceError(
                        (
                            "Table evidence repository "
                            "violated authorization scope."
                        )
                    )
                )

            if (
                row.logical_key
                != logical_key
            ):
                raise (
                    AuthorizedTableEvidenceError(
                        (
                            "Table evidence repository "
                            "returned a row outside the "
                            "requested logical key."
                        )
                    )
                )

        selection = (
            self._lookup.lookup(
                request=request,
                evidence=rows,
            )
        )

        if selection is None:
            return None

        # --------------------------------------------
        # Reauthorize exact selected document
        # immediately before disclosure.
        # --------------------------------------------

        self._authorization.require_document_read(
            identity,
            selection.row.document_id,
        )

        return selection
