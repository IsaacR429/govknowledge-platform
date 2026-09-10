from uuid import UUID


class AuthorizationDeniedError(Exception):
    """Access to a GovKnowledge resource was denied."""

    def __init__(
        self,
        document_id: UUID,
    ) -> None:
        self.document_id = document_id
        super().__init__(
            "Access to the requested document is denied."
        )
