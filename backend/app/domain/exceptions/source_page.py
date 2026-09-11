from uuid import UUID


class SourcePageNotFoundError(Exception):
    def __init__(
        self,
        page_id: UUID,
    ) -> None:
        self.page_id = page_id

        super().__init__(
            "Source page was not found."
        )
