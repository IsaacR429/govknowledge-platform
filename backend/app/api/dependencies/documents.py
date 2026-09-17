from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.application.services.document_ingestion import (
    DocumentIngestionService,
)
from app.core.config import get_settings
from app.infrastructure.database.repositories.document_ingestion import (
    PostgresDocumentIngestionRepository,
)
from app.infrastructure.database.session import get_db
from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


def get_document_ingestion_service(
    db: Annotated[
        Session,
        Depends(get_db),
    ],
) -> DocumentIngestionService:
    settings = get_settings()

    return DocumentIngestionService(
        repository=(
            PostgresDocumentIngestionRepository(
                db
            )
        ),
        storage=LocalFileStorage(
            Path(
                settings.document_storage_root
            )
        ),
    )
