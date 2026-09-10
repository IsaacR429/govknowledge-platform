from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID


class FileStorage(Protocol):
    def store_document_version(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        original_filename: str,
        source: BinaryIO,
    ) -> Path:
        ...

    def resolve(
        self,
        storage_path: str,
    ) -> Path:
        ...
