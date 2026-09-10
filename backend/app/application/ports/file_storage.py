from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol
from uuid import UUID


@dataclass(frozen=True)
class StoredFile:
    relative_path: Path
    sha256: str
    size_bytes: int


class FileStorage(Protocol):
    def store_document_version(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        original_filename: str,
        source: BinaryIO,
    ) -> StoredFile:
        ...

    def resolve(
        self,
        storage_path: str,
    ) -> Path:
        ...

    def delete(
        self,
        storage_path: str,
    ) -> None:
        ...
