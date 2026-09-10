from hashlib import sha256
from pathlib import Path
from typing import BinaryIO
from uuid import UUID

from app.application.ports.file_storage import StoredFile


class LocalFileStorage:
    """Local P0 storage with path safety and streaming SHA-256."""

    def __init__(
        self,
        root: Path,
    ) -> None:
        self._root = root.resolve()
        self._root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _safe_filename(
        filename: str,
    ) -> str:
        safe_name = Path(filename).name

        if not safe_name:
            raise ValueError(
                "Document filename cannot be empty."
            )

        return safe_name

    def store_document_version(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        original_filename: str,
        source: BinaryIO,
    ) -> StoredFile:
        safe_filename = self._safe_filename(
            original_filename
        )

        relative_path = (
            Path(str(document_id))
            / str(version_id)
            / safe_filename
        )

        destination = (
            self._root / relative_path
        ).resolve()

        if self._root not in destination.parents:
            raise ValueError(
                "Invalid document storage path."
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        digest = sha256()
        size_bytes = 0

        try:
            with destination.open("wb") as output:
                while True:
                    block = source.read(
                        1024 * 1024
                    )

                    if not block:
                        break

                    output.write(block)
                    digest.update(block)
                    size_bytes += len(block)

        except Exception:
            destination.unlink(
                missing_ok=True
            )
            raise

        return StoredFile(
            relative_path=relative_path,
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
        )

    def resolve(
        self,
        storage_path: str,
    ) -> Path:
        resolved = (
            self._root / storage_path
        ).resolve()

        if (
            resolved != self._root
            and self._root not in resolved.parents
        ):
            raise ValueError(
                "Invalid document storage path."
            )

        return resolved

    def delete(
        self,
        storage_path: str,
    ) -> None:
        path = self.resolve(storage_path)

        path.unlink(
            missing_ok=True
        )
