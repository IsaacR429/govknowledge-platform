from pathlib import Path
from shutil import copyfileobj
from typing import BinaryIO
from uuid import UUID


class LocalFileStorage:
    """Local P0 document storage.

    The database stores relative paths rather than machine-specific
    absolute paths.
    """

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
        return Path(filename).name

    def store_document_version(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        original_filename: str,
        source: BinaryIO,
    ) -> Path:
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

        with destination.open("wb") as output:
            copyfileobj(
                source,
                output,
            )

        return relative_path

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
