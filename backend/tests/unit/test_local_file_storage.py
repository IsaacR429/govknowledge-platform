from io import BytesIO
from uuid import uuid4

from app.infrastructure.storage.local_file_storage import (
    LocalFileStorage,
)


def test_local_storage_uses_safe_relative_path(
    tmp_path,
):
    storage = LocalFileStorage(tmp_path)

    document_id = uuid4()
    version_id = uuid4()

    relative_path = storage.store_document_version(
        document_id=document_id,
        version_id=version_id,
        original_filename="../../policy.pdf",
        source=BytesIO(b"govknowledge"),
    )

    assert not relative_path.is_absolute()
    assert relative_path.name == "policy.pdf"

    stored_file = storage.resolve(
        str(relative_path)
    )

    assert stored_file.read_bytes() == b"govknowledge"
