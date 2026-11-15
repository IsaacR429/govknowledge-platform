from hashlib import sha256
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

    content = b"govknowledge"

    stored = storage.store_document_version(
        document_id=document_id,
        version_id=version_id,
        original_filename="../../policy.pdf",
        source=BytesIO(content),
    )

    assert not stored.relative_path.is_absolute()
    assert stored.relative_path.name == "policy.pdf"
    assert stored.size_bytes == len(content)
    assert stored.sha256 == sha256(
        content
    ).hexdigest()

    stored_file = storage.resolve(
        str(stored.relative_path)
    )

    assert stored_file.read_bytes() == content


def test_local_storage_can_delete_file(
    tmp_path,
):
    storage = LocalFileStorage(tmp_path)

    stored = storage.store_document_version(
        document_id=uuid4(),
        version_id=uuid4(),
        original_filename="policy.pdf",
        source=BytesIO(b"temporary"),
    )

    path = storage.resolve(
        str(stored.relative_path)
    )

    assert path.exists()

    storage.delete(
        str(stored.relative_path)
    )

    assert not path.exists()


def test_ingestion_removes_file_when_repository_fails(
    tmp_path,
):
    from app.application.services.document_ingestion import (
        CreateDocumentRequest,
        DocumentIngestionService,
    )

    class FailingRepository:
        def create_initial_upload(
            self,
            upload,
        ):
            raise RuntimeError(
                "Simulated database failure"
            )

    class DummyAuditService:
        def record(self, event):
            pass

    storage = LocalFileStorage(
        tmp_path / "documents"
    )

    service = DocumentIngestionService(
        repository=FailingRepository(),
        storage=storage,
        audit=DummyAuditService(),
    )

    request = CreateDocumentRequest(
        title="Failure test",
        description=None,
        document_type="POLICY",
        issuing_organization="Test",
        jurisdiction="Burkina Faso",
        language="fr",
        publication_date=None,
        source_url=None,
        confidentiality_level="INTERNAL",
        original_filename="failure.pdf",
    )

    import pytest

    with pytest.raises(
        RuntimeError,
        match="Simulated database failure",
    ):
        service.create_initial_upload(
            request=request,
            source=BytesIO(b"temporary"),
            actor_user_id=uuid4(),
            correlation_id=uuid4(),
        )

    remaining_files = [
        path
        for path in (
            tmp_path / "documents"
        ).rglob("*")
        if path.is_file()
    ]

    assert remaining_files == []
