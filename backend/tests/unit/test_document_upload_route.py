from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

import app.api.routes.documents as document_routes
from app.api.dependencies.authentication import get_current_identity
from app.api.dependencies.document_processing import (
    get_routed_parser_executor,
)
from app.api.dependencies.documents import get_document_ingestion_service
from app.core.config import get_settings
from app.infrastructure.database.session import get_db
from app.main import create_app


def test_admin_upload_schedules_processing_without_database(monkeypatch):
    document_id = uuid4()
    version_id = uuid4()
    job_id = uuid4()
    parser = object()
    uploads = []
    scheduled = []

    class FakeIngestionService:
        def create_initial_upload(
            self,
            request,
            source,
            *,
            actor_user_id,
            correlation_id,
            content_type="application/pdf",
        ):
            uploads.append((request, source.read(), actor_user_id, correlation_id))
            return SimpleNamespace(
                document_id=document_id,
                version_id=version_id,
                processing_job_id=job_id,
                file_hash="a" * 64,
                size_bytes=len(uploads[-1][1]),
            )

    def record_processing(
        processing_job_id,
        parser_executor,
        actor_user_id=None,
        correlation_id=None,
    ):
        scheduled.append(
            (
                processing_job_id,
                parser_executor,
                actor_user_id,
                correlation_id,
            )
        )

    monkeypatch.setattr(
        document_routes,
        "run_document_processing_job",
        record_processing,
    )

    app = create_app()
    app.dependency_overrides[get_document_ingestion_service] = (
        lambda: FakeIngestionService()
    )
    app.dependency_overrides[get_routed_parser_executor] = (
        lambda: parser
    )

    url = f"{get_settings().api_prefix}/documents"
    pdf_bytes = b"%PDF-1.4\n"
    request = {
        "files": {
            "file": (
                "sample.pdf",
                pdf_bytes,
                "application/pdf",
            )
        },
        "data": {
            "title": "QA sample",
            "issuing_organization": "QA",
            "jurisdiction": "QA",
            "language": "en",
            "confidentiality_level": "PUBLIC",
        },
    }

    with TestClient(app) as client:
        app.dependency_overrides[get_current_identity] = (
            lambda: SimpleNamespace(roles=frozenset())
        )
        denied = client.post(url, **request)
        assert denied.status_code == 403
        assert uploads == []
        assert scheduled == []

        app.dependency_overrides[get_current_identity] = (
            lambda: SimpleNamespace(
                user_id=uuid4(),
                roles=frozenset({"ADMIN"}),
            )
        )
        accepted = client.post(url, **request)

    assert accepted.status_code == 201, accepted.text
    body = accepted.json()
    assert body["document_id"] == str(document_id)
    assert body["version_id"] == str(version_id)
    assert body["processing_job_id"] == str(job_id)

    assert len(uploads) == 1
    assert uploads[0][0].original_filename == "sample.pdf"
    assert uploads[0][1] == pdf_bytes
    assert len(scheduled) == 1
    assert scheduled[0][0] == job_id
    assert scheduled[0][1] is parser
    assert scheduled[0][2] is not None
    assert scheduled[0][3] is not None


def test_auditor_is_denied_all_admin_mutation_routes():
    app = create_app()

    auditor_id = uuid4()
    document_id = uuid4()
    user_id = uuid4()
    processing_job_id = uuid4()

    downstream_calls = []

    def fail_if_ingestion_reached():
        downstream_calls.append("ingestion")
        raise AssertionError(
            "Auditor reached document ingestion dependency."
        )

    def fail_if_db_reached():
        downstream_calls.append("database")
        raise AssertionError(
            "Auditor reached database mutation dependency."
        )

    def fail_if_parser_reached():
        downstream_calls.append("parser")
        raise AssertionError(
            "Auditor reached parser dependency."
        )

    app.dependency_overrides[get_current_identity] = (
        lambda: SimpleNamespace(
            user_id=auditor_id,
            roles=frozenset({"AUDITOR"}),
        )
    )

    app.dependency_overrides[
        get_document_ingestion_service
    ] = fail_if_ingestion_reached

    app.dependency_overrides[
        get_routed_parser_executor
    ] = fail_if_parser_reached

    app.dependency_overrides[
        get_db
    ] = fail_if_db_reached

    prefix = get_settings().api_prefix

    with TestClient(app) as client:
        responses = [
            client.post(
                f"{prefix}/documents",
                files={
                    "file": (
                        "auditor.pdf",
                        b"%PDF-1.4\n%%EOF\n",
                        "application/pdf",
                    )
                },
                data={
                    "title": "Auditor forbidden upload",
                    "issuing_organization": "QA",
                    "jurisdiction": "QA",
                    "language": "en",
                    "confidentiality_level": "PUBLIC",
                },
            ),
            client.post(
                (
                    f"{prefix}/documents/jobs/"
                    f"{processing_job_id}/retry-indexing"
                )
            ),
            client.patch(
                f"{prefix}/documents/{document_id}",
                json={
                    "title": "Auditor forbidden update",
                },
            ),
            client.post(
                (
                    f"{prefix}/documents/"
                    f"{document_id}/archive"
                )
            ),
            client.post(
                (
                    f"{prefix}/documents/"
                    f"{document_id}/permissions/read/"
                    f"{user_id}"
                )
            ),
            client.delete(
                (
                    f"{prefix}/documents/"
                    f"{document_id}/permissions/read/"
                    f"{user_id}"
                )
            ),
        ]

    assert [
        response.status_code
        for response in responses
    ] == [
        403,
        403,
        403,
        403,
        403,
        403,
    ]

    assert downstream_calls == []
