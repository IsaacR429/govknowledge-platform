import os
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import make_url


def _require_isolated_qa():
    from app.core.config import get_settings
    from app.infrastructure.database.session import SessionLocal, engine

    raw_url = os.environ.get("DATABASE_URL")
    expected_system_id = os.environ.get("GOVKNOWLEDGE_QA_SYSTEM_ID")

    if not raw_url or expected_system_id != "7686228710959620134":
        raise RuntimeError("STOP: isolated QA configuration is missing.")

    expected = make_url(raw_url)

    # Check the destination BEFORE opening a database connection.
    if not (
        expected.drivername == "postgresql+psycopg"
        and expected.host == "127.0.0.1"
        and expected.port == 15432
        and expected.database == "govknowledge_qa"
        and expected.username == "govknowledge_qa"
        and expected.password
        and make_url(get_settings().database_url) == expected
        and engine.url == expected
    ):
        raise RuntimeError("STOP: database configuration is not QA.")

    with SessionLocal() as db:
        if db.get_bind() is not engine:
            raise RuntimeError("STOP: session is not bound to the QA engine.")

        db.execute(text("SET TRANSACTION READ ONLY"))
        database, username, system_id = db.execute(
            text(
                "SELECT current_database(), current_user, "
                "system_identifier::text FROM pg_control_system()"
            )
        ).one()

    if (database, username, system_id) != (
        "govknowledge_qa",
        "govknowledge_qa",
        expected_system_id,
    ):
        raise RuntimeError("STOP: PostgreSQL instance is not QA.")

    print("ISOLATED QA DATABASE VERIFIED")
    return SessionLocal


def _create_small_pdf(path: Path) -> bytes:
    """Create a one-page PDF using only Python's standard library."""
    lines = [
        "QA POLICY REFERENCE",
        "Employees must verify identity before granting document access.",
        "An answer must cite a page from an authorized source document.",
        "When source evidence is missing, the assistant must say so.",
    ]

    stream = (
        b"BT\n/F1 12 Tf\n72 730 Td\n20 TL\n"
        + b"".join(
            f"({line}) Tj\nT*\n".encode("ascii")
            for line in lines
        )
        + b"ET\n"
    )

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R "
            b"/MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> "
            b"/Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream"
        ),
    ]

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []

    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{number} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets:
        pdf.extend(f"{offset:010} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    result = bytes(pdf)
    path.write_bytes(result)
    return result


def test_p0_admin_upload_processing_indexing_and_readiness(
    monkeypatch,
    tmp_path,
):
    # Must be the first operation that could access PostgreSQL.
    SessionLocal = _require_isolated_qa()

    from app.api.dependencies.administration import require_admin_identity
    from app.core.config import get_settings
    from app.infrastructure.database.models.documents import DocumentVersion
    from app.main import create_app

    pdf_path = tmp_path / "p0_qa_policy.pdf"
    pdf_bytes = _create_small_pdf(pdf_path)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) < 5_000

    platform_root = Path(__file__).resolve().parents[3]
    qa_storage = (
        platform_root
        / ".runtime"
        / "qa-artifacts"
        / "p0-upload-documents"
    )

    settings = get_settings()
    monkeypatch.setattr(
        settings,
        "document_storage_root",
        str(qa_storage),
    )
    monkeypatch.setenv("GOVKNOWLEDGE_EXPERIMENTAL_VECTOR_TABLES", "0")
    monkeypatch.setenv("GOVKNOWLEDGE_EXPERIMENTAL_TABLE_EVIDENCE", "0")

    app = create_app()
    app.dependency_overrides[require_admin_identity] = (
        lambda: SimpleNamespace(roles=frozenset({"ADMIN"}))
    )

    prefix = settings.api_prefix
    fixture_title = f"P0 QA acceptance {uuid4()}"
    print(f"QA FIXTURE TITLE: {fixture_title}")

    with TestClient(app) as client:
        with pdf_path.open("rb") as source:
            response = client.post(
                f"{prefix}/documents",
                files={
                    "file": (
                        pdf_path.name,
                        source,
                        "application/pdf",
                    )
                },
                data={
                    "title": fixture_title,
                    "issuing_organization": "QA acceptance test",
                    "jurisdiction": "QA",
                    "language": "en",
                    "confidentiality_level": "PUBLIC",
                },
            )

        assert response.status_code == 201, response.text

        created = response.json()
        document_id = UUID(created["document_id"])
        version_id = UUID(created["version_id"])
        job_id = created["processing_job_id"]

        print(f"QA TEST DOCUMENT ID: {document_id}")
        print(f"QA PROCESSING JOB ID: {job_id}")

        job_response = client.get(
            f"{prefix}/documents/jobs/{job_id}"
        )
        assert job_response.status_code == 200, job_response.text
        job = job_response.json()

    print(f"QA PROCESSING STATUS: {job['processing_status']}")
    print(f"QA INDEXING STATUS: {job['indexing_status']}")
    print(f"QA READINESS: {job['readiness_status']}")
    print(f"QA CHUNKS: {job['chunk_count']}")
    print(f"QA SEARCHABLE CHUNKS: {job['searchable_chunk_count']}")
    print(f"QA SEARCHABLE INDEXED: {job['searchable_indexed_count']}")

    assert job["document_id"] == str(document_id)
    assert job["version_id"] == str(version_id)
    assert job["processing_status"] == "SUCCEEDED", job
    assert job["version_processing_status"] == "SUCCEEDED", job
    assert job["indexing_status"] == "SUCCEEDED", job
    assert job["readiness_status"] == "SEARCH_READY", job
    assert job["ready_for_authorized_users"] is True
    assert job["chunk_count"] >= 1, job
    assert job["searchable_chunk_count"] == job["chunk_count"], job
    assert job["indexed_count"] == job["chunk_count"], job
    assert (
        job["searchable_indexed_count"]
        == job["searchable_chunk_count"]
    ), job

    with SessionLocal() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        version = db.get(DocumentVersion, version_id)
        assert version is not None
        assert version.document_id == document_id

        stored_pdf = qa_storage / version.storage_path
        assert stored_pdf.is_file()
        assert stored_pdf.read_bytes() == pdf_bytes
        assert version.file_hash == sha256(pdf_bytes).hexdigest()

    print("QA P0 UPLOAD, PROCESSING, INDEXING, READINESS VERIFIED")
