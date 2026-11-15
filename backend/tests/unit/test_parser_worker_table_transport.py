from __future__ import annotations

import json

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.infrastructure.parsing.parser_worker_entry import (
    _serialize_parsed_document,
)
from app.infrastructure.parsing.table_evidence_transport import (
    deserialize_table_evidence,
)
from tests.unit.test_table_evidence_transport import (
    valid_document_payload,
)


def test_outer_worker_serializes_table_evidence() -> None:
    evidence = deserialize_table_evidence(
        valid_document_payload()
    )
    parsed = ParsedDocument(
        pages=(),
        chunks=(),
        table_evidence=evidence,
    )
    serialized = (
        _serialize_parsed_document(
            parsed
        )
    )
    assert (
        serialized[
            "table_evidence_schema_version"
        ]
        == 1
    )
    assert (
        len(
            serialized[
                "table_evidence"
            ]
        )
        == 1
    )
    # The worker payload must remain safe for
    # result.json transport.
    json.dumps(serialized)
    reconstructed = (
        deserialize_table_evidence(
            serialized
        )
    )
    assert reconstructed == evidence


def test_outer_worker_emits_explicit_empty_table_evidence() -> None:
    parsed = ParsedDocument(
        pages=(),
        chunks=(),
        table_evidence=(),
    )
    serialized = (
        _serialize_parsed_document(
            parsed
        )
    )
    assert (
        serialized[
            "table_evidence_schema_version"
        ]
        == 1
    )
    assert (
        serialized[
            "table_evidence"
        ]
        == []
    )
    assert (
        deserialize_table_evidence(
            serialized
        )
        == ()
    )
