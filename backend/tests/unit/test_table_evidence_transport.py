import pytest

from app.domain.enums.table import (
    TableCellTextStatus,
    TableEvidenceLayer,
)
from app.infrastructure.parsing.table_evidence_transport import (
    deserialize_table_evidence,
)


def valid_document_payload() -> dict[str, object]:
    return {
        "table_evidence_schema_version": 1,
        "table_evidence": [
            {
                "source_ref": "native:p9:t1",
                "page_number": 9,
                "layer": "PHYSICAL",
                "column_count": 3,
                "bbox": [
                    10.0,
                    20.0,
                    500.0,
                    700.0,
                ],
                "source_spans": [
                    {
                        "page_number": 9,
                        "kind": "GEOMETRY",
                        "source_ref": (
                            "native:p9:t1"
                        ),
                        "bbox": [
                            10.0,
                            20.0,
                            500.0,
                            700.0,
                        ],
                        "text": None,
                    }
                ],
                "rows": [
                    {
                        "row_index": 0,
                        "source_header_signal": True,
                        "source_role": (
                            "native-header"
                        ),
                        "source_spans": [],
                        "cells": [
                            {
                                "column_start": 0,
                                "column_end": 1,
                                "text_status": (
                                    "OBSERVED"
                                ),
                                "text": "Domaine",
                                "source_role": None,
                                "source_spans": [
                                    {
                                        "page_number": 9,
                                        "kind": "TEXT",
                                        "source_ref": (
                                            "native:"
                                            "p9:t1:r0:c0"
                                        ),
                                        "bbox": None,
                                        "text": "Domaine",
                                    }
                                ],
                            },
                            {
                                "column_start": 1,
                                "column_end": 2,
                                "text_status": (
                                    "OBSERVED"
                                ),
                                "text": "",
                                "source_role": None,
                                "source_spans": [],
                            },
                            {
                                "column_start": 2,
                                "column_end": 3,
                                "text_status": (
                                    "UNAVAILABLE"
                                ),
                                "text": None,
                                "source_role": None,
                                "source_spans": [],
                            },
                        ],
                    }
                ],
            }
        ],
    }


def test_absent_table_evidence_is_backward_compatible() -> None:
    result = deserialize_table_evidence(
        {
            "pages": [],
            "chunks": [],
        }
    )

    assert result == ()


def test_deserializes_normalized_table_evidence() -> None:
    result = deserialize_table_evidence(
        valid_document_payload()
    )

    assert len(result) == 1

    table = result[0]

    assert (
        table.layer
        == TableEvidenceLayer.PHYSICAL
    )

    assert table.page_number == 9
    assert table.column_count == 3
    assert table.row_count == 1

    cells = table.rows[0].cells

    assert (
        cells[0].text_status
        == TableCellTextStatus.OBSERVED
    )
    assert cells[0].text == "Domaine"

    assert cells[1].observed_blank is True

    assert (
        cells[2].text_status
        == TableCellTextStatus.UNAVAILABLE
    )
    assert cells[2].text is None


def test_rejects_unknown_schema_version() -> None:
    payload = valid_document_payload()

    payload[
        "table_evidence_schema_version"
    ] = 99

    with pytest.raises(
        ValueError,
        match="schema version",
    ):
        deserialize_table_evidence(
            payload
        )


def test_rejects_evidence_without_version() -> None:
    payload = valid_document_payload()

    del payload[
        "table_evidence_schema_version"
    ]

    with pytest.raises(
        ValueError,
        match="schema version",
    ):
        deserialize_table_evidence(
            payload
        )


def test_rejects_cross_page_raw_span() -> None:
    payload = valid_document_payload()

    table = payload[
        "table_evidence"
    ][0]

    table[
        "rows"
    ][0][
        "cells"
    ][0][
        "source_spans"
    ][0][
        "page_number"
    ] = 10

    with pytest.raises(
        ValueError,
        match="another page",
    ):
        deserialize_table_evidence(
            payload
        )


def test_rejects_invalid_text_status_contract() -> None:
    payload = valid_document_payload()

    table = payload[
        "table_evidence"
    ][0]

    cell = (
        table[
            "rows"
        ][0][
            "cells"
        ][2]
    )

    cell["text_status"] = "UNAVAILABLE"
    cell["text"] = "invented text"

    with pytest.raises(
        ValueError,
        match="UNAVAILABLE",
    ):
        deserialize_table_evidence(
            payload
        )


def test_rejects_unknown_layer() -> None:
    payload = valid_document_payload()

    payload[
        "table_evidence"
    ][0][
        "layer"
    ] = "MAGIC"

    with pytest.raises(
        ValueError,
        match="layer",
    ):
        deserialize_table_evidence(
            payload
        )


def test_parsed_document_remains_backward_compatible() -> None:
    from app.application.ports.document_parser import (
        ParsedDocument,
    )

    document = ParsedDocument(
        pages=(),
        chunks=(),
    )

    assert document.table_evidence == ()
