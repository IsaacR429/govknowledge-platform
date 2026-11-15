from __future__ import annotations

from typing import Any

from app.domain.enums.table import (
    TableCellTextStatus,
    TableEvidenceLayer,
    TableSpanKind,
)
from app.domain.models.table import (
    TableBBox,
    TableSourceSpan,
)
from app.domain.tables.evidence import (
    NormalizedTableCellEvidence,
    NormalizedTableEvidence,
    NormalizedTableRowEvidence,
)


TABLE_EVIDENCE_SCHEMA_VERSION = 1


def _require_dict(
    value: object,
    *,
    name: str,
) -> dict[str, object]:

    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must be an object."
        )

    return value


def _require_list(
    value: object,
    *,
    name: str,
) -> list[object]:

    if not isinstance(value, list):
        raise ValueError(
            f"{name} must be a list."
        )

    return value


def _require_string(
    value: object,
    *,
    name: str,
) -> str:

    if not isinstance(value, str):
        raise ValueError(
            f"{name} must be a string."
        )

    return value


def _optional_string(
    value: object,
    *,
    name: str,
) -> str | None:

    if value is None:
        return None

    return _require_string(
        value,
        name=name,
    )


def _require_int(
    value: object,
    *,
    name: str,
) -> int:

    # bool is a subclass of int in Python,
    # but must not be accepted here.
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"{name} must be an integer."
        )

    return value


def _require_number(
    value: object,
    *,
    name: str,
) -> float:

    if (
        not isinstance(
            value,
            (int, float),
        )
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"{name} must be numeric."
        )

    return float(value)


def _parse_bbox(
    value: object,
    *,
    name: str,
) -> TableBBox | None:

    if value is None:
        return None

    values = _require_list(
        value,
        name=name,
    )

    if len(values) != 4:
        raise ValueError(
            f"{name} must contain four coordinates."
        )

    return TableBBox(
        x0=_require_number(
            values[0],
            name=f"{name}[0]",
        ),
        y0=_require_number(
            values[1],
            name=f"{name}[1]",
        ),
        x1=_require_number(
            values[2],
            name=f"{name}[2]",
        ),
        y1=_require_number(
            values[3],
            name=f"{name}[3]",
        ),
    )


def _parse_source_span(
    value: object,
    *,
    name: str,
) -> TableSourceSpan:

    raw = _require_dict(
        value,
        name=name,
    )

    try:
        kind = TableSpanKind(
            _require_string(
                raw.get("kind"),
                name=f"{name}.kind",
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"{name}.kind is unsupported."
        ) from exc

    return TableSourceSpan(
        page_number=_require_int(
            raw.get("page_number"),
            name=f"{name}.page_number",
        ),
        kind=kind,
        source_ref=_require_string(
            raw.get("source_ref"),
            name=f"{name}.source_ref",
        ),
        bbox=_parse_bbox(
            raw.get("bbox"),
            name=f"{name}.bbox",
        ),
        text=_optional_string(
            raw.get("text"),
            name=f"{name}.text",
        ),
    )


def _parse_spans(
    value: object,
    *,
    name: str,
) -> tuple[TableSourceSpan, ...]:

    if value is None:
        return ()

    raw_spans = _require_list(
        value,
        name=name,
    )

    return tuple(
        _parse_source_span(
            item,
            name=f"{name}[{index}]",
        )
        for index, item
        in enumerate(raw_spans)
    )


def _parse_cell(
    value: object,
    *,
    name: str,
) -> NormalizedTableCellEvidence:

    raw = _require_dict(
        value,
        name=name,
    )

    try:
        text_status = TableCellTextStatus(
            _require_string(
                raw.get("text_status"),
                name=f"{name}.text_status",
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"{name}.text_status is unsupported."
        ) from exc

    raw_text = raw.get("text")

    if raw_text is not None:
        raw_text = _require_string(
            raw_text,
            name=f"{name}.text",
        )

    return NormalizedTableCellEvidence(
        column_start=_require_int(
            raw.get("column_start"),
            name=f"{name}.column_start",
        ),
        column_end=_require_int(
            raw.get("column_end"),
            name=f"{name}.column_end",
        ),
        text_status=text_status,
        text=raw_text,
        source_spans=_parse_spans(
            raw.get("source_spans"),
            name=f"{name}.source_spans",
        ),
        source_role=_optional_string(
            raw.get("source_role"),
            name=f"{name}.source_role",
        ),
    )


def _parse_row(
    value: object,
    *,
    name: str,
) -> NormalizedTableRowEvidence:

    raw = _require_dict(
        value,
        name=name,
    )

    raw_cells = _require_list(
        raw.get("cells"),
        name=f"{name}.cells",
    )

    header_signal = raw.get(
        "source_header_signal"
    )

    if (
        header_signal is not None
        and not isinstance(
            header_signal,
            bool,
        )
    ):
        raise ValueError(
            f"{name}.source_header_signal "
            "must be boolean or null."
        )

    return NormalizedTableRowEvidence(
        row_index=_require_int(
            raw.get("row_index"),
            name=f"{name}.row_index",
        ),
        cells=tuple(
            _parse_cell(
                cell,
                name=(
                    f"{name}.cells[{index}]"
                ),
            )
            for index, cell
            in enumerate(raw_cells)
        ),
        source_spans=_parse_spans(
            raw.get("source_spans"),
            name=f"{name}.source_spans",
        ),
        source_header_signal=(
            header_signal
        ),
        source_role=_optional_string(
            raw.get("source_role"),
            name=f"{name}.source_role",
        ),
    )


def _parse_table(
    value: object,
    *,
    name: str,
) -> NormalizedTableEvidence:

    raw = _require_dict(
        value,
        name=name,
    )

    try:
        layer = TableEvidenceLayer(
            _require_string(
                raw.get("layer"),
                name=f"{name}.layer",
            )
        )
    except ValueError as exc:
        raise ValueError(
            f"{name}.layer is unsupported."
        ) from exc

    raw_rows = _require_list(
        raw.get("rows"),
        name=f"{name}.rows",
    )

    return NormalizedTableEvidence(
        source_ref=_require_string(
            raw.get("source_ref"),
            name=f"{name}.source_ref",
        ),
        page_number=_require_int(
            raw.get("page_number"),
            name=f"{name}.page_number",
        ),
        layer=layer,
        column_count=_require_int(
            raw.get("column_count"),
            name=f"{name}.column_count",
        ),
        rows=tuple(
            _parse_row(
                row,
                name=(
                    f"{name}.rows[{index}]"
                ),
            )
            for index, row
            in enumerate(raw_rows)
        ),
        bbox=_parse_bbox(
            raw.get("bbox"),
            name=f"{name}.bbox",
        ),
        source_spans=_parse_spans(
            raw.get("source_spans"),
            name=f"{name}.source_spans",
        ),
    )


def _serialize_bbox(
    bbox: TableBBox | None,
) -> list[float] | None:
    if bbox is None:
        return None

    return [
        bbox.x0,
        bbox.y0,
        bbox.x1,
        bbox.y1,
    ]


def _serialize_source_span(
    span: TableSourceSpan,
) -> dict[str, object]:
    return {
        "page_number": (
            span.page_number
        ),
        "kind": (
            span.kind.value
        ),
        "source_ref": (
            span.source_ref
        ),
        "bbox": (
            _serialize_bbox(
                span.bbox
            )
        ),
        "text": (
            span.text
        ),
    }


def _serialize_spans(
    spans: tuple[TableSourceSpan, ...],
) -> list[dict[str, object]]:
    return [
        _serialize_source_span(
            span
        )
        for span in spans
    ]


def _serialize_cell(
    cell: NormalizedTableCellEvidence,
) -> dict[str, object]:
    return {
        "column_start": (
            cell.column_start
        ),
        "column_end": (
            cell.column_end
        ),
        "text_status": (
            cell.text_status.value
        ),
        "text": (
            cell.text
        ),
        "source_spans": (
            _serialize_spans(
                cell.source_spans
            )
        ),
        "source_role": (
            cell.source_role
        ),
    }


def _serialize_row(
    row: NormalizedTableRowEvidence,
) -> dict[str, object]:
    return {
        "row_index": (
            row.row_index
        ),
        "cells": [
            _serialize_cell(
                cell
            )
            for cell in row.cells
        ],
        "source_spans": (
            _serialize_spans(
                row.source_spans
            )
        ),
        "source_header_signal": (
            row.source_header_signal
        ),
        "source_role": (
            row.source_role
        ),
    }


def _serialize_table(
    table: NormalizedTableEvidence,
) -> dict[str, object]:
    return {
        "source_ref": (
            table.source_ref
        ),
        "page_number": (
            table.page_number
        ),
        "layer": (
            table.layer.value
        ),
        "column_count": (
            table.column_count
        ),
        "rows": [
            _serialize_row(
                row
            )
            for row in table.rows
        ],
        "bbox": (
            _serialize_bbox(
                table.bbox
            )
        ),
        "source_spans": (
            _serialize_spans(
                table.source_spans
            )
        ),
    }


def serialize_table_evidence(
    evidence: tuple[
        NormalizedTableEvidence,
        ...,
    ],
) -> dict[str, object]:
    """Serialize normalized table evidence for worker transport."""
    return {
        "table_evidence_schema_version": (
            TABLE_EVIDENCE_SCHEMA_VERSION
        ),
        "table_evidence": [
            _serialize_table(
                table
            )
            for table in evidence
        ],
    }


def deserialize_table_evidence(
    raw_document: dict[str, object],
) -> tuple[NormalizedTableEvidence, ...]:
    """Deserialize optional structured table evidence.

    Legacy workers that emit only pages/chunks remain valid.
    """

    raw_evidence = raw_document.get(
        "table_evidence"
    )

    if raw_evidence is None:
        return ()

    version = raw_document.get(
        "table_evidence_schema_version"
    )

    if version != TABLE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported table evidence schema version."
        )

    items = _require_list(
        raw_evidence,
        name="table_evidence",
    )

    return tuple(
        _parse_table(
            item,
            name=f"table_evidence[{index}]",
        )
        for index, item
        in enumerate(items)
    )
