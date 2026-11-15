from __future__ import annotations

from typing import Any


TABLE_EVIDENCE_SCHEMA_VERSION = 1


def _bbox(
    value: object,
) -> list[float] | None:
    if value is None:
        return None

    if all(
        hasattr(value, attr)
        for attr in (
            "x0",
            "y0",
            "x1",
            "y1",
        )
    ):
        values = [
            float(value.x0),
            float(value.y0),
            float(value.x1),
            float(value.y1),
        ]

    else:
        try:
            raw_values = list(value)
        except TypeError as exc:
            raise ValueError(
                "PyMuPDF table bbox is invalid."
            ) from exc

        if len(raw_values) != 4:
            raise ValueError(
                "PyMuPDF table bbox must contain "
                "four coordinates."
            )

        values = [
            float(item)
            for item in raw_values
        ]

    x0, y0, x1, y1 = values

    if x1 < x0 or y1 < y0:
        raise ValueError(
            "PyMuPDF table bbox coordinates are invalid."
        )

    return values


def _geometry_span(
    *,
    page_number: int,
    source_ref: str,
    bbox: list[float] | None,
) -> dict[str, object] | None:
    if bbox is None:
        return None

    return {
        "page_number":
            page_number,
        "kind":
            "GEOMETRY",
        "source_ref":
            source_ref,
        "bbox":
            bbox,
        "text":
            None,
    }


def _text_span(
    *,
    page_number: int,
    source_ref: str,
    bbox: list[float] | None,
    text: str,
) -> dict[str, object]:
    return {
        "page_number":
            page_number,
        "kind":
            "TEXT",
        "source_ref":
            source_ref,
        "bbox":
            bbox,
        "text":
            text,
    }


def _row_bbox(
    cell_bboxes: list[
        list[float] | None
    ],
) -> list[float] | None:
    usable = [
        bbox
        for bbox in cell_bboxes
        if bbox is not None
    ]

    if not usable:
        return None

    return [
        min(item[0] for item in usable),
        min(item[1] for item in usable),
        max(item[2] for item in usable),
        max(item[3] for item in usable),
    ]



def _extract_page_words(
    page: object,
) -> tuple[
    tuple[
        float,
        float,
        float,
        float,
        str,
    ],
    ...
]:
    """Return usable PyMuPDF word observations.

    Missing / malformed word geometry is tolerated.
    The downstream row aligner must fail closed if
    text geometry is required but unavailable.
    """

    get_text = getattr(
        page,
        "get_text",
        None,
    )

    if not callable(get_text):
        return ()

    try:
        raw_words = get_text(
            "words"
        )
    except Exception:
        return ()

    if not isinstance(
        raw_words,
        (list, tuple),
    ):
        return ()

    words = []

    for raw_word in raw_words:

        if (
            not isinstance(
                raw_word,
                (list, tuple),
            )
            or len(raw_word) < 5
        ):
            continue

        try:
            x0 = float(
                raw_word[0]
            )
            y0 = float(
                raw_word[1]
            )
            x1 = float(
                raw_word[2]
            )
            y1 = float(
                raw_word[3]
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        raw_text = raw_word[4]

        if not isinstance(
            raw_text,
            str,
        ):
            continue

        word_text = (
            raw_text.strip()
        )

        if not word_text:
            continue

        if (
            x1 < x0
            or y1 < y0
        ):
            continue

        words.append(
            (
                x0,
                y0,
                x1,
                y1,
                word_text,
            )
        )

    return tuple(
        words
    )


def _content_bbox(
    words: tuple[
        tuple[
            float,
            float,
            float,
            float,
            str,
        ],
        ...
    ],
    cell_bbox: list[float] | None,
) -> list[float] | None:
    """Return the actual text extent inside one cell.

    Word centres are used for cell membership so words
    touching a grid boundary are not assigned twice.
    """

    if cell_bbox is None:
        return None

    cell_x0, cell_y0, cell_x1, cell_y1 = (
        cell_bbox
    )

    selected = []

    for (
        word_x0,
        word_y0,
        word_x1,
        word_y1,
        _,
    ) in words:

        center_x = (
            word_x0 + word_x1
        ) / 2.0

        center_y = (
            word_y0 + word_y1
        ) / 2.0

        if (
            cell_x0 <= center_x <= cell_x1
            and cell_y0 <= center_y <= cell_y1
        ):
            selected.append(
                (
                    word_x0,
                    word_y0,
                    word_x1,
                    word_y1,
                )
            )

    if not selected:
        return None

    return [
        min(
            item[0]
            for item in selected
        ),
        min(
            item[1]
            for item in selected
        ),
        max(
            item[2]
            for item in selected
        ),
        max(
            item[3]
            for item in selected
        ),
    ]


def _table_header_signal(
    table: object,
) -> bool | None:
    header = getattr(
        table,
        "header",
        None,
    )

    if header is None:
        return None

    external = getattr(
        header,
        "external",
        None,
    )

    if not isinstance(
        external,
        bool,
    ):
        return None

    # external=False means PyMuPDF considers the
    # header to be inside the extracted table.
    #
    # This remains only a SOURCE signal.
    # Canonicalization does not trust it blindly.
    return not external


def extract_physical_table_evidence(
    page: object,
    *,
    page_number: int,
) -> list[dict[str, object]]:
    """Extract generic physical table observations.

    No canonical semantics are inferred here.

    This function does NOT:
    - join tables across pages,
    - repair columns,
    - forward-fill blank cells,
    - classify sections/summaries,
    - infer hierarchy,
    - call an LLM.
    """

    if page_number < 1:
        raise ValueError(
            "page_number must be >= 1."
        )

    find_tables = getattr(
        page,
        "find_tables",
        None,
    )

    if not callable(find_tables):
        raise RuntimeError(
            "PyMuPDF page does not expose find_tables()."
        )

    finder = find_tables()

    raw_tables = getattr(
        finder,
        "tables",
        None,
    )

    if raw_tables is None:
        raise ValueError(
            "PyMuPDF table finder returned no tables collection."
        )

    page_words = (
        _extract_page_words(
            page
        )
    )

    output: list[
        dict[str, object]
    ] = []

    for table_index, table in enumerate(
        raw_tables
    ):
        source_ref = (
            f"native:p{page_number}:"
            f"t{table_index}"
        )

        extracted = table.extract()

        if not isinstance(
            extracted,
            (list, tuple),
        ):
            raise ValueError(
                "PyMuPDF table.extract() must "
                "return a row collection."
            )

        row_count = int(
            getattr(
                table,
                "row_count",
                len(extracted),
            )
        )

        if row_count != len(extracted):
            raise ValueError(
                "PyMuPDF table row_count does not "
                "match extracted row count."
            )

        raw_col_count = getattr(
            table,
            "col_count",
            None,
        )

        if raw_col_count is None:
            col_count = max(
                (
                    len(row)
                    for row in extracted
                    if isinstance(
                        row,
                        (list, tuple),
                    )
                ),
                default=0,
            )
        else:
            col_count = int(
                raw_col_count
            )

        if col_count < 1:
            raise ValueError(
                "PyMuPDF table must contain "
                "at least one column."
            )

        table_bbox = _bbox(
            getattr(
                table,
                "bbox",
                None,
            )
        )

        table_spans = []

        table_geometry = (
            _geometry_span(
                page_number=page_number,
                source_ref=(
                    f"{source_ref}:geometry"
                ),
                bbox=table_bbox,
            )
        )

        if table_geometry is not None:
            table_spans.append(
                table_geometry
            )

        table_rows = getattr(
            table,
            "rows",
            (),
        )

        header_signal = (
            _table_header_signal(
                table
            )
        )

        normalized_rows = []

        for row_index, raw_row in enumerate(
            extracted
        ):
            if not isinstance(
                raw_row,
                (list, tuple),
            ):
                raise ValueError(
                    "PyMuPDF extracted table row "
                    "must be a sequence."
                )

            if len(raw_row) != col_count:
                raise ValueError(
                    "PyMuPDF extracted row width "
                    "does not match table col_count."
                )

            physical_row = (
                table_rows[row_index]
                if row_index
                < len(table_rows)
                else None
            )

            physical_cells = getattr(
                physical_row,
                "cells",
                (),
            )

            normalized_cells = []
            cell_bboxes: list[
                list[float] | None
            ] = []

            for column_index, raw_text in enumerate(
                raw_row
            ):
                cell_ref = (
                    f"{source_ref}:"
                    f"r{row_index}:"
                    f"c{column_index}"
                )

                raw_cell_bbox = (
                    physical_cells[
                        column_index
                    ]
                    if column_index
                    < len(physical_cells)
                    else None
                )

                cell_bbox = _bbox(
                    raw_cell_bbox
                )

                cell_bboxes.append(
                    cell_bbox
                )

                spans = []

                geometry_span = (
                    _geometry_span(
                        page_number=page_number,
                        source_ref=(
                            f"{cell_ref}:geometry"
                        ),
                        bbox=cell_bbox,
                    )
                )

                if geometry_span is not None:
                    spans.append(
                        geometry_span
                    )

                if raw_text is None:
                    text_status = (
                        "UNAVAILABLE"
                    )
                    text = None

                elif isinstance(
                    raw_text,
                    str,
                ):
                    text_status = (
                        "OBSERVED"
                    )
                    text = raw_text.strip()

                    if text:
                        spans.append(
                            _text_span(
                                page_number=(
                                    page_number
                                ),
                                source_ref=(
                                    f"{cell_ref}:text"
                                ),
                                bbox=(
                                    _content_bbox(
                                        page_words,
                                        cell_bbox,
                                    )
                                ),
                                text=text,
                            )
                        )

                else:
                    raise TypeError(
                        "PyMuPDF extracted cell text "
                        "must be a string or None."
                    )

                normalized_cells.append(
                    {
                        "column_start":
                            column_index,
                        "column_end":
                            column_index + 1,
                        "text_status":
                            text_status,
                        "text":
                            text,
                        "source_spans":
                            spans,
                        "source_role":
                            None,
                    }
                )

            row_bbox = _row_bbox(
                cell_bboxes
            )

            row_spans = []

            row_geometry = (
                _geometry_span(
                    page_number=page_number,
                    source_ref=(
                        f"{source_ref}:"
                        f"r{row_index}:geometry"
                    ),
                    bbox=row_bbox,
                )
            )

            if row_geometry is not None:
                row_spans.append(
                    row_geometry
                )

            row_header_signal = (
                header_signal
                if row_index == 0
                else False
            )

            normalized_rows.append(
                {
                    "row_index":
                        row_index,
                    "cells":
                        normalized_cells,
                    "source_spans":
                        row_spans,
                    "source_header_signal":
                        row_header_signal,
                    "source_role": (
                        "native-header"
                        if row_header_signal
                        is True
                        else None
                    ),
                }
            )

        output.append(
            {
                "source_ref":
                    source_ref,
                "page_number":
                    page_number,
                "layer":
                    "PHYSICAL",
                "column_count":
                    col_count,
                "rows":
                    normalized_rows,
                "bbox":
                    table_bbox,
                "source_spans":
                    table_spans,
            }
        )

    return output
