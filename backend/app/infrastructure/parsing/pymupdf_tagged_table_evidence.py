from __future__ import annotations

import re


_REF_RE = re.compile(
    r"(\d+)\s+\d+\s+R"
)


def _key_raw(
    document: object,
    xref: int,
    key: str,
) -> str | None:
    try:
        result = document.xref_get_key(
            xref,
            key,
        )
    except Exception:
        return None

    if (
        not isinstance(result, tuple)
        or len(result) != 2
    ):
        return None

    kind, value = result

    if kind in {
        "null",
        None,
    }:
        return None

    if value is None:
        return None

    return str(value)


def _refs(
    raw: str | None,
) -> tuple[int, ...]:
    if not raw:
        return ()

    return tuple(
        int(match.group(1))
        for match in _REF_RE.finditer(
            raw
        )
    )


def _tag(
    document: object,
    xref: int,
) -> str | None:
    raw = _key_raw(
        document,
        xref,
        "S",
    )

    if not raw:
        return None

    raw = raw.strip()

    if raw.startswith("/"):
        raw = raw[1:]

    return raw or None


def _children(
    document: object,
    xref: int,
) -> tuple[int, ...]:
    return _refs(
        _key_raw(
            document,
            xref,
            "K",
        )
    )


def _page_xref_map(
    document: object,
) -> dict[int, int]:
    mapping: dict[int, int] = {}

    page_count = int(
        getattr(
            document,
            "page_count",
            0,
        )
    )

    for offset in range(
        page_count
    ):
        page = document.load_page(
            offset
        )

        page_xref = getattr(
            page,
            "xref",
            None,
        )

        if isinstance(
            page_xref,
            int,
        ):
            mapping[
                page_xref
            ] = offset + 1

    return mapping


def _element_pages(
    document: object,
    xref: int,
    *,
    page_map: dict[int, int],
    cache: dict[int, tuple[int, ...]],
    visiting: set[int] | None = None,
) -> tuple[int, ...]:

    if xref in cache:
        return cache[xref]

    if visiting is None:
        visiting = set()

    if xref in visiting:
        return ()

    visiting = {
        *visiting,
        xref,
    }

    pages: set[int] = set()

    for page_xref in _refs(
        _key_raw(
            document,
            xref,
            "Pg",
        )
    ):
        page_number = (
            page_map.get(
                page_xref
            )
        )

        if page_number is not None:
            pages.add(
                page_number
            )

    for child_xref in _children(
        document,
        xref,
    ):
        pages.update(
            _element_pages(
                document,
                child_xref,
                page_map=page_map,
                cache=cache,
                visiting=visiting,
            )
        )

    result = tuple(
        sorted(pages)
    )

    cache[xref] = result

    return result


def _find_struct_tree_root(
    document: object,
) -> int | None:
    catalog = getattr(
        document,
        "pdf_catalog",
        None,
    )

    key_reader = getattr(
        document,
        "xref_get_key",
        None,
    )

    if (
        not callable(catalog)
        or not callable(key_reader)
    ):
        return None

    try:
        catalog_xref = int(
            catalog()
        )
    except Exception:
        return None

    refs = _refs(
        _key_raw(
            document,
            catalog_xref,
            "StructTreeRoot",
        )
    )

    if not refs:
        return None

    return refs[0]


def _collect_table_xrefs(
    document: object,
    root_xref: int,
) -> tuple[int, ...]:

    tables: list[int] = []
    visited: set[int] = set()

    def walk(
        xref: int,
    ) -> None:
        if xref in visited:
            return

        visited.add(
            xref
        )

        if _tag(
            document,
            xref,
        ) == "Table":
            tables.append(
                xref
            )

            # Nested tables should be independent
            # structural observations.
            for child in _children(
                document,
                xref,
            ):
                if (
                    _tag(
                        document,
                        child,
                    )
                    == "Table"
                ):
                    walk(child)

            return

        for child in _children(
            document,
            xref,
        ):
            walk(
                child
            )

    walk(
        root_xref
    )

    return tuple(
        tables
    )


def _collect_rows(
    document: object,
    table_xref: int,
) -> tuple[int, ...]:

    rows: list[int] = []
    visited: set[int] = set()

    def walk(
        xref: int,
    ) -> None:
        if xref in visited:
            return

        visited.add(
            xref
        )

        tag = _tag(
            document,
            xref,
        )

        if tag == "TR":
            rows.append(
                xref
            )
            return

        if (
            tag == "Table"
            and xref != table_xref
        ):
            return

        for child in _children(
            document,
            xref,
        ):
            walk(
                child
            )

    for child in _children(
        document,
        table_xref,
    ):
        walk(
            child
        )

    return tuple(
        rows
    )


def _collect_cells(
    document: object,
    row_xref: int,
) -> tuple[int, ...]:

    cells: list[int] = []
    visited: set[int] = set()

    def walk(
        xref: int,
    ) -> None:
        if xref in visited:
            return

        visited.add(
            xref
        )

        tag = _tag(
            document,
            xref,
        )

        if tag in {
            "TH",
            "TD",
        }:
            cells.append(
                xref
            )
            return

        if tag in {
            "TR",
            "Table",
        }:
            return

        for child in _children(
            document,
            xref,
        ):
            walk(
                child
            )

    for child in _children(
        document,
        row_xref,
    ):
        walk(
            child
        )

    return tuple(
        cells
    )


def extract_structural_table_evidence(
    document: object,
) -> list[dict[str, object]]:
    """Extract logical table structure from PDF tags.

    This adapter intentionally extracts STRUCTURE only.

    It does not attempt to decode MCIDs into text.
    Physical/native evidence owns observed text and geometry.

    Multi-page tagged tables are emitted as page-local
    structural slices while preserving the original table
    xref in source_ref.
    """

    root_xref = (
        _find_struct_tree_root(
            document
        )
    )

    if root_xref is None:
        return []

    page_map = _page_xref_map(
        document
    )

    page_cache: dict[
        int,
        tuple[int, ...],
    ] = {}

    output: list[
        dict[str, object]
    ] = []

    for table_xref in (
        _collect_table_xrefs(
            document,
            root_xref,
        )
    ):
        row_xrefs = (
            _collect_rows(
                document,
                table_xref,
            )
        )

        if not row_xrefs:
            continue

        cells_by_row = {
            row_xref:
                _collect_cells(
                    document,
                    row_xref,
                )
            for row_xref
            in row_xrefs
        }

        column_count = max(
            (
                len(cells)
                for cells
                in cells_by_row.values()
            ),
            default=0,
        )

        if column_count < 1:
            continue

        table_pages: set[int] = set()

        for row_xref in row_xrefs:
            table_pages.update(
                _element_pages(
                    document,
                    row_xref,
                    page_map=page_map,
                    cache=page_cache,
                )
            )

        if not table_pages:
            table_pages.update(
                _element_pages(
                    document,
                    table_xref,
                    page_map=page_map,
                    cache=page_cache,
                )
            )

        for page_number in sorted(
            table_pages
        ):
            rows_payload = []

            for row_index, row_xref in enumerate(
                row_xrefs
            ):
                row_pages = (
                    _element_pages(
                        document,
                        row_xref,
                        page_map=page_map,
                        cache=page_cache,
                    )
                )

                cells = (
                    cells_by_row[
                        row_xref
                    ]
                )

                cells_payload = []

                cell_roles = []

                for column_index, cell_xref in enumerate(
                    cells
                ):
                    cell_pages = (
                        _element_pages(
                            document,
                            cell_xref,
                            page_map=page_map,
                            cache=page_cache,
                        )
                    )

                    # Some valid tagged PDFs put /Pg on the
                    # row rather than each logical cell.
                    if (
                        not cell_pages
                        and len(row_pages) == 1
                    ):
                        cell_pages = (
                            row_pages
                        )

                    if (
                        page_number
                        not in cell_pages
                    ):
                        continue

                    role = (
                        _tag(
                            document,
                            cell_xref,
                        )
                        or "TD"
                    )

                    cell_roles.append(
                        role
                    )

                    cells_payload.append(
                        {
                            "column_start":
                                column_index,
                            "column_end":
                                column_index + 1,

                            # Tags define structure.
                            # They do not provide portable
                            # visible-text extraction.
                            "text_status":
                                "UNAVAILABLE",
                            "text":
                                None,

                            "source_spans": [
                                {
                                    "page_number":
                                        page_number,
                                    "kind":
                                        "STRUCTURE",
                                    "source_ref": (
                                        f"tag:xref:"
                                        f"{cell_xref}"
                                    ),
                                    "bbox":
                                        None,
                                    "text":
                                        None,
                                }
                            ],

                            "source_role":
                                role,
                        }
                    )

                if not cells_payload:
                    continue

                header_signal = (
                    len(cells_payload)
                    == column_count
                    and all(
                        role == "TH"
                        for role
                        in cell_roles
                    )
                )

                rows_payload.append(
                    {
                        "row_index":
                            row_index,
                        "cells":
                            cells_payload,

                        "source_spans": [
                            {
                                "page_number":
                                    page_number,
                                "kind":
                                    "STRUCTURE",
                                "source_ref": (
                                    f"tag:xref:"
                                    f"{row_xref}"
                                ),
                                "bbox":
                                    None,
                                "text":
                                    None,
                            }
                        ],

                        "source_header_signal":
                            header_signal,
                        "source_role":
                            "TR",
                    }
                )

            if not rows_payload:
                continue

            output.append(
                {
                    "source_ref": (
                        f"tag:xref:"
                        f"{table_xref}:"
                        f"p{page_number}"
                    ),
                    "page_number":
                        page_number,
                    "layer":
                        "STRUCTURAL",
                    "column_count":
                        column_count,
                    "rows":
                        rows_payload,
                    "bbox":
                        None,

                    "source_spans": [
                        {
                            "page_number":
                                page_number,
                            "kind":
                                "STRUCTURE",
                            "source_ref": (
                                f"tag:xref:"
                                f"{table_xref}"
                            ),
                            "bbox":
                                None,
                            "text":
                                None,
                        }
                    ],
                }
            )

    return output
