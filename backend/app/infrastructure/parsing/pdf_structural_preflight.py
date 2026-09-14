from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium


OBJECT_RE = re.compile(
    rb"(?ms)(?:^|\r\n|\r|\n)"
    rb"(\d+)\s+0\s+obj\s*"
    rb"(.*?)"
    rb"\s*endobj"
)

TYPE_PAGE_RE = re.compile(
    rb"/Type\s*/Page(?!s)\b"
)

PARENT_RE = re.compile(
    rb"/Parent\s+(\d+)\s+0\s+R"
)

MEDIABOX_RE = re.compile(
    rb"/MediaBox\s*\[\s*"
    rb"([-+]?\d+(?:\.\d+)?)\s+"
    rb"([-+]?\d+(?:\.\d+)?)\s+"
    rb"([-+]?\d+(?:\.\d+)?)\s+"
    rb"([-+]?\d+(?:\.\d+)?)\s*"
    rb"\]"
)


@dataclass(frozen=True)
class PdfStructuralPreflight:
    page_count: int
    page_count_method: str
    page_object_numbers: tuple[int, ...]
    unresolved_mediabox_pages: tuple[int, ...]
    invalid_mediabox_pages: tuple[int, ...]


def _dictionary_part(body: bytes) -> bytes:
    marker = re.search(
        rb"\r?\nstream(?:\r?\n|\r)",
        body,
    )

    if marker is None:
        return body

    return body[:marker.start()]


def _parse_classic_objects(
    data: bytes,
) -> dict[int, bytes]:
    return {
        int(match.group(1)): match.group(2)
        for match in OBJECT_RE.finditer(data)
    }


def _classic_page_objects(
    objects: dict[int, bytes],
) -> tuple[int, ...]:
    return tuple(
        sorted(
            number
            for number, body in objects.items()
            if TYPE_PAGE_RE.search(
                _dictionary_part(body)
            )
        )
    )


def _parent_of(
    objects: dict[int, bytes],
    object_number: int,
) -> int | None:
    body = _dictionary_part(
        objects[object_number]
    )

    match = PARENT_RE.search(body)

    if match is None:
        return None

    return int(match.group(1))


def _mediabox_of(
    objects: dict[int, bytes],
    object_number: int,
) -> tuple[
    float,
    float,
    float,
    float,
] | None:
    body = _dictionary_part(
        objects[object_number]
    )

    match = MEDIABOX_RE.search(body)

    if match is None:
        return None

    return tuple(
        float(value)
        for value in match.groups()
    )


def _resolve_mediabox(
    objects: dict[int, bytes],
    page_object: int,
) -> tuple[
    float,
    float,
    float,
    float,
] | None:
    current = page_object
    seen: set[int] = set()

    while True:
        if current in seen:
            return None

        seen.add(current)

        if current not in objects:
            return None

        box = _mediabox_of(
            objects,
            current,
        )

        if box is not None:
            return box

        parent = _parent_of(
            objects,
            current,
        )

        if parent is None:
            return None

        current = parent


def _pdfium_preflight(
    path: Path,
) -> PdfStructuralPreflight:
    pdf = pdfium.PdfDocument(path)

    try:
        count = len(pdf)

        if count < 1:
            raise ValueError(
                "PDF contains no pages."
            )

        unresolved = []
        invalid = []

        for index in range(count):
            page = pdf[index]

            try:
                box = page.get_mediabox(
                    fallback_ok=True
                )
            finally:
                page.close()

            if box is None:
                unresolved.append(
                    index + 1
                )
                continue

            left, bottom, right, top = box

            if (
                right <= left
                or top <= bottom
            ):
                invalid.append(
                    index + 1
                )

        return PdfStructuralPreflight(
            page_count=count,
            page_count_method=(
                "PDFIUM_FALLBACK"
            ),
            page_object_numbers=(),
            unresolved_mediabox_pages=(
                tuple(unresolved)
            ),
            invalid_mediabox_pages=(
                tuple(invalid)
            ),
        )

    finally:
        pdf.close()


def preflight_pdf(
    path: Path,
) -> PdfStructuralPreflight:
    source = path.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    data = source.read_bytes()

    objects = _parse_classic_objects(
        data
    )

    pages = _classic_page_objects(
        objects
    )

    # When classic page leaves are independently
    # visible, use them rather than trusting the
    # page-tree /Count value.
    if pages:
        unresolved = []
        invalid = []

        for logical_index, page_object in enumerate(
            pages,
            start=1,
        ):
            box = _resolve_mediabox(
                objects,
                page_object,
            )

            if box is None:
                unresolved.append(
                    logical_index
                )
                continue

            left, bottom, right, top = box

            if (
                right <= left
                or top <= bottom
            ):
                invalid.append(
                    logical_index
                )

        return PdfStructuralPreflight(
            page_count=len(pages),
            page_count_method=(
                "CLASSIC_PAGE_LEAF_ENUMERATION"
            ),
            page_object_numbers=pages,
            unresolved_mediabox_pages=(
                tuple(unresolved)
            ),
            invalid_mediabox_pages=(
                tuple(invalid)
            ),
        )

    return _pdfium_preflight(
        source
    )
