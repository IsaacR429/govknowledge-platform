from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


F01_SHA256 = (
    "eafe731a1bb8d1ce0478f3b2942b8cb"
    "81ca2ee827d8869b8098cbabfcf203ed8"
)

EXPECTED_FILES = {
    "F01": "carta_nu_deep_inherited_mediabox.pdf",
    "F02": "page_tree_count_mismatch.pdf",
    "F03": "page_tree_fragmented_parent_chain.pdf",
    "F04": "rotated_pages_90_180_270.pdf",
    "F05": "recoverable_damaged_xref.pdf",
    "F06": "ocr_only_scan.pdf",
    "F07": "mixed_text_and_scan.pdf",
}


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

COUNT_RE = re.compile(
    rb"/Count\s+(-?\d+)"
)

KIDS_RE = re.compile(
    rb"/Kids\s*\[(.*?)\]",
    re.DOTALL,
)

REF_RE = re.compile(
    rb"(\d+)\s+0\s+R"
)

ROTATE_RE = re.compile(
    rb"/Rotate\s+(-?\d+)"
)

CONTENTS_RE = re.compile(
    rb"/Contents\s+(\d+)\s+0\s+R"
)

STARTXREF_RE = re.compile(
    rb"startxref\s+(\d+)\s+%%EOF\s*$",
    re.DOTALL,
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dictionary_part(body: bytes) -> bytes:
    marker = re.search(
        rb"\r?\nstream(?:\r?\n|\r)",
        body,
    )

    if marker is None:
        return body

    return body[: marker.start()]


def parse_objects(data: bytes) -> dict[int, bytes]:
    objects: dict[int, bytes] = {}

    for match in OBJECT_RE.finditer(data):
        number = int(match.group(1))
        objects[number] = match.group(2)

    if not objects:
        raise AssertionError(
            "No classic PDF indirect objects found."
        )

    return objects


def page_objects(
    objects: dict[int, bytes],
) -> list[int]:
    result = []

    for number, body in objects.items():
        dictionary = dictionary_part(body)

        if TYPE_PAGE_RE.search(dictionary):
            result.append(number)

    return sorted(result)


def parent_of(
    objects: dict[int, bytes],
    object_number: int,
) -> int | None:
    body = dictionary_part(objects[object_number])
    match = PARENT_RE.search(body)

    if match is None:
        return None

    return int(match.group(1))


def mediabox_of(
    objects: dict[int, bytes],
    object_number: int,
) -> tuple[float, float, float, float] | None:
    body = dictionary_part(objects[object_number])
    match = MEDIABOX_RE.search(body)

    if match is None:
        return None

    return tuple(
        float(value)
        for value in match.groups()
    )


def resolve_mediabox(
    objects: dict[int, bytes],
    page_number: int,
) -> tuple[
    tuple[float, float, float, float],
    int,
    list[int],
]:
    current = page_number
    depth = 0
    chain = [current]
    seen = set()

    while True:
        if current in seen:
            raise AssertionError(
                f"Cycle in parent chain for page "
                f"object {page_number}: {chain}"
            )

        seen.add(current)

        if current not in objects:
            raise AssertionError(
                f"Missing object {current} in parent "
                f"chain for page {page_number}"
            )

        box = mediabox_of(
            objects,
            current,
        )

        if box is not None:
            return box, depth, chain

        parent = parent_of(
            objects,
            current,
        )

        if parent is None:
            raise AssertionError(
                f"MediaBox unresolved for page "
                f"object {page_number}; chain={chain}"
            )

        current = parent
        depth += 1
        chain.append(current)


def kids_of(
    objects: dict[int, bytes],
    object_number: int,
) -> list[int]:
    body = dictionary_part(objects[object_number])
    match = KIDS_RE.search(body)

    if match is None:
        return []

    return [
        int(ref)
        for ref in REF_RE.findall(match.group(1))
    ]


def contents_of(
    objects: dict[int, bytes],
    page_number: int,
) -> int:
    body = dictionary_part(objects[page_number])
    match = CONTENTS_RE.search(body)

    if match is None:
        raise AssertionError(
            f"Page object {page_number} "
            "has no single /Contents reference."
        )

    return int(match.group(1))


def rotation_of(
    objects: dict[int, bytes],
    page_number: int,
) -> int:
    body = dictionary_part(objects[page_number])
    match = ROTATE_RE.search(body)

    if match is None:
        return 0

    return int(match.group(1))


def stream_payload(body: bytes) -> bytes:
    match = re.search(
        rb"\r?\nstream(?:\r?\n|\r)",
        body,
    )

    if match is None:
        raise AssertionError(
            "Expected stream object."
        )

    start = match.end()

    end_match = re.search(
        rb"\r?\nendstream",
        body[start:],
    )

    if end_match is None:
        raise AssertionError(
            "Stream has no endstream marker."
        )

    end = start + end_match.start()

    return body[start:end]


def assert_count(
    actual: int,
    expected: int,
    label: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"{label}: expected {expected}, got {actual}"
        )


def audit_f01(path: Path) -> None:
    data = path.read_bytes()

    if sha256_bytes(data) != F01_SHA256:
        raise AssertionError(
            "F01 is not byte-identical to locked Carta_NU.pdf."
        )

    objects = parse_objects(data)
    pages = page_objects(objects)

    assert_count(
        len(pages),
        34,
        "F01 page leaf count",
    )

    deep_inheritance = []

    for page in pages:
        box, depth, chain = resolve_mediabox(
            objects,
            page,
        )

        width = abs(box[2] - box[0])
        height = abs(box[3] - box[1])

        if (
            abs(width - 612.0) >= 0.01
            or abs(height - 792.0) >= 0.01
        ):
            raise AssertionError(
                f"F01 page object {page}: "
                f"unexpected MediaBox {box}"
            )

        if depth >= 2:
            deep_inheritance.append(
                (page, depth, chain)
            )

    if not deep_inheritance:
        raise AssertionError(
            "F01 does not contain a page whose "
            "MediaBox is inherited through at least "
            "two parent edges."
        )

    example = deep_inheritance[0]

    print(
        "F01: PASS - "
        f"34 page leaves; deep MediaBox inheritance "
        f"confirmed (page object {example[0]}, "
        f"depth={example[1]}, chain={example[2]})"
    )


def audit_f02(path: Path) -> None:
    data = path.read_bytes()
    objects = parse_objects(data)
    pages = page_objects(objects)

    assert_count(
        len(pages),
        3,
        "F02 page leaf count",
    )

    kids = kids_of(objects, 2)

    if kids != [3, 4, 5]:
        raise AssertionError(
            f"F02 object 2 unexpected Kids: {kids}"
        )

    body = dictionary_part(objects[2])
    match = COUNT_RE.search(body)

    if match is None:
        raise AssertionError(
            "F02 object 2 has no /Count."
        )

    declared_count = int(match.group(1))

    if declared_count != 99:
        raise AssertionError(
            f"F02 expected /Count 99, "
            f"got {declared_count}"
        )

    print(
        "F02: PASS - "
        "3 page leaves with deliberately wrong "
        "/Pages /Count 99"
    )


def audit_f03(path: Path) -> None:
    data = path.read_bytes()
    objects = parse_objects(data)
    pages = page_objects(objects)

    assert_count(
        len(pages),
        3,
        "F03 page leaf count",
    )

    kids3 = kids_of(objects, 3)

    if kids3 != [5, 6]:
        raise AssertionError(
            f"F03 object 3 unexpected Kids: {kids3}"
        )

    parent6 = parent_of(objects, 6)

    if parent6 != 4:
        raise AssertionError(
            f"F03 object 6 expected /Parent 4, "
            f"got {parent6}"
        )

    if 6 not in kids3:
        raise AssertionError(
            "F03 object 6 is not actually listed "
            "as a child of object 3."
        )

    print(
        "F03: PASS - "
        "object 6 is listed under /Kids of object 3 "
        "but declares /Parent 4 0 R"
    )


def audit_f04(path: Path) -> None:
    data = path.read_bytes()
    objects = parse_objects(data)

    root_kids = kids_of(objects, 2)

    if root_kids != [3, 4, 5, 6]:
        raise AssertionError(
            f"F04 unexpected page order: {root_kids}"
        )

    rotations = [
        rotation_of(objects, page)
        for page in root_kids
    ]

    expected = [0, 90, 180, 270]

    if rotations != expected:
        raise AssertionError(
            f"F04 rotations expected {expected}, "
            f"got {rotations}"
        )

    print(
        "F04: PASS - "
        "page rotations are exactly "
        "0, 90, 180, 270 in order"
    )


def audit_f05(path: Path) -> None:
    data = path.read_bytes()
    objects = parse_objects(data)
    pages = page_objects(objects)

    assert_count(
        len(pages),
        3,
        "F05 page leaf count",
    )

    actual_xref = data.find(b"xref\n")

    if actual_xref < 0:
        actual_xref = data.find(b"xref\r\n")

    if actual_xref < 0:
        raise AssertionError(
            "F05 has no classic xref table."
        )

    match = STARTXREF_RE.search(data)

    if match is None:
        raise AssertionError(
            "F05 has no parseable startxref."
        )

    declared_xref = int(match.group(1))
    delta = declared_xref - actual_xref

    if delta != 17:
        raise AssertionError(
            f"F05 expected startxref delta +17, "
            f"got {delta}; declared={declared_xref}, "
            f"actual={actual_xref}"
        )

    print(
        "F05: PASS - "
        f"xref table exists at byte {actual_xref}; "
        f"startxref deliberately points to "
        f"{declared_xref} (+17)"
    )


def audit_f06(path: Path) -> None:
    data = path.read_bytes()
    objects = parse_objects(data)
    pages = page_objects(objects)

    assert_count(
        len(pages),
        2,
        "F06 page leaf count",
    )

    image_objects = []

    for number, body in objects.items():
        dictionary = dictionary_part(body)

        if re.search(
            rb"/Subtype\s*/Image\b",
            dictionary,
        ):
            image_objects.append(number)

    if image_objects != [5, 6]:
        raise AssertionError(
            f"F06 expected image objects [5, 6], "
            f"got {image_objects}"
        )

    for page in pages:
        content_number = contents_of(
            objects,
            page,
        )

        payload = stream_payload(
            objects[content_number]
        )

        if b"/Im1 Do" not in payload:
            raise AssertionError(
                f"F06 page object {page} "
                "does not invoke image XObject."
            )

        if (
            re.search(rb"\bBT\b", payload)
            or re.search(rb"\bTj\b", payload)
            or re.search(rb"\bTJ\b", payload)
        ):
            raise AssertionError(
                f"F06 page object {page} "
                "contains native PDF text operators."
            )

        page_dictionary = dictionary_part(
            objects[page]
        )

        if b"/Font" in page_dictionary:
            raise AssertionError(
                f"F06 page object {page} "
                "unexpectedly has a Font resource."
            )

    print(
        "F06: PASS - "
        "2 image-only pages; image XObjects present; "
        "page content has no native BT/Tj/TJ text"
    )


def audit_f07(path: Path) -> None:
    data = path.read_bytes()
    objects = parse_objects(data)

    pages = kids_of(objects, 2)

    if pages != [3, 4, 5, 6]:
        raise AssertionError(
            f"F07 unexpected page order: {pages}"
        )

    native_pages = []
    image_pages = []

    for index, page in enumerate(
        pages,
        start=1,
    ):
        content_number = contents_of(
            objects,
            page,
        )

        payload = stream_payload(
            objects[content_number]
        )

        has_text = (
            re.search(rb"\bBT\b", payload)
            is not None
            and re.search(rb"\bTj\b", payload)
            is not None
        )

        has_image = b"/Im1 Do" in payload

        if has_text and has_image:
            raise AssertionError(
                f"F07 page {index} unexpectedly "
                "contains both paths."
            )

        if has_text:
            native_pages.append(index)
        elif has_image:
            image_pages.append(index)
        else:
            raise AssertionError(
                f"F07 page {index} exercises "
                "neither native text nor image path."
            )

    if native_pages != [1, 3]:
        raise AssertionError(
            f"F07 native text pages expected "
            f"[1, 3], got {native_pages}"
        )

    if image_pages != [2, 4]:
        raise AssertionError(
            f"F07 image pages expected "
            f"[2, 4], got {image_pages}"
        )

    print(
        "F07: PASS - "
        "native text on pages 1/3 and "
        "raster image content on pages 2/4"
    )


def audit_directory(fixtures_dir: Path) -> None:
    fixtures_dir = fixtures_dir.resolve()

    print(
        "===== INDEPENDENT PARSER-RESILIENCE "
        "FIXTURE AUDIT ====="
    )
    print(f"FIXTURES: {fixtures_dir}")
    print()

    expected_names = set(
        EXPECTED_FILES.values()
    )

    actual_names = {
        path.name
        for path in fixtures_dir.glob("*.pdf")
    }

    if actual_names != expected_names:
        raise AssertionError(
            "Fixture filenames differ from frozen "
            f"F01-F07 set. expected={sorted(expected_names)}, "
            f"actual={sorted(actual_names)}"
        )

    audit_f01(
        fixtures_dir / EXPECTED_FILES["F01"]
    )
    audit_f02(
        fixtures_dir / EXPECTED_FILES["F02"]
    )
    audit_f03(
        fixtures_dir / EXPECTED_FILES["F03"]
    )
    audit_f04(
        fixtures_dir / EXPECTED_FILES["F04"]
    )
    audit_f05(
        fixtures_dir / EXPECTED_FILES["F05"]
    )
    audit_f06(
        fixtures_dir / EXPECTED_FILES["F06"]
    )
    audit_f07(
        fixtures_dir / EXPECTED_FILES["F07"]
    )

    print()
    print("STRUCTURAL AUDIT: 7/7 PASS")
    print("RESULT: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    try:
        audit_directory(
            args.fixtures_dir,
        )
    except BaseException as exc:
        print()
        print("RESULT: FAIL")
        print(
            f"EXCEPTION: "
            f"{type(exc).__name__}: {exc}"
        )
        raise


if __name__ == "__main__":
    main()
