from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zlib
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]

EVIDENCE_ROOT = (
    PROJECT_ROOT
    / "documents"
    / "qualification"
    / "parser-resilience"
)

CORPUS_MANIFEST = (
    EVIDENCE_ROOT
    / "govknowledge-parser-corpus-2026-09-14.json"
)

FIXTURE_SPEC = (
    EVIDENCE_ROOT
    / "adversarial-fixtures-spec-2026-09-14.json"
)

DEFAULT_OUTPUT_DIR = EVIDENCE_ROOT / "fixtures"

DEFAULT_GENERATED_MANIFEST = (
    EVIDENCE_ROOT
    / "adversarial-fixtures-manifest-2026-09-14.json"
)

EXPECTED_CORPUS_MANIFEST_SHA256 = (
    "098e6eaa143bab2c0892cfd2c67626ff"
    "4eb10a4eb7d0f9be10a862859bd25ef1"
)

EXPECTED_FIXTURE_SPEC_SHA256 = (
    "fe1c6a1767482f927350f075568e4bec"
    "5b1e05d2b4f56d48cae1302444b260ba"
)

F01_SOURCE_SHA256 = (
    "eafe731a1bb8d1ce0478f3b2942b8cb"
    "81ca2ee827d8869b8098cbabfcf203ed8"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def require_hash(
    path: Path,
    expected: str,
    label: str,
) -> None:
    actual = sha256_file(path)

    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: "
            f"expected {expected}, got {actual}"
        )


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("xb") as handle:
        handle.write(data)


def pdf_escape(text: str) -> str:
    return (
        text
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def stream_object(
    data: bytes,
    entries: str = "",
) -> bytes:
    dictionary = f"<< /Length {len(data)}"

    if entries:
        dictionary += f" {entries}"

    dictionary += " >>"

    return (
        dictionary.encode("ascii")
        + b"\nstream\n"
        + data
        + b"\nendstream"
    )


def text_stream(
    text: str,
    *,
    x: int = 72,
    y: int = 720,
    size: int = 18,
) -> bytes:
    command = (
        "BT\n"
        f"/F1 {size} Tf\n"
        f"{x} {y} Td\n"
        f"({pdf_escape(text)}) Tj\n"
        "ET\n"
    )

    return stream_object(command.encode("ascii"))


def build_pdf(
    objects: Mapping[int, bytes],
    *,
    root_object: int = 1,
    startxref_delta: int = 0,
) -> bytes:
    if not objects:
        raise ValueError("objects cannot be empty")

    highest_object = max(objects)

    output = bytearray(
        b"%PDF-1.4\n"
        b"%\xe2\xe3\xcf\xd3\n"
    )

    offsets = [0] * (highest_object + 1)

    for object_number in range(1, highest_object + 1):
        object_bytes = objects.get(object_number)

        if object_bytes is None:
            continue

        offsets[object_number] = len(output)

        output.extend(
            f"{object_number} 0 obj\n".encode("ascii")
        )
        output.extend(object_bytes)

        if not object_bytes.endswith(b"\n"):
            output.extend(b"\n")

        output.extend(b"endobj\n")

    xref_offset = len(output)

    output.extend(
        f"xref\n0 {highest_object + 1}\n".encode("ascii")
    )
    output.extend(b"0000000000 65535 f \n")

    for object_number in range(1, highest_object + 1):
        offset = offsets[object_number]

        if offset:
            output.extend(
                f"{offset:010d} 00000 n \n".encode("ascii")
            )
        else:
            output.extend(b"0000000000 65535 f \n")

    output.extend(
        (
            "trailer\n"
            f"<< /Size {highest_object + 1} "
            f"/Root {root_object} 0 R >>\n"
            "startxref\n"
            f"{xref_offset + startxref_delta}\n"
            "%%EOF\n"
        ).encode("ascii")
    )

    return bytes(output)


def font_object() -> bytes:
    return (
        b"<< /Type /Font "
        b"/Subtype /Type1 "
        b"/BaseFont /Helvetica >>"
    )


def page_object(
    *,
    parent: int,
    contents: int,
    resources: str,
    rotate: int | None = None,
) -> bytes:
    parts = [
        "<< /Type /Page",
        f"/Parent {parent} 0 R",
        f"/Resources {resources}",
        f"/Contents {contents} 0 R",
    ]

    if rotate is not None:
        parts.append(f"/Rotate {rotate}")

    parts.append(">>")

    return " ".join(parts).encode("ascii")


# Minimal deterministic 5x7 raster font.
GLYPHS = {
    " ": [
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ],
    "A": [
        "01110",
        "10001",
        "10001",
        "11111",
        "10001",
        "10001",
        "10001",
    ],
    "C": [
        "01110",
        "10001",
        "10000",
        "10000",
        "10000",
        "10001",
        "01110",
    ],
    "D": [
        "11110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "11110",
    ],
    "E": [
        "11111",
        "10000",
        "10000",
        "11110",
        "10000",
        "10000",
        "11111",
    ],
    "G": [
        "01110",
        "10001",
        "10000",
        "10111",
        "10001",
        "10001",
        "01110",
    ],
    "I": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "11111",
    ],
    "L": [
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "10000",
        "11111",
    ],
    "M": [
        "10001",
        "11011",
        "10101",
        "10101",
        "10001",
        "10001",
        "10001",
    ],
    "N": [
        "10001",
        "11001",
        "10101",
        "10011",
        "10001",
        "10001",
        "10001",
    ],
    "O": [
        "01110",
        "10001",
        "10001",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    "P": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10000",
        "10000",
        "10000",
    ],
    "R": [
        "11110",
        "10001",
        "10001",
        "11110",
        "10100",
        "10010",
        "10001",
    ],
    "T": [
        "11111",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "W": [
        "10001",
        "10001",
        "10001",
        "10101",
        "10101",
        "10101",
        "01010",
    ],
    "X": [
        "10001",
        "10001",
        "01010",
        "00100",
        "01010",
        "10001",
        "10001",
    ],
    "Y": [
        "10001",
        "10001",
        "01010",
        "00100",
        "00100",
        "00100",
        "00100",
    ],
    "0": [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    "1": [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    "2": [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    "3": [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    "4": [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    "6": [
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
}


def bitmap_text(
    text: str,
    *,
    width: int = 1200,
    height: int = 300,
    scale: int = 8,
) -> tuple[int, int, bytes]:
    pixels = bytearray([255]) * (width * height)

    glyph_width = 5 * scale
    advance = 6 * scale
    glyph_height = 7 * scale

    total_width = max(
        0,
        len(text) * advance - scale,
    )

    start_x = max(
        20,
        (width - total_width) // 2,
    )

    start_y = max(
        20,
        (height - glyph_height) // 2,
    )

    x = start_x

    for char in text:
        glyph = GLYPHS.get(char)

        if glyph is None:
            raise ValueError(
                f"Unsupported raster glyph: {char!r}"
            )

        for row, row_bits in enumerate(glyph):
            for column, bit in enumerate(row_bits):
                if bit != "1":
                    continue

                px = x + column * scale
                py = start_y + row * scale

                for dy in range(scale):
                    destination_y = py + dy

                    if destination_y >= height:
                        continue

                    row_start = destination_y * width

                    for dx in range(scale):
                        destination_x = px + dx

                        if destination_x >= width:
                            continue

                        pixels[
                            row_start + destination_x
                        ] = 0

        x += advance

    return width, height, bytes(pixels)


def image_object(text: str) -> bytes:
    width, height, pixels = bitmap_text(text)

    compressed = zlib.compress(
        pixels,
        level=9,
    )

    entries = (
        "/Type /XObject "
        "/Subtype /Image "
        f"/Width {width} "
        f"/Height {height} "
        "/ColorSpace /DeviceGray "
        "/BitsPerComponent 8 "
        "/Filter /FlateDecode"
    )

    return stream_object(
        compressed,
        entries,
    )


def image_page_content(
    image_name: str = "Im1",
) -> bytes:
    commands = (
        "q\n"
        "500 0 0 125 56 500 cm\n"
        f"/{image_name} Do\n"
        "Q\n"
    )

    return stream_object(
        commands.encode("ascii")
    )


def generate_f02() -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",

        # Deliberately wrong Count: 99 instead of 3.
        2: (
            b"<< /Type /Pages "
            b"/Kids [3 0 R 4 0 R 5 0 R] "
            b"/Count 99 "
            b"/MediaBox [0 0 612 792] >>"
        ),

        3: page_object(
            parent=2,
            contents=7,
            resources="<< /Font << /F1 6 0 R >> >>",
        ),
        4: page_object(
            parent=2,
            contents=8,
            resources="<< /Font << /F1 6 0 R >> >>",
        ),
        5: page_object(
            parent=2,
            contents=9,
            resources="<< /Font << /F1 6 0 R >> >>",
        ),

        6: font_object(),

        7: text_stream("F02 PAGE 1 COUNT MISMATCH"),
        8: text_stream("F02 PAGE 2 COUNT MISMATCH"),
        9: text_stream("F02 PAGE 3 COUNT MISMATCH"),
    }

    return build_pdf(objects)


def generate_f03() -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",

        2: (
            b"<< /Type /Pages "
            b"/Kids [3 0 R 4 0 R] "
            b"/Count 3 "
            b"/MediaBox [0 0 612 792] >>"
        ),

        3: (
            b"<< /Type /Pages "
            b"/Parent 2 0 R "
            b"/Kids [5 0 R 6 0 R] "
            b"/Count 2 >>"
        ),

        4: (
            b"<< /Type /Pages "
            b"/Parent 2 0 R "
            b"/Kids [7 0 R] "
            b"/Count 1 >>"
        ),

        5: page_object(
            parent=3,
            contents=9,
            resources="<< /Font << /F1 8 0 R >> >>",
        ),

        # Deliberate inconsistency:
        # this page is a Kid of object 3,
        # but claims object 4 as its Parent.
        6: page_object(
            parent=4,
            contents=10,
            resources="<< /Font << /F1 8 0 R >> >>",
        ),

        7: page_object(
            parent=4,
            contents=11,
            resources="<< /Font << /F1 8 0 R >> >>",
        ),

        8: font_object(),

        9: text_stream("F03 PAGE 1 PARENT CONTROL"),
        10: text_stream("F03 PAGE 2 PARENT MISMATCH"),
        11: text_stream("F03 PAGE 3 PARENT CONTROL"),
    }

    return build_pdf(objects)


def generate_f04() -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",

        2: (
            b"<< /Type /Pages "
            b"/Kids [3 0 R 4 0 R 5 0 R 6 0 R] "
            b"/Count 4 "
            b"/MediaBox [0 0 612 792] >>"
        ),

        3: page_object(
            parent=2,
            contents=8,
            resources="<< /Font << /F1 7 0 R >> >>",
            rotate=0,
        ),
        4: page_object(
            parent=2,
            contents=9,
            resources="<< /Font << /F1 7 0 R >> >>",
            rotate=90,
        ),
        5: page_object(
            parent=2,
            contents=10,
            resources="<< /Font << /F1 7 0 R >> >>",
            rotate=180,
        ),
        6: page_object(
            parent=2,
            contents=11,
            resources="<< /Font << /F1 7 0 R >> >>",
            rotate=270,
        ),

        7: font_object(),

        8: text_stream("F04 PAGE 1 ROTATE 0"),
        9: text_stream("F04 PAGE 2 ROTATE 90"),
        10: text_stream("F04 PAGE 3 ROTATE 180"),
        11: text_stream("F04 PAGE 4 ROTATE 270"),
    }

    return build_pdf(objects)


def generate_f05() -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",

        2: (
            b"<< /Type /Pages "
            b"/Kids [3 0 R 4 0 R 5 0 R] "
            b"/Count 3 "
            b"/MediaBox [0 0 612 792] >>"
        ),

        3: page_object(
            parent=2,
            contents=7,
            resources="<< /Font << /F1 6 0 R >> >>",
        ),
        4: page_object(
            parent=2,
            contents=8,
            resources="<< /Font << /F1 6 0 R >> >>",
        ),
        5: page_object(
            parent=2,
            contents=9,
            resources="<< /Font << /F1 6 0 R >> >>",
        ),

        6: font_object(),

        7: text_stream("F05 PAGE 1 XREF CONTROL"),
        8: text_stream("F05 PAGE 2 XREF CONTROL"),
        9: text_stream("F05 PAGE 3 XREF CONTROL"),
    }

    # Deliberately wrong startxref pointer.
    # The xref table itself remains intact so the
    # damage is recoverable by a repairing reader.
    return build_pdf(
        objects,
        startxref_delta=17,
    )


def generate_f06() -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",

        2: (
            b"<< /Type /Pages "
            b"/Kids [3 0 R 4 0 R] "
            b"/Count 2 "
            b"/MediaBox [0 0 612 792] >>"
        ),

        3: page_object(
            parent=2,
            contents=7,
            resources="<< /XObject << /Im1 5 0 R >> >>",
        ),
        4: page_object(
            parent=2,
            contents=8,
            resources="<< /XObject << /Im1 6 0 R >> >>",
        ),

        5: image_object(
            "OCR ONLY PAGE ONE 2026"
        ),
        6: image_object(
            "OCR ONLY PAGE TWO 2026"
        ),

        7: image_page_content(),
        8: image_page_content(),
    }

    return build_pdf(objects)


def generate_f07() -> bytes:
    objects = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",

        2: (
            b"<< /Type /Pages "
            b"/Kids [3 0 R 4 0 R 5 0 R 6 0 R] "
            b"/Count 4 "
            b"/MediaBox [0 0 612 792] >>"
        ),

        3: page_object(
            parent=2,
            contents=10,
            resources="<< /Font << /F1 7 0 R >> >>",
        ),
        4: page_object(
            parent=2,
            contents=11,
            resources="<< /XObject << /Im1 8 0 R >> >>",
        ),
        5: page_object(
            parent=2,
            contents=12,
            resources="<< /Font << /F1 7 0 R >> >>",
        ),
        6: page_object(
            parent=2,
            contents=13,
            resources="<< /XObject << /Im1 9 0 R >> >>",
        ),

        7: font_object(),

        8: image_object(
            "MIXED OCR PAGE 2"
        ),
        9: image_object(
            "MIXED OCR PAGE 4"
        ),

        10: text_stream(
            "MIXED NATIVE PAGE 1"
        ),
        11: image_page_content(),
        12: text_stream(
            "MIXED NATIVE PAGE 3"
        ),
        13: image_page_content(),
    }

    return build_pdf(objects)


def locate_f01_source(
    corpus_manifest: dict,
    explicit_path: Path | None,
) -> Path:
    if explicit_path is not None:
        candidate = explicit_path.resolve()

        if not candidate.is_file():
            raise FileNotFoundError(candidate)

        require_hash(
            candidate,
            F01_SOURCE_SHA256,
            "explicit F01 source",
        )

        return candidate

    candidates: list[Path] = []

    for item in corpus_manifest["physical_files"]:
        if item["sha256"] != F01_SOURCE_SHA256:
            continue

        full_path = Path(item["full_path"])
        candidates.append(full_path)

        source_root = Path(
            corpus_manifest["source_root"]
        )

        candidates.append(
            source_root / item["relative_path"]
        )

    for candidate in candidates:
        if not candidate.is_file():
            continue

        if sha256_file(candidate) == F01_SOURCE_SHA256:
            return candidate

    raise RuntimeError(
        "Could not locate the locked F01 Carta_NU.pdf "
        "source. Supply --f01-source explicitly."
    )


def load_locked_inputs() -> tuple[dict, dict]:
    require_hash(
        CORPUS_MANIFEST,
        EXPECTED_CORPUS_MANIFEST_SHA256,
        "corpus manifest",
    )

    require_hash(
        FIXTURE_SPEC,
        EXPECTED_FIXTURE_SPEC_SHA256,
        "fixture specification",
    )

    corpus = json.loads(
        CORPUS_MANIFEST.read_text(
            encoding="utf-8-sig"
        )
    )

    spec = json.loads(
        FIXTURE_SPEC.read_text(
            encoding="utf-8"
        )
    )

    return corpus, spec


def generate(
    *,
    output_dir: Path,
    manifest_path: Path,
    f01_source: Path | None,
) -> None:
    corpus, spec = load_locked_inputs()

    fixture_entries = {
        item["fixture_id"]: item
        for item in spec["fixtures"]
    }

    expected_ids = [
        "F01",
        "F02",
        "F03",
        "F04",
        "F05",
        "F06",
        "F07",
    ]

    if list(fixture_entries) != expected_ids:
        raise RuntimeError(
            "Frozen fixture IDs are not exactly F01-F07."
        )

    output_dir = output_dir.resolve()
    manifest_path = manifest_path.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    target_paths = {
        fixture_id: (
            output_dir
            / fixture_entries[fixture_id]["filename"]
        )
        for fixture_id in expected_ids
    }

    existing = [
        path
        for path in target_paths.values()
        if path.exists()
    ]

    if manifest_path.exists():
        existing.append(manifest_path)

    if existing:
        rendered = "\n".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Refusing to modify previously generated "
            "fixture artifacts:\n"
            f"{rendered}"
        )

    source_f01 = locate_f01_source(
        corpus,
        f01_source,
    )

    f01_target = target_paths["F01"]

    with source_f01.open("rb") as source:
        write_new(
            f01_target,
            source.read(),
        )

    require_hash(
        f01_target,
        F01_SOURCE_SHA256,
        "generated F01",
    )

    generated = {
        "F02": generate_f02(),
        "F03": generate_f03(),
        "F04": generate_f04(),
        "F05": generate_f05(),
        "F06": generate_f06(),
        "F07": generate_f07(),
    }

    for fixture_id, payload in generated.items():
        write_new(
            target_paths[fixture_id],
            payload,
        )

    records = []

    for fixture_id in expected_ids:
        path = target_paths[fixture_id]
        fixture_spec = fixture_entries[fixture_id]

        records.append(
            {
                "fixture_id": fixture_id,
                "filename": path.name,
                "expected_page_count": (
                    fixture_spec[
                        "expected_page_count"
                    ]
                ),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    generator_hash = sha256_file(
        Path(__file__).resolve()
    )

    manifest = {
        "manifest_id": (
            "govknowledge-parser-adversarial-"
            "fixtures-manifest-2026-09-14"
        ),
        "suite_id": (
            "PARSER-RESILIENCE-REGRESSION-01"
        ),
        "fixture_spec_sha256": (
            EXPECTED_FIXTURE_SPEC_SHA256
        ),
        "source_corpus_manifest_sha256": (
            EXPECTED_CORPUS_MANIFEST_SHA256
        ),
        "generator_version": 1,
        "generator_sha256": generator_hash,
        "deterministic": True,
        "fixture_count": len(records),
        "fixtures": records,
    }

    manifest_bytes = (
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_new(
        manifest_path,
        manifest_bytes,
    )

    print(
        "===== PARSER RESILIENCE FIXTURE GENERATION ====="
    )
    print(f"OUTPUT DIR : {output_dir}")
    print(f"MANIFEST   : {manifest_path}")
    print(
        f"GENERATOR  : {generator_hash}"
    )
    print()

    for record in records:
        print(
            f"{record['fixture_id']}: "
            f"{record['size_bytes']} bytes "
            f"{record['sha256']} "
            f"{record['filename']}"
        )

    print()
    print(
        f"FIXTURES GENERATED: {len(records)}/7"
    )
    print("RESULT: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the frozen GovKnowledge "
            "parser-resilience adversarial fixtures."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_GENERATED_MANIFEST,
    )

    parser.add_argument(
        "--f01-source",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    generate(
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        f01_source=args.f01_source,
    )


if __name__ == "__main__":
    main()
