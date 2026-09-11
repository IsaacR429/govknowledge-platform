import argparse
from importlib.metadata import version
import json
from pathlib import Path
import shutil

import rapidocr

from app.infrastructure.parsing.docling_artifacts import (
    BUNDLE_ID,
    CHUNKING_STRATEGY,
    EXPECTED_PACKAGE_VERSIONS,
    LAYOUT_FOLDER,
    LAYOUT_REPOSITORY,
    LAYOUT_REVISION,
    OCR_ASSET_PATHS,
    OCR_ASSET_SHA256,
    OCR_BACKEND,
    OCR_ENGINE,
    OCR_FOLDER,
    OCR_MODEL_FAMILY,
    P0_MAX_CHUNK_CHARS,
    TABLE_FOLDER,
    TABLE_MODEL_PATH,
    TABLE_REPOSITORY,
    TABLE_REVISION,
    compute_bundle_sha256,
    compute_file_sha256,
)


def copy_snapshot(
    source: Path,
    destination: Path,
) -> None:
    if not source.is_dir():
        raise RuntimeError(
            f"Missing source snapshot: {source}"
        )

    shutil.copytree(
        source,
        destination,
        symlinks=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    for package, expected in (
        EXPECTED_PACKAGE_VERSIONS.items()
    ):
        actual = version(package)

        if actual != expected:
            raise RuntimeError(
                f"{package}: expected "
                f"{expected}, found {actual}"
            )

    hub = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
    )

    layout_source = (
        hub
        / (
            "models--docling-project--"
            "docling-layout-heron"
        )
        / "snapshots"
        / LAYOUT_REVISION
    )

    table_source = (
        hub
        / (
            "models--docling-project--"
            "docling-models"
        )
        / "snapshots"
        / TABLE_REVISION
    )

    output = args.output.expanduser().resolve()

    if output.exists():
        shutil.rmtree(output)

    output.mkdir(
        parents=True,
        exist_ok=False,
    )

    copy_snapshot(
        layout_source,
        output / LAYOUT_FOLDER,
    )

    copy_snapshot(
        table_source,
        output / TABLE_FOLDER,
    )

    expected_table_path = (
        output
        / TABLE_FOLDER
        / TABLE_MODEL_PATH
    )

    if not expected_table_path.is_dir():
        raise RuntimeError(
            "Copied TableFormer bundle "
            "has unexpected structure."
        )

    rapidocr_models = (
        Path(
            rapidocr.__file__
        )
        .resolve()
        .parent
        / "models"
    )

    for name, relative_path in (
        OCR_ASSET_PATHS.items()
    ):
        source = (
            rapidocr_models
            / relative_path.name
        )

        if not source.is_file():
            raise RuntimeError(
                "RapidOCR packaged model "
                f"is missing: {source}"
            )

        expected_hash = (
            OCR_ASSET_SHA256[name]
        )

        actual_hash = compute_file_sha256(
            source
        )

        if actual_hash != expected_hash:
            raise RuntimeError(
                "RapidOCR packaged model "
                f"hash mismatch: {name}"
            )

        destination = (
            output
            / OCR_FOLDER
            / relative_path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            source,
            destination,
        )

        copied_hash = compute_file_sha256(
            destination
        )

        if copied_hash != expected_hash:
            raise RuntimeError(
                "Copied RapidOCR model "
                f"hash mismatch: {name}"
            )

    content_hash = (
        compute_bundle_sha256(output)
    )

    manifest = {
        "bundle_id": BUNDLE_ID,
        "content_sha256": content_hash,
        "models": {
            "layout": {
                "repository": (
                    LAYOUT_REPOSITORY
                ),
                "revision": (
                    LAYOUT_REVISION
                ),
            },
            "table": {
                "repository": (
                    TABLE_REPOSITORY
                ),
                "revision": (
                    TABLE_REVISION
                ),
            },
        },
        "ocr": {
            "engine": OCR_ENGINE,
            "backend": OCR_BACKEND,
            "model_family": (
                OCR_MODEL_FAMILY
            ),
            "rec_keys_path": None,
            "assets": {
                name: {
                    "path": (
                        relative_path.as_posix()
                    ),
                    "sha256": (
                        OCR_ASSET_SHA256[name]
                    ),
                }
                for name, relative_path in (
                    OCR_ASSET_PATHS.items()
                )
            },
        },
        "chunking": {
            "strategy": (
                CHUNKING_STRATEGY
            ),
            "max_chunk_chars": (
                P0_MAX_CHUNK_CHARS
            ),
        },
        "packages": (
            EXPECTED_PACKAGE_VERSIONS
        ),
    }

    (
        output / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Bundle: {output}"
    )
    print(
        f"Revision: {content_hash}"
    )


if __name__ == "__main__":
    main()
