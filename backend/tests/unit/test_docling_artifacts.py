import json

import pytest

from app.infrastructure.parsing import (
    docling_artifacts as artifacts,
)


def _stub_runtime_versions(
    monkeypatch,
    *,
    wrong_package: str | None = None,
) -> None:
    def fake_version(
        package: str,
    ) -> str:
        if package == wrong_package:
            return "0.0.0"

        return (
            artifacts
            .EXPECTED_PACKAGE_VERSIONS[
                package
            ]
        )

    monkeypatch.setattr(
        artifacts,
        "version",
        fake_version,
    )


def _write_manifest(
    root,
    manifest,
) -> None:
    (
        root
        / artifacts.MANIFEST_NAME
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_fake_bundle(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "bundle"

    root.mkdir()

    layout_root = (
        root
        / artifacts.LAYOUT_FOLDER
    )

    layout_root.mkdir(
        parents=True,
    )

    (
        layout_root
        / "layout.bin"
    ).write_bytes(
        b"fake-layout"
    )

    table_root = (
        root
        / artifacts.TABLE_FOLDER
        / artifacts.TABLE_MODEL_PATH
    )

    table_root.mkdir(
        parents=True,
    )

    (
        table_root
        / "table.bin"
    ).write_bytes(
        b"fake-table"
    )

    fake_ocr_hashes = {}

    for name, relative_path in (
        artifacts.OCR_ASSET_PATHS.items()
    ):
        path = (
            root
            / artifacts.OCR_FOLDER
            / relative_path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_bytes(
            f"fake-{name}-model".encode(
                "utf-8"
            )
        )

        fake_ocr_hashes[name] = (
            artifacts.compute_file_sha256(
                path
            )
        )

    monkeypatch.setattr(
        artifacts,
        "OCR_ASSET_SHA256",
        fake_ocr_hashes,
    )

    content_hash = (
        artifacts.compute_bundle_sha256(
            root
        )
    )

    manifest = {
        "bundle_id": (
            artifacts.BUNDLE_ID
        ),
        "content_sha256": (
            content_hash
        ),
        "models": {
            "layout": {
                "repository": (
                    artifacts
                    .LAYOUT_REPOSITORY
                ),
                "revision": (
                    artifacts
                    .LAYOUT_REVISION
                ),
            },
            "table": {
                "repository": (
                    artifacts
                    .TABLE_REPOSITORY
                ),
                "revision": (
                    artifacts
                    .TABLE_REVISION
                ),
            },
        },
        "ocr": {
            "engine": (
                artifacts.OCR_ENGINE
            ),
            "backend": (
                artifacts.OCR_BACKEND
            ),
            "model_family": (
                artifacts
                .OCR_MODEL_FAMILY
            ),
            "rec_keys_path": None,
            "assets": {
                name: {
                    "path": (
                        relative_path
                        .as_posix()
                    ),
                    "sha256": (
                        fake_ocr_hashes[
                            name
                        ]
                    ),
                }
                for name, relative_path in (
                    artifacts
                    .OCR_ASSET_PATHS
                    .items()
                )
            },
        },
        "chunking": {
            "strategy": (
                artifacts
                .CHUNKING_STRATEGY
            ),
            "max_chunk_chars": (
                artifacts
                .P0_MAX_CHUNK_CHARS
            ),
        },
        "packages": (
            artifacts
            .EXPECTED_PACKAGE_VERSIONS
        ),
    }

    _write_manifest(
        root,
        manifest,
    )

    monkeypatch.setattr(
        artifacts,
        "EXPECTED_BUNDLE_REVISION",
        content_hash,
    )

    return (
        root,
        manifest,
        content_hash,
    )


def test_valid_artifact_bundle_is_accepted(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    (
        root,
        _,
        expected_revision,
    ) = _build_fake_bundle(
        tmp_path,
        monkeypatch,
    )

    bundle = (
        artifacts
        .load_docling_artifact_bundle(
            root
        )
    )

    assert (
        bundle.bundle_id
        == artifacts.BUNDLE_ID
    )

    assert (
        bundle.bundle_revision
        == expected_revision
    )

    assert (
        bundle.ocr_model_family
        == artifacts.OCR_MODEL_FAMILY
    )

    assert (
        bundle.chunking_strategy
        == artifacts.CHUNKING_STRATEGY
    )

    assert (
        bundle.max_chunk_chars
        == artifacts.P0_MAX_CHUNK_CHARS
    )


def test_missing_artifact_environment_is_rejected(
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    monkeypatch.delenv(
        artifacts.ENVIRONMENT_VARIABLE,
        raising=False,
    )

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match="must point to",
    ):
        (
            artifacts
            .load_docling_artifact_bundle()
        )


def test_runtime_package_mismatch_is_rejected(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch,
        wrong_package="docling",
    )

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match=(
            "runtime package version "
            "mismatch"
        ),
    ):
        (
            artifacts
            .load_docling_artifact_bundle(
                tmp_path
            )
        )


def test_wrong_ocr_identity_is_rejected(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    root, manifest, _ = (
        _build_fake_bundle(
            tmp_path,
            monkeypatch,
        )
    )

    manifest[
        "ocr"
    ][
        "model_family"
    ] = "PP-OCRv4"

    _write_manifest(
        root,
        manifest,
    )

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match="OCR configuration",
    ):
        (
            artifacts
            .load_docling_artifact_bundle(
                root
            )
        )


def test_wrong_chunk_configuration_is_rejected(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    root, manifest, _ = (
        _build_fake_bundle(
            tmp_path,
            monkeypatch,
        )
    )

    manifest[
        "chunking"
    ][
        "max_chunk_chars"
    ] = 3000

    _write_manifest(
        root,
        manifest,
    )

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match=(
            "chunking configuration"
        ),
    ):
        (
            artifacts
            .load_docling_artifact_bundle(
                root
            )
        )


def test_missing_ocr_artifact_is_rejected(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    root, _, _ = (
        _build_fake_bundle(
            tmp_path,
            monkeypatch,
        )
    )

    missing = (
        root
        / artifacts.OCR_FOLDER
        / artifacts.OCR_ASSET_PATHS[
            "det"
        ]
    )

    missing.unlink()

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match=(
            "Required RapidOCR artifact"
        ),
    ):
        (
            artifacts
            .load_docling_artifact_bundle(
                root
            )
        )


def test_changed_ocr_bytes_are_rejected(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    root, _, _ = (
        _build_fake_bundle(
            tmp_path,
            monkeypatch,
        )
    )

    recognizer = (
        root
        / artifacts.OCR_FOLDER
        / artifacts.OCR_ASSET_PATHS[
            "rec"
        ]
    )

    recognizer.write_bytes(
        b"tampered-recognizer"
    )

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match=(
            "RapidOCR artifact "
            "integrity verification failed"
        ),
    ):
        (
            artifacts
            .load_docling_artifact_bundle(
                root
            )
        )


def test_changed_bundle_content_is_rejected(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    root, _, _ = (
        _build_fake_bundle(
            tmp_path,
            monkeypatch,
        )
    )

    layout_file = (
        root
        / artifacts.LAYOUT_FOLDER
        / "layout.bin"
    )

    layout_file.write_bytes(
        b"tampered-layout"
    )

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match=(
            "bundle integrity "
            "verification failed"
        ),
    ):
        (
            artifacts
            .load_docling_artifact_bundle(
                root
            )
        )


def test_unapproved_bundle_revision_is_rejected(
    tmp_path,
    monkeypatch,
):
    _stub_runtime_versions(
        monkeypatch
    )

    root, _, _ = (
        _build_fake_bundle(
            tmp_path,
            monkeypatch,
        )
    )

    monkeypatch.setattr(
        artifacts,
        "EXPECTED_BUNDLE_REVISION",
        "0" * 64,
    )

    with pytest.raises(
        artifacts.DoclingArtifactBundleError,
        match=(
            "accepted P0 revision"
        ),
    ):
        (
            artifacts
            .load_docling_artifact_bundle(
                root
            )
        )
