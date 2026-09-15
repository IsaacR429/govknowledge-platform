from __future__ import annotations

from pathlib import Path

import pytest

from app.infrastructure.parsing import (
    modern_docling_worker_entry as worker,
)


def test_split_page_is_page_local_and_bounded() -> None:
    text = (
        ("A" * 700)
        + "\n\n"
        + ("B" * 700)
        + "\n\n"
        + ("C" * 700)
    )

    chunks = worker._split_page(
        text,
        max_chunk_chars=1500,
    )

    assert len(chunks) == 2

    assert all(
        0 < len(chunk) <= 1500
        for chunk in chunks
    )


def test_split_page_prefers_sentence_boundary() -> None:
    first = "A" * 800
    second = "B" * 800

    text = (
        first
        + ". "
        + second
    )

    chunks = worker._split_page(
        text,
        max_chunk_chars=1500,
    )

    assert len(chunks) == 2
    assert chunks[0].endswith(".")


def test_artifact_root_environment_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "bundle"
    root.mkdir()

    monkeypatch.setenv(
        worker.ARTIFACT_ROOT_ENV,
        str(root),
    )

    assert (
        worker._resolve_artifact_root()
        == root.resolve()
    )


def test_resolve_ocr_models_requires_all_models(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"

    model_root = (
        root
        / "rapidocr"
        / "ppocr-v6"
    )

    model_root.mkdir(
        parents=True
    )

    (
        model_root
        / "PP-OCRv6_det_small.onnx"
    ).write_bytes(
        b"det"
    )

    (
        model_root
        / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    ).write_bytes(
        b"cls"
    )

    with pytest.raises(
        FileNotFoundError,
        match="recognizer|Required OCR model missing",
    ):
        worker._resolve_ocr_models(
            root
        )


def test_resolve_ocr_models_returns_expected_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"

    model_root = (
        root
        / "rapidocr"
        / "ppocr-v6"
    )

    model_root.mkdir(
        parents=True
    )

    expected = (
        model_root
        / "PP-OCRv6_det_small.onnx",
        model_root
        / "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
        model_root
        / "PP-OCRv6_rec_small.onnx",
    )

    for path in expected:
        path.write_bytes(
            path.name.encode(
                "utf-8"
            )
        )

    assert (
        worker._resolve_ocr_models(
            root
        )
        == tuple(
            path.resolve()
            for path in expected
        )
    )