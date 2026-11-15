from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from app.infrastructure.parsing import (
    pymupdf_worker_entry as worker,
)


class FakePage:
    def __init__(
        self,
        *,
        native_text: str,
        image_count: int,
    ) -> None:
        self._native_text = native_text
        self._image_count = image_count

    def get_text(
        self,
        mode: str,
    ) -> str:
        assert mode == "text"
        return self._native_text

    def get_images(
        self,
        *,
        full: bool,
    ) -> list[tuple[int]]:
        assert full is True

        return [
            (index,)
            for index in range(
                self._image_count
            )
        ]

    def find_tables(self):
        return types.SimpleNamespace(
            tables=[]
        )


class FakePdf:
    def __init__(
        self,
        pages: list[FakePage],
    ) -> None:
        self._pages = pages
        self.page_count = len(pages)

    def __enter__(
        self,
    ) -> "FakePdf":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        return None

    def load_page(
        self,
        index: int,
    ) -> FakePage:
        return self._pages[index]


def install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    markdown_pages: list[str],
    native_pages: list[str],
    image_counts: list[int],
) -> None:
    fake_llm = types.ModuleType(
        "pymupdf4llm"
    )

    def to_markdown(
        path: str,
        *,
        page_chunks: bool,
        use_ocr: bool,
        show_progress: bool,
    ) -> list[dict[str, object]]:
        assert path
        assert page_chunks is True
        assert use_ocr is True
        assert show_progress is False

        return [
            {
                "text": text,
                "metadata": {
                    "file_path": path,
                },
            }
            for text in markdown_pages
        ]

    fake_llm.to_markdown = to_markdown

    fake_pymupdf = types.ModuleType(
        "pymupdf"
    )

    fake_pages = [
        FakePage(
            native_text=native,
            image_count=images,
        )
        for native, images in zip(
            native_pages,
            image_counts,
            strict=True,
        )
    ]

    fake_pymupdf.open = (
        lambda path: FakePdf(
            fake_pages
        )
    )

    monkeypatch.setitem(
        sys.modules,
        "pymupdf4llm",
        fake_llm,
    )

    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        fake_pymupdf,
    )

    monkeypatch.setattr(
        worker,
        "version",
        lambda package: {
            "pymupdf4llm": "1.28.2",
            "pymupdf": "1.28.2",
        }[package],
    )


def test_split_page_empty_text_has_no_chunks() -> None:
    assert (
        worker._split_page(
            "   \n\n  ",
            max_chunk_chars=1500,
        )
        == ()
    )


def test_split_page_respects_hard_limit() -> None:
    text = (
        "first paragraph\n\n"
        + ("word " * 100)
        + "\n\nlast paragraph"
    )

    chunks = worker._split_page(
        text,
        max_chunk_chars=80,
    )

    assert chunks

    assert all(
        0 < len(chunk) <= 80
        for chunk in chunks
    )


def test_run_parse_produces_canonical_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"%PDF-test")

    install_fake_runtime(
        monkeypatch,
        markdown_pages=[
            "Page one paragraph.",
            (
                "Page two first paragraph.\n\n"
                "Page two second paragraph."
            ),
        ],
        native_pages=[
            "native one",
            "",
        ],
        image_counts=[
            0,
            1,
        ],
    )

    (
        parsed,
        runtime,
        worker_setup_ms,
        parser_execution_ms,
    ) = worker._run_parse(
        file_path=source,
        language="en",
        max_chunk_chars=1500,
    )

    assert runtime == {
        "pymupdf4llm_version": "1.28.2",
        "pymupdf_version": "1.28.2",
    }

    assert (
        parsed[
            "table_evidence_schema_version"
        ]
        == 1
    )

    assert (
        parsed[
            "table_evidence"
        ]
        == []
    )

    pages = parsed["pages"]
    chunks = parsed["chunks"]

    assert [
        page["page_number"]
        for page in pages
    ] == [1, 2]

    assert [
        chunk["chunk_index"]
        for chunk in chunks
    ] == list(
        range(len(chunks))
    )

    assert all(
        chunk["language"] == "en"
        for chunk in chunks
    )

    assert all(
        chunk["page_number"] in {1, 2}
        for chunk in chunks
    )

    assert pages[0]["metadata"][
        "native_text_chars"
    ] == len("native one")

    assert pages[1]["metadata"][
        "native_text_chars"
    ] == 0

    assert pages[1]["metadata"][
        "image_count"
    ] == 1

    serialized = json.dumps(
        parsed
    )

    assert str(
        source.resolve()
    ) not in serialized

    assert '"file_path"' not in serialized


def test_run_parse_rejects_page_count_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"%PDF-test")

    install_fake_runtime(
        monkeypatch,
        markdown_pages=[
            "Only one parser page.",
        ],
        native_pages=[
            "native one",
            "native two",
        ],
        image_counts=[
            0,
            0,
        ],
    )

    with pytest.raises(
        ValueError,
        match="page count",
    ):
        worker._run_parse(
            file_path=source,
            language="en",
            max_chunk_chars=1500,
        )