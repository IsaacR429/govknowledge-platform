from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path


def _write_result(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _wait_for_parent_signal() -> None:
    signal = input()

    if signal != "GO":
        raise RuntimeError(
            "Worker did not receive expected GO signal."
        )


def _split_page(
    text: str,
    *,
    max_chunk_chars: int,
) -> tuple[str, ...]:
    """
    Preserve the GovKnowledge P0 paragraph-greedy
    page-local chunking semantics.
    """
    text = text.strip()

    if not text:
        return ()

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    def flush() -> None:
        nonlocal current_parts
        nonlocal current_length

        if current_parts:
            chunks.append(
                "\n\n".join(
                    current_parts
                )
            )

        current_parts = []
        current_length = 0

    for paragraph in paragraphs:

        if len(paragraph) > max_chunk_chars:
            flush()

            remaining = paragraph

            while remaining:

                if len(remaining) <= max_chunk_chars:
                    piece = remaining.strip()

                    if piece:
                        chunks.append(piece)

                    break

                window = remaining[
                    :max_chunk_chars
                ]

                sentence_boundaries = (
                    window.rfind(". "),
                    window.rfind("? "),
                    window.rfind("! "),
                    window.rfind(".\n"),
                    window.rfind("?\n"),
                    window.rfind("!\n"),
                )

                boundary = max(
                    sentence_boundaries
                )

                if boundary >= (
                    max_chunk_chars // 2
                ):
                    boundary += 1

                else:
                    whitespace_boundaries = (
                        window.rfind(" "),
                        window.rfind("\n"),
                        window.rfind("\t"),
                    )

                    boundary = max(
                        whitespace_boundaries
                    )

                    if boundary < 1:
                        boundary = (
                            max_chunk_chars
                        )

                piece = remaining[
                    :boundary
                ].strip()

                if piece:
                    chunks.append(piece)

                remaining = remaining[
                    boundary:
                ].strip()

            continue

        separator_length = (
            2 if current_parts else 0
        )

        proposed_length = (
            current_length
            + separator_length
            + len(paragraph)
        )

        if (
            current_parts
            and proposed_length
            > max_chunk_chars
        ):
            flush()

        if current_parts:
            current_length += 2

        current_parts.append(
            paragraph
        )

        current_length += len(
            paragraph
        )

    flush()

    return tuple(chunks)


def _run_parse(
    *,
    file_path: Path,
    language: str,
    max_chunk_chars: int,
) -> tuple[
    dict[str, object],
    dict[str, str],
    float,
    float,
]:
    if max_chunk_chars < 1:
        raise ValueError(
            "max_chunk_chars must be positive."
        )

    source = file_path.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if not language:
        raise ValueError(
            "language cannot be empty."
        )

    setup_started = time.perf_counter()

    import pymupdf
    import pymupdf4llm

    runtime = {
        "pymupdf4llm_version": version(
            "pymupdf4llm"
        ),
        "pymupdf_version": version(
            "pymupdf"
        ),
    }

    worker_setup_ms = (
        time.perf_counter()
        - setup_started
    ) * 1000.0

    execution_started = (
        time.perf_counter()
    )

    page_chunks = (
        pymupdf4llm.to_markdown(
            str(source),
            page_chunks=True,
            use_ocr=True,
            show_progress=False,
        )
    )

    if not isinstance(
        page_chunks,
        list,
    ):
        raise TypeError(
            "PyMuPDF4LLM page_chunks output "
            "must be a list."
        )

    pages: list[
        dict[str, object]
    ] = []

    chunks: list[
        dict[str, object]
    ] = []

    chunk_index = 0

    with pymupdf.open(
        str(source)
    ) as pdf:

        if len(page_chunks) != pdf.page_count:
            raise ValueError(
                "PyMuPDF4LLM page count does not "
                "match PyMuPDF page count."
            )

        for offset, raw_page in enumerate(
            page_chunks
        ):
            if not isinstance(
                raw_page,
                dict,
            ):
                raise TypeError(
                    "PyMuPDF4LLM page chunk "
                    "must be an object."
                )

            page_number = offset + 1

            text = str(
                raw_page.get(
                    "text",
                    "",
                )
            ).strip()

            pdf_page = pdf.load_page(
                offset
            )

            native_text = (
                pdf_page.get_text(
                    "text"
                )
                or ""
            ).strip()

            image_count = len(
                pdf_page.get_images(
                    full=True
                )
            )

            metadata = {
                "parser":
                    "pymupdf4llm",

                "pymupdf4llm_version":
                    runtime[
                        "pymupdf4llm_version"
                    ],

                "pymupdf_version":
                    runtime[
                        "pymupdf_version"
                    ],

                "max_chunk_chars":
                    max_chunk_chars,

                "chunking_strategy":
                    "paragraph_greedy_page_local",

                "page_number":
                    page_number,

                "ocr_requested":
                    True,

                "native_text_chars":
                    len(native_text),

                "image_count":
                    image_count,
            }

            pages.append(
                {
                    "page_number":
                        page_number,

                    "text":
                        text,

                    "metadata":
                        metadata.copy(),
                }
            )

            for chunk_text in _split_page(
                text,
                max_chunk_chars=(
                    max_chunk_chars
                ),
            ):
                chunks.append(
                    {
                        "page_number":
                            page_number,

                        "chunk_index":
                            chunk_index,

                        "content":
                            chunk_text,

                        "section_title":
                            None,

                        "language":
                            language,

                        "metadata":
                            metadata.copy(),
                    }
                )

                chunk_index += 1

    parser_execution_ms = (
        time.perf_counter()
        - execution_started
    ) * 1000.0

    return (
        {
            "pages": pages,
            "chunks": chunks,
        },
        runtime,
        worker_setup_ms,
        parser_execution_ms,
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--file-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--language",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=1500,
    )

    args = parser.parse_args()

    _wait_for_parent_signal()

    started = time.perf_counter()

    worker_setup_ms: float | None = None
    parser_execution_ms: float | None = None

    try:
        (
            parsed_document,
            runtime,
            worker_setup_ms,
            parser_execution_ms,
        ) = _run_parse(
            file_path=args.file_path,
            language=args.language,
            max_chunk_chars=(
                args.max_chunk_chars
            ),
        )

        _write_result(
            args.result_path,
            {
                "status": "SUCCESS",
                "failure_class": None,
                "worker_setup_ms":
                    worker_setup_ms,
                "parser_execution_ms":
                    parser_execution_ms,
                "runtime":
                    runtime,
                "parsed_document":
                    parsed_document,
            },
        )

        return 0

    except MemoryError as exc:
        _write_result(
            args.result_path,
            {
                "status": "FAILED",
                "failure_class": "OOM",
                "worker_setup_ms":
                    worker_setup_ms,
                "parser_execution_ms":
                    parser_execution_ms,
                "execution_ms": (
                    time.perf_counter()
                    - started
                ) * 1000.0,
                "exception_type":
                    type(exc).__name__,
                "exception_message":
                    str(exc)[:2000],
            },
        )

        return 42

    except BaseException as exc:
        _write_result(
            args.result_path,
            {
                "status": "FAILED",
                "failure_class": "UNKNOWN",
                "worker_setup_ms":
                    worker_setup_ms,
                "parser_execution_ms":
                    parser_execution_ms,
                "execution_ms": (
                    time.perf_counter()
                    - started
                ) * 1000.0,
                "exception_type":
                    type(exc).__name__,
                "exception_message":
                    str(exc)[:2000],
            },
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )