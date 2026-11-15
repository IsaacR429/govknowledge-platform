from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from app.infrastructure.parsing.table_evidence_transport import (
    serialize_table_evidence,
)


def _write_result(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            payload,
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


def _run_sleep(
    *,
    sleep_seconds: float,
) -> None:
    time.sleep(sleep_seconds)


def _run_memory_pressure(
    *,
    allocation_bytes: int,
) -> None:
    if allocation_bytes < 1:
        raise ValueError(
            "allocation_bytes must be positive."
        )

    blocks: list[bytearray] = []
    block_size = 8 * 1024 * 1024
    allocated = 0

    while allocated < allocation_bytes:
        remaining = allocation_bytes - allocated

        current = min(
            block_size,
            remaining,
        )

        block = bytearray(current)

        for offset in range(
            0,
            len(block),
            4096,
        ):
            block[offset] = 1

        blocks.append(block)
        allocated += current


def _serialize_parsed_document(
    parsed: object,
) -> dict[str, object]:
    pages = []

    for page in parsed.pages:
        pages.append(
            {
                "page_number": page.page_number,
                "text": page.text,
                "metadata": page.metadata,
            }
        )

    chunks = []

    for chunk in parsed.chunks:
        chunks.append(
            {
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "section_title": chunk.section_title,
                "language": chunk.language,
                "metadata": chunk.metadata,
            }
        )

    result: dict[str, object] = {
        "pages": pages,
        "chunks": chunks,
    }
    result.update(
        serialize_table_evidence(
            parsed.table_evidence
        )
    )
    return result


def _run_parse(
    *,
    file_path: Path,
    language: str,
    artifacts_path: Path | None,
) -> tuple[
    dict[str, object],
    float,
    float,
]:
    # These must be set before Docling and its model
    # dependencies are imported.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    setup_started = time.perf_counter()

    from app.infrastructure.parsing.docling_parser import (
        DoclingDocumentParser,
    )

    parser = DoclingDocumentParser(
        artifacts_path=artifacts_path,
    )

    worker_setup_ms = (
        time.perf_counter()
        - setup_started
    ) * 1000.0

    execution_started = time.perf_counter()

    parsed = parser.parse(
        file_path,
        language=language,
    )

    parser_execution_ms = (
        time.perf_counter()
        - execution_started
    ) * 1000.0

    return (
        _serialize_parsed_document(parsed),
        worker_setup_ms,
        parser_execution_ms,
    )


def _classify_failure(
    exc: BaseException,
) -> str:
    """Map known parser failures to the frozen failure taxonomy."""

    exception_type = type(exc).__name__
    message = str(exc).lower()

    if (
        exception_type == "ConversionError"
        and "could not find the page-dimensions"
        in message
    ):
        return "UNRESOLVED_PAGE_GEOMETRY"

    return "UNKNOWN"


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "sleep",
            "memory",
            "parse",
        ),
        required=True,
    )

    parser.add_argument(
        "--result-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--allocation-bytes",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--file-path",
        type=Path,
    )

    parser.add_argument(
        "--language",
        type=str,
    )

    parser.add_argument(
        "--artifacts-path",
        type=Path,
    )

    args = parser.parse_args()

    _wait_for_parent_signal()

    started = time.perf_counter()

    worker_setup_ms: float | None = None
    parser_execution_ms: float | None = None

    try:
        if args.mode == "sleep":
            _run_sleep(
                sleep_seconds=args.sleep_seconds,
            )

        elif args.mode == "memory":
            _run_memory_pressure(
                allocation_bytes=(
                    args.allocation_bytes
                ),
            )

        elif args.mode == "parse":
            if args.file_path is None:
                raise ValueError(
                    "--file-path is required "
                    "for parse mode."
                )

            if not args.language:
                raise ValueError(
                    "--language is required "
                    "for parse mode."
                )

            (
                serialized,
                worker_setup_ms,
                parser_execution_ms,
            ) = _run_parse(
                file_path=args.file_path,
                language=args.language,
                artifacts_path=args.artifacts_path,
            )

            _write_result(
                args.result_path,
                {
                    "status": "SUCCESS",
                    "failure_class": None,
                    "worker_setup_ms": (
                        worker_setup_ms
                    ),
                    "parser_execution_ms": (
                        parser_execution_ms
                    ),
                    "parsed_document": serialized,
                },
            )

            return 0

        else:
            raise AssertionError(
                f"Unsupported worker mode: "
                f"{args.mode}"
            )

    except MemoryError as exc:
        _write_result(
            args.result_path,
            {
                "status": "FAILED",
                "failure_class": "OOM",
                "worker_setup_ms": (
                    worker_setup_ms
                ),
                "parser_execution_ms": (
                    parser_execution_ms
                ),
                "execution_ms": (
                    time.perf_counter()
                    - started
                ) * 1000.0,
                "exception_type": (
                    type(exc).__name__
                ),
                "exception_message": (
                    str(exc)[:2000]
                ),
            },
        )

        return 42

    except BaseException as exc:
        _write_result(
            args.result_path,
            {
                "status": "FAILED",
                "failure_class": _classify_failure(
                    exc
                ),
                "worker_setup_ms": (
                    worker_setup_ms
                ),
                "parser_execution_ms": (
                    parser_execution_ms
                ),
                "execution_ms": (
                    time.perf_counter()
                    - started
                ) * 1000.0,
                "exception_type": (
                    type(exc).__name__
                ),
                "exception_message": (
                    str(exc)[:2000]
                ),
            },
        )

        return 1

    elapsed_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    _write_result(
        args.result_path,
        {
            "status": "SUCCESS",
            "failure_class": None,
            "execution_ms": elapsed_ms,
        },
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
