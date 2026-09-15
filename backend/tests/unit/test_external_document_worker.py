from __future__ import annotations

import sys
from pathlib import Path

from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)
from app.infrastructure.parsing.parser_worker import (
    run_external_document_worker,
)


SUCCESS_WORKER = r"""
import argparse
import json


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
    required=True,
)
parser.add_argument(
    "--max-chunk-chars",
    type=int,
    required=True,
)

args = parser.parse_args()

if input() != "GO":
    raise SystemExit(9)

payload = {
    "status": "SUCCESS",
    "failure_class": None,
    "worker_setup_ms": 1.25,
    "parser_execution_ms": 2.5,
    "parsed_document": {
        "pages": [
            {
                "page_number": 1,
                "text": "hello",
                "metadata": {
                    "parser": "test",
                },
            }
        ],
        "chunks": [
            {
                "page_number": 1,
                "chunk_index": 0,
                "content": "hello",
                "section_title": None,
                "language": args.language,
                "metadata": {
                    "parser": "test",
                },
            }
        ],
    },
}

args.result_path.write_text(
    json.dumps(payload),
    encoding="utf-8",
)
"""


TIMEOUT_WORKER = r"""
import argparse
import time
from pathlib import Path


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
    required=True,
)
parser.add_argument(
    "--max-chunk-chars",
    type=int,
    required=True,
)

parser.parse_args()

if input() != "GO":
    raise SystemExit(9)

time.sleep(5.0)
"""


def write_worker(
    path: Path,
    source: str,
) -> None:
    path.write_text(
        "from pathlib import Path\n"
        + source,
        encoding="utf-8",
    )


def test_external_worker_success_deserializes_document(
    tmp_path: Path,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"%PDF-test")

    worker = tmp_path / "success_worker.py"

    write_worker(
        worker,
        SUCCESS_WORKER,
    )

    result = run_external_document_worker(
        python_executable=Path(
            sys.executable
        ),
        worker_script=worker,
        file_path=source,
        language="en",
        timeout_seconds=5.0,
        max_chunk_chars=1500,
    )

    assert (
        result.parsing_status
        is ParsingStatus.SUCCESS
    )

    assert result.failure_class is None
    assert result.return_code == 0

    assert (
        result.worker_setup_ms
        == 1.25
    )

    assert (
        result.parser_execution_ms
        == 2.5
    )

    assert result.parsed_document is not None

    parsed = result.parsed_document

    assert len(parsed.pages) == 1
    assert len(parsed.chunks) == 1

    assert parsed.pages[0].page_number == 1
    assert parsed.pages[0].text == "hello"

    assert (
        parsed.chunks[0].content
        == "hello"
    )

    assert (
        parsed.chunks[0].language
        == "en"
    )


def test_external_worker_timeout_is_bounded(
    tmp_path: Path,
) -> None:
    source = tmp_path / "document.pdf"
    source.write_bytes(b"%PDF-test")

    worker = tmp_path / "timeout_worker.py"

    write_worker(
        worker,
        TIMEOUT_WORKER,
    )

    result = run_external_document_worker(
        python_executable=Path(
            sys.executable
        ),
        worker_script=worker,
        file_path=source,
        language="en",
        timeout_seconds=0.5,
        max_chunk_chars=1500,
    )

    assert (
        result.parsing_status
        is ParsingStatus.FAILED
    )

    assert (
        result.failure_class
        is ParserFailureClass.TIMEOUT
    )

    assert result.parsed_document is None

    assert (
        result.end_to_end_document_processing_ms
        < 4000.0
    )