from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium

from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)
from app.infrastructure.parsing.parser_worker import (
    run_control_worker,
)
from app.infrastructure.parsing.resilient_parser_worker import (
    ResilientDocumentWorkerResult,
    run_resilient_document_worker,
)


SUITE_ID = "PARSER-RESILIENCE-REGRESSION-01"

EXPECTED_PHYSICAL_COUNT = 24
EXPECTED_UNIQUE_COUNT = 6
EXPECTED_FIXTURE_COUNT = 7
EXPECTED_LOGICAL_PDF_INPUTS = 31

ARTIFACTS_PATH = Path(
    r"C:\Users\isaac\.govknowledge\artifacts\docling-p0"
)

ROOT = Path(
    r"C:\Users\isaac\govknowledge-parser-resilience"
)

BASE = (
    ROOT
    / "documents"
    / "qualification"
    / "parser-resilience"
)

CORPUS_MANIFEST = (
    BASE
    / "govknowledge-parser-corpus-2026-09-14.json"
)

SPEC_PATH = (
    BASE
    / "adversarial-fixtures-spec-2026-09-14.json"
)

FIXTURE_MANIFEST = (
    BASE
    / "adversarial-fixtures-manifest-2026-09-14.json"
)

FIXTURE_DIR = BASE / "fixtures"

RESULT_DIR = BASE / "results"

PARTIAL_PATH = (
    RESULT_DIR
    / "PARSER-RESILIENCE-REGRESSION-01.partial.json"
)

FINAL_PATH = (
    RESULT_DIR
    / "PARSER-RESILIENCE-REGRESSION-01.json"
)


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def save_json(
    path: Path,
    payload: Any,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def page_count(path: Path) -> int:
    pdf = pdfium.PdfDocument(path)

    try:
        return len(pdf)
    finally:
        pdf.close()


def enum_value(value: Any) -> Any:
    if hasattr(value, "value"):
        return value.value

    return value


def result_to_dict(
    result: ResilientDocumentWorkerResult,
) -> dict[str, Any]:
    worker = result.result

    parsed_pages = None
    parsed_chunks = None

    if worker.parsed_document is not None:
        parsed_pages = len(
            worker.parsed_document.pages
        )
        parsed_chunks = len(
            worker.parsed_document.chunks
        )

    return {
        "final_status": (
            worker.parsing_status.value
        ),
        "final_failure_class": (
            worker.failure_class.value
            if worker.failure_class
            else None
        ),
        "initial_failure_class": (
            result.initial_failure_class.value
            if result.initial_failure_class
            else None
        ),
        "normalization_attempted": (
            result.normalization_attempted
        ),
        "normalization_attempt_count": (
            result.normalization_attempt_count
        ),
        "original_sha256": (
            result.original_sha256
        ),
        "derived_sha256": (
            result.derived_sha256
        ),
        "source_page_count": (
            result.source_page_count
        ),
        "derived_page_count": (
            result.derived_page_count
        ),
        "retry_status": (
            result.retry_status.value
            if result.retry_status
            else None
        ),
        "parsed_page_count": parsed_pages,
        "parsed_chunk_count": parsed_chunks,
        "return_code": worker.return_code,
        "exception_type": (
            worker.exception_type
        ),
        "exception_message": (
            worker.exception_message
        ),
        "worker_spawn_ms": (
            worker.worker_spawn_ms
        ),
        "worker_setup_ms": (
            worker.worker_setup_ms
        ),
        "parser_execution_ms": (
            worker.parser_execution_ms
        ),
        "worker_teardown_ms": (
            worker.worker_teardown_ms
        ),
        "worker_peak_rss_bytes": (
            worker.worker_peak_rss_bytes
        ),
        "worker_end_to_end_ms": (
            worker
            .end_to_end_document_processing_ms
        ),
        "resilient_end_to_end_ms": (
            result
            .end_to_end_document_processing_ms
        ),
    }


def choose_language(
    path: Path,
) -> str:
    name = path.name.lower()

    if (
        "-fr" in name
        or "_fr" in name
    ):
        return "fr"

    return "en"


def find_fixture_hashes(
    manifest: Any,
) -> dict[str, str]:
    """Extract filename -> sha256 without assuming manifest layout."""

    found: dict[str, str] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            filename = value.get(
                "filename"
            )
            sha = value.get("sha256")

            if (
                isinstance(filename, str)
                and isinstance(sha, str)
                and filename.lower().endswith(
                    ".pdf"
                )
            ):
                found[
                    Path(filename).name
                ] = sha.lower()

            for child in value.values():
                walk(child)

        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(manifest)

    return found


def run_w01() -> dict[str, Any]:
    result = run_control_worker(
        mode="sleep",
        timeout_seconds=0.75,
        sleep_seconds=5.0,
    )

    passed = (
        result.parsing_status
        is ParsingStatus.FAILED
        and result.failure_class
        is ParserFailureClass.TIMEOUT
        and (
            result
            .end_to_end_document_processing_ms
            < 4000.0
        )
    )

    return {
        "control_id": "W01",
        "passed": passed,
        "status": (
            result.parsing_status.value
        ),
        "failure_class": (
            result.failure_class.value
            if result.failure_class
            else None
        ),
        "end_to_end_ms": (
            result
            .end_to_end_document_processing_ms
        ),
    }


def run_w02() -> dict[str, Any]:
    name = (
        "govknowledge-regression-w02-"
        + str(int(time.time()))
    )

    memory = 64 * 1024 * 1024
    requested = 512 * 1024 * 1024

    code = """
import sys
size = 512 * 1024 * 1024
print("REQUESTED", size, flush=True)
data = bytearray(size)
for offset in range(0, size, 4096):
    data[offset] = 1
print("ALLOCATED", len(data), flush=True)
"""

    command = [
        "docker",
        "run",
        "--name",
        name,
        "--memory=64m",
        "--memory-swap=64m",
        "--network=none",
        "-i",
        "python:3.11-slim",
        "python",
        "-",
    ]

    try:
        run = subprocess.run(
            command,
            input=code,
            text=True,
            capture_output=True,
            timeout=30,
        )

        inspect_run = subprocess.run(
            [
                "docker",
                "inspect",
                name,
            ],
            text=True,
            capture_output=True,
            timeout=15,
            check=True,
        )

        inspected = json.loads(
            inspect_run.stdout
        )[0]

        state = inspected["State"]
        host = inspected["HostConfig"]

        logs_run = subprocess.run(
            [
                "docker",
                "logs",
                name,
            ],
            text=True,
            capture_output=True,
            timeout=15,
        )

        logs = (
            logs_run.stdout
            + logs_run.stderr
        ).strip()

        passed = (
            int(host["Memory"]) == memory
            and int(
                host["MemorySwap"]
            ) == memory
            and bool(
                state["OOMKilled"]
            )
            and int(
                state["ExitCode"]
            ) != 0
            and (
                f"ALLOCATED {requested}"
                not in logs
            )
        )

        return {
            "control_id": "W02",
            "passed": passed,
            "configured_memory_bytes": (
                int(host["Memory"])
            ),
            "configured_memory_swap_bytes": (
                int(host["MemorySwap"])
            ),
            "requested_bytes": requested,
            "oom_killed": bool(
                state["OOMKilled"]
            ),
            "container_exit_code": int(
                state["ExitCode"]
            ),
            "docker_run_exit_code": (
                run.returncode
            ),
            "logs": logs,
        }

    finally:
        subprocess.run(
            [
                "docker",
                "rm",
                "-f",
                name,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def parse_one(
    *,
    path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    expected_pages = page_count(path)

    started = time.perf_counter()

    try:
        result = (
            run_resilient_document_worker(
                file_path=path,
                language=choose_language(
                    path
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
                artifacts_path=(
                    ARTIFACTS_PATH
                ),
            )
        )

        record = result_to_dict(
            result
        )

        record["harness_exception"] = None

    except BaseException as exc:
        record = {
            "final_status": "FAILED",
            "final_failure_class": (
                "HARNESS_UNHANDLED_EXCEPTION"
            ),
            "initial_failure_class": None,
            "normalization_attempted": False,
            "normalization_attempt_count": 0,
            "original_sha256": (
                sha256_file(path)
            ),
            "derived_sha256": None,
            "source_page_count": (
                expected_pages
            ),
            "derived_page_count": None,
            "retry_status": None,
            "parsed_page_count": None,
            "parsed_chunk_count": None,
            "return_code": None,
            "exception_type": (
                type(exc).__name__
            ),
            "exception_message": (
                str(exc)[:2000]
            ),
            "harness_exception": True,
        }

    record["independent_page_count"] = (
        expected_pages
    )

    record["wall_clock_ms"] = (
        time.perf_counter()
        - started
    ) * 1000.0

    parsed_count = record.get(
        "parsed_page_count"
    )

    record["silent_page_count_change"] = (
        parsed_count is not None
        and parsed_count
        != expected_pages
    )

    return record


def fixture_acceptance(
    *,
    fixture: dict[str, Any],
    result: dict[str, Any],
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    fixture_id = fixture["fixture_id"]

    status = result.get(
        "final_status"
    )

    failure = result.get(
        "final_failure_class"
    )

    attempts = int(
        result.get(
            "normalization_attempt_count",
            0,
        )
        or 0
    )

    expected_pages = int(
        fixture["expected_page_count"]
    )

    parsed_pages = result.get(
        "parsed_page_count"
    )

    if result.get("harness_exception"):
        failures.append(
            "UNHANDLED_EXCEPTION"
        )

    if result.get(
        "silent_page_count_change"
    ):
        failures.append(
            "SILENT_PAGE_COUNT_CHANGE"
        )

    if attempts > 1:
        failures.append(
            "RECURSIVE_NORMALIZATION_RETRY"
        )

    if (
        not fixture[
            "normalization_allowed"
        ]
        and attempts != 0
    ):
        failures.append(
            "NORMALIZATION_NOT_ALLOWED"
        )

    required = fixture[
        "required_final_behavior"
    ]

    if required == "SUCCESS_DIRECT":
        if status != "SUCCESS":
            failures.append(
                "EXPECTED_SUCCESS_DIRECT"
            )

        if attempts != 0:
            failures.append(
                "EXPECTED_ZERO_NORMALIZATION"
            )

    elif required in {
        "SUCCESS_AFTER_AT_MOST_ONE_NORMALIZATION_RETRY",
        "SUCCESS_DIRECT_OR_AFTER_AT_MOST_ONE_NORMALIZATION_RETRY",
    }:
        if status != "SUCCESS":
            failures.append(
                "EXPECTED_SUCCESS"
            )

    elif required == (
        "SUCCESS_AFTER_AT_MOST_ONE_NORMALIZATION_RETRY_OR_CLEAN_FAILED"
    ):
        if status not in {
            "SUCCESS",
            "FAILED",
        }:
            failures.append(
                "INVALID_TERMINAL_STATUS"
            )

    else:
        failures.append(
            "UNKNOWN_REQUIRED_BEHAVIOR"
        )

    if (
        status == "SUCCESS"
        and parsed_pages
        != expected_pages
    ):
        failures.append(
            "EXPECTED_PAGE_COUNT_MISMATCH"
        )

    if status not in {
        "SUCCESS",
        "FAILED",
    }:
        failures.append(
            "INVALID_TERMINAL_STATUS"
        )

    if (
        status == "FAILED"
        and failure is None
    ):
        failures.append(
            "FAILED_WITHOUT_FAILURE_CLASS"
        )

    return (
        len(failures) == 0,
        failures,
    )


def main() -> int:
    argument_parser = (
        argparse.ArgumentParser()
    )

    argument_parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=600.0,
    )

    argument_parser.add_argument(
        "--reset",
        action="store_true",
    )

    args = argument_parser.parse_args()

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.reset:
        PARTIAL_PATH.unlink(
            missing_ok=True
        )

    corpus = load_json(
        CORPUS_MANIFEST
    )

    spec = load_json(
        SPEC_PATH
    )

    fixture_manifest = load_json(
        FIXTURE_MANIFEST
    )

    fixture_hashes = (
        find_fixture_hashes(
            fixture_manifest
        )
    )

    assert (
        corpus["physical_pdf_count"]
        == EXPECTED_PHYSICAL_COUNT
    )

    assert (
        corpus["unique_sha256_count"]
        == EXPECTED_UNIQUE_COUNT
    )

    assert (
        len(spec["fixtures"])
        == EXPECTED_FIXTURE_COUNT
    )

    report: dict[str, Any]

    if PARTIAL_PATH.is_file():
        report = load_json(
            PARTIAL_PATH
        )

        print(
            "RESUMING:",
            PARTIAL_PATH,
        )

    else:
        report = {
            "suite_id": SUITE_ID,
            "started_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "status": "RUNNING",
            "physical_inputs": {},
            "unique_corpus_executions": {},
            "fixture_executions": {},
            "worker_controls": {},
        }

    source_root = Path(
        corpus["source_root"]
    )

    print(
        "===== PHYSICAL CORPUS ACCOUNTING ====="
    )

    grouped: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for physical in corpus[
        "physical_files"
    ]:
        relative = physical[
            "relative_path"
        ]

        expected_hash = physical[
            "sha256"
        ].lower()

        path = (
            source_root
            / Path(relative)
        )

        if not path.is_file():
            raise RuntimeError(
                f"Missing physical corpus path: "
                f"{path}"
            )

        actual_hash = sha256_file(
            path
        )

        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA mismatch for {path}: "
                f"{actual_hash} != "
                f"{expected_hash}"
            )

        grouped.setdefault(
            expected_hash,
            [],
        ).append(
            {
                "relative_path": relative,
                "path": str(path),
                "sha256": expected_hash,
            }
        )

    if len(grouped) != EXPECTED_UNIQUE_COUNT:
        raise RuntimeError(
            "Unique SHA count changed."
        )

    print(
        "physical paths:",
        sum(
            len(items)
            for items in grouped.values()
        ),
    )
    print(
        "unique identities:",
        len(grouped),
    )
    print(
        "PHYSICAL ACCOUNTING: PASS"
    )

    print()
    print(
        "===== UNIQUE CORPUS EXECUTIONS ====="
    )

    for index, (
        sha,
        physicals,
    ) in enumerate(
        sorted(grouped.items()),
        start=1,
    ):
        if sha in report[
            "unique_corpus_executions"
        ]:
            print(
                f"[{index}/"
                f"{len(grouped)}] "
                f"{sha[:12]} RESUME-SKIP"
            )
            continue

        representative = Path(
            physicals[0]["path"]
        )

        print(
            f"[{index}/"
            f"{len(grouped)}] "
            f"{representative.name}"
        )

        result = parse_one(
            path=representative,
            timeout_seconds=(
                args.timeout_seconds
            ),
        )

        result[
            "representative_path"
        ] = str(representative)

        result["sha256"] = sha

        report[
            "unique_corpus_executions"
        ][sha] = result

        save_json(
            PARTIAL_PATH,
            report,
        )

        print(
            "  final:",
            result["final_status"],
            result[
                "final_failure_class"
            ],
        )
        print(
            "  pages:",
            result[
                "parsed_page_count"
            ],
            "/",
            result[
                "independent_page_count"
            ],
        )

    # Expand unique execution lineage to
    # all 24 physical paths.
    for sha, physicals in grouped.items():
        execution = report[
            "unique_corpus_executions"
        ][sha]

        for physical in physicals:
            key = physical[
                "relative_path"
            ]

            report[
                "physical_inputs"
            ][key] = {
                "path": physical["path"],
                "sha256": sha,
                "execution_mode": (
                    "UNIQUE_SHA256_REPRESENTATIVE"
                ),
                "representative_path": (
                    execution[
                        "representative_path"
                    ]
                ),
                "result": execution,
            }

    save_json(
        PARTIAL_PATH,
        report,
    )

    print()
    print(
        "===== ADVERSARIAL FIXTURES ====="
    )

    fixture_specs = {
        item["fixture_id"]: item
        for item in spec["fixtures"]
    }

    for index, fixture in enumerate(
        spec["fixtures"],
        start=1,
    ):
        fixture_id = fixture[
            "fixture_id"
        ]

        if fixture_id in report[
            "fixture_executions"
        ]:
            print(
                f"[{index}/7] "
                f"{fixture_id} RESUME-SKIP"
            )
            continue

        filename = fixture["filename"]

        path = FIXTURE_DIR / filename

        if not path.is_file():
            raise RuntimeError(
                f"Missing fixture: {path}"
            )

        actual_hash = sha256_file(
            path
        )

        expected_fixture_hash = (
            fixture_hashes.get(filename)
        )

        if expected_fixture_hash is None:
            raise RuntimeError(
                "Fixture manifest did not "
                f"provide SHA for {filename}."
            )

        if (
            actual_hash
            != expected_fixture_hash
        ):
            raise RuntimeError(
                f"Fixture SHA mismatch "
                f"for {fixture_id}: "
                f"{actual_hash} != "
                f"{expected_fixture_hash}"
            )

        print(
            f"[{index}/7] "
            f"{fixture_id} "
            f"{filename}"
        )

        result = parse_one(
            path=path,
            timeout_seconds=(
                args.timeout_seconds
            ),
        )

        passed, failures = (
            fixture_acceptance(
                fixture=fixture,
                result=result,
            )
        )

        result[
            "fixture_id"
        ] = fixture_id
        result[
            "fixture_sha256"
        ] = actual_hash
        result[
            "expected_page_count"
        ] = fixture[
            "expected_page_count"
        ]
        result[
            "normalization_allowed"
        ] = fixture[
            "normalization_allowed"
        ]
        result[
            "required_final_behavior"
        ] = fixture[
            "required_final_behavior"
        ]
        result[
            "acceptance_passed"
        ] = passed
        result[
            "acceptance_failures"
        ] = failures

        report[
            "fixture_executions"
        ][fixture_id] = result

        save_json(
            PARTIAL_PATH,
            report,
        )

        print(
            "  final:",
            result["final_status"],
            result[
                "final_failure_class"
            ],
        )
        print(
            "  normalize attempts:",
            result[
                "normalization_attempt_count"
            ],
        )
        print(
            "  acceptance:",
            "PASS"
            if passed
            else "FAIL",
            failures,
        )

    print()
    print(
        "===== WORKER CONTROLS ====="
    )

    w01 = run_w01()

    report[
        "worker_controls"
    ]["W01"] = w01

    print(
        "W01:",
        "PASS"
        if w01["passed"]
        else "FAIL",
    )

    w02 = run_w02()

    report[
        "worker_controls"
    ]["W02"] = w02

    print(
        "W02:",
        "PASS"
        if w02["passed"]
        else "FAIL",
    )

    save_json(
        PARTIAL_PATH,
        report,
    )

    print()
    print(
        "===== SUITE ACCEPTANCE ====="
    )

    physical_count = len(
        report["physical_inputs"]
    )

    unique_count = len(
        report[
            "unique_corpus_executions"
        ]
    )

    fixture_count = len(
        report[
            "fixture_executions"
        ]
    )

    fixture_failures = {
        fixture_id: result[
            "acceptance_failures"
        ]
        for fixture_id, result
        in report[
            "fixture_executions"
        ].items()
        if not result[
            "acceptance_passed"
        ]
    }

    corpus_bad_status = {
        sha: result[
            "final_status"
        ]
        for sha, result
        in report[
            "unique_corpus_executions"
        ].items()
        if result[
            "final_status"
        ] not in {
            "SUCCESS",
            "FAILED",
        }
    }

    corpus_unhandled = {
        sha: result.get(
            "exception_message"
        )
        for sha, result
        in report[
            "unique_corpus_executions"
        ].items()
        if result.get(
            "harness_exception"
        )
    }

    corpus_silent_loss = {
        sha: result.get(
            "parsed_page_count"
        )
        for sha, result
        in report[
            "unique_corpus_executions"
        ].items()
        if result.get(
            "silent_page_count_change"
        )
    }

    recursive_retries = {
        key: result[
            "normalization_attempt_count"
        ]
        for key, result
        in {
            **report[
                "unique_corpus_executions"
            ],
            **report[
                "fixture_executions"
            ],
        }.items()
        if (
            result.get(
                "normalization_attempt_count",
                0,
            )
            > 1
        )
    }

    controls_pass = (
        report[
            "worker_controls"
        ]["W01"]["passed"]
        and report[
            "worker_controls"
        ]["W02"]["passed"]
    )

    checks = {
        "physical_pdf_paths_accounted_for": (
            physical_count
            == EXPECTED_PHYSICAL_COUNT
        ),
        "unique_pdf_identities_accounted_for": (
            unique_count
            == EXPECTED_UNIQUE_COUNT
        ),
        "adversarial_fixture_cases_accounted_for": (
            fixture_count
            == EXPECTED_FIXTURE_COUNT
        ),
        "logical_pdf_inputs_accounted_for": (
            physical_count
            + fixture_count
            == EXPECTED_LOGICAL_PDF_INPUTS
        ),
        "fixture_acceptance_all_pass": (
            not fixture_failures
        ),
        "no_invalid_corpus_terminal_status": (
            not corpus_bad_status
        ),
        "no_unhandled_exceptions": (
            not corpus_unhandled
        ),
        "no_silent_page_count_changes": (
            not corpus_silent_loss
        ),
        "no_recursive_normalization_retries": (
            not recursive_retries
        ),
        "worker_controls_pass": (
            controls_pass
        ),
    }

    overall_pass = all(
        checks.values()
    )

    report["acceptance"] = {
        "checks": checks,
        "fixture_failures": (
            fixture_failures
        ),
        "corpus_bad_status": (
            corpus_bad_status
        ),
        "corpus_unhandled": (
            corpus_unhandled
        ),
        "corpus_silent_page_changes": (
            corpus_silent_loss
        ),
        "recursive_retries": (
            recursive_retries
        ),
        "physical_pdf_count": (
            physical_count
        ),
        "unique_sha256_count": (
            unique_count
        ),
        "fixture_count": (
            fixture_count
        ),
        "logical_pdf_input_count": (
            physical_count
            + fixture_count
        ),
    }

    report["status"] = (
        "PASS"
        if overall_pass
        else "FAIL"
    )

    report["completed_at_utc"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    save_json(
        FINAL_PATH,
        report,
    )

    print(
        "physical paths:",
        physical_count,
        "/",
        EXPECTED_PHYSICAL_COUNT,
    )
    print(
        "unique identities:",
        unique_count,
        "/",
        EXPECTED_UNIQUE_COUNT,
    )
    print(
        "fixtures:",
        fixture_count,
        "/",
        EXPECTED_FIXTURE_COUNT,
    )
    print(
        "logical PDF inputs:",
        physical_count
        + fixture_count,
        "/",
        EXPECTED_LOGICAL_PDF_INPUTS,
    )

    for key, passed in checks.items():
        print(
            f"{key}:",
            "PASS"
            if passed
            else "FAIL",
        )

    if fixture_failures:
        print(
            "fixture_failures:",
            fixture_failures,
        )

    if corpus_unhandled:
        print(
            "corpus_unhandled:",
            corpus_unhandled,
        )

    print()
    print(
        f"{SUITE_ID}:",
        report["status"],
    )
    print(
        "REPORT:",
        FINAL_PATH,
    )

    # Keep the partial checkpoint for
    # reproducibility/resume evidence.
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
