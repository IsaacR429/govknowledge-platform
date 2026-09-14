from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)


if os.name != "nt":
    raise RuntimeError(
        "P0 parser worker isolation currently requires "
        "Windows Job Objects."
    )


JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(
    ctypes.Structure
):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(
    ctypes.Structure
):
    _fields_ = [
        (
            "BasicLimitInformation",
            JOBOBJECT_BASIC_LIMIT_INFORMATION,
        ),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        (
            "QuotaPeakPagedPoolUsage",
            ctypes.c_size_t,
        ),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        (
            "QuotaPeakNonPagedPoolUsage",
            ctypes.c_size_t,
        ),
        (
            "QuotaNonPagedPoolUsage",
            ctypes.c_size_t,
        ),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


kernel32 = ctypes.WinDLL(
    "kernel32",
    use_last_error=True,
)

psapi = ctypes.WinDLL(
    "psapi",
    use_last_error=True,
)


kernel32.CreateJobObjectW.restype = wintypes.HANDLE

kernel32.SetInformationJobObject.restype = (
    wintypes.BOOL
)

kernel32.AssignProcessToJobObject.restype = (
    wintypes.BOOL
)

kernel32.TerminateJobObject.restype = (
    wintypes.BOOL
)

kernel32.OpenProcess.restype = wintypes.HANDLE

kernel32.CloseHandle.restype = wintypes.BOOL

psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


@dataclass(frozen=True)
class WorkerRunResult:
    parsing_status: ParsingStatus
    failure_class: ParserFailureClass | None
    return_code: int | None
    worker_spawn_ms: float
    worker_setup_ms: float
    parser_execution_ms: float | None
    worker_teardown_ms: float
    end_to_end_document_processing_ms: float
    worker_peak_rss_bytes: int | None


def _raise_last_winerror(
    operation: str,
) -> None:
    error = ctypes.get_last_error()

    raise OSError(
        error,
        f"{operation} failed",
    )


def _create_job() -> wintypes.HANDLE:
    """Create a Windows process-tree isolation job.

    P0 qualification proved hard timeout / termination
    through this Job Object.

    Hard memory limiting is intentionally NOT claimed
    here. Windows Job Object memory limits were tested
    and did not enforce the configured ceiling in the
    qualification environment. Production hard-memory
    enforcement is provided by the Linux/container
    cgroup boundary.
    """

    job = kernel32.CreateJobObjectW(
        None,
        None,
    )

    if not job:
        _raise_last_winerror(
            "CreateJobObjectW"
        )

    info = (
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    )

    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )

    ok = kernel32.SetInformationJobObject(
        job,
        JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )

    if not ok:
        kernel32.CloseHandle(job)

        _raise_last_winerror(
            "SetInformationJobObject"
        )

    return job


def _assign_process(
    *,
    job: wintypes.HANDLE,
    process_handle: int,
) -> None:
    ok = kernel32.AssignProcessToJobObject(
        job,
        process_handle,
    )

    if not ok:
        _raise_last_winerror(
            "AssignProcessToJobObject"
        )


def _open_process_for_memory(
    pid: int,
) -> wintypes.HANDLE | None:
    handle = kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION
        | PROCESS_VM_READ,
        False,
        pid,
    )

    if not handle:
        return None

    return handle


def _peak_rss(
    process_handle: wintypes.HANDLE | None,
) -> int | None:
    if not process_handle:
        return None

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)

    ok = psapi.GetProcessMemoryInfo(
        process_handle,
        ctypes.byref(counters),
        ctypes.sizeof(counters),
    )

    if not ok:
        return None

    return int(
        counters.PeakWorkingSetSize
    )


def _load_worker_result(
    path: Path,
) -> dict[str, object] | None:
    if not path.is_file():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        OSError,
        json.JSONDecodeError,
    ):
        return None


@dataclass(frozen=True)
class _RawWorkerRun:
    payload: dict[str, object] | None
    return_code: int | None
    timed_out: bool
    worker_spawn_ms: float
    parent_setup_ms: float
    worker_teardown_ms: float
    end_to_end_ms: float
    worker_peak_rss_bytes: int | None


@dataclass(frozen=True)
class DocumentWorkerRunResult:
    parsing_status: ParsingStatus
    failure_class: ParserFailureClass | None
    parsed_document: object | None
    return_code: int | None
    worker_spawn_ms: float
    worker_setup_ms: float | None
    parser_execution_ms: float | None
    worker_teardown_ms: float
    end_to_end_document_processing_ms: float
    worker_peak_rss_bytes: int | None
    exception_type: str | None
    exception_message: str | None


def _execute_worker(
    *,
    command: list[str],
    timeout_seconds: float,
) -> _RawWorkerRun:
    if timeout_seconds <= 0:
        raise ValueError(
            "timeout_seconds must be positive."
        )

    total_started = time.perf_counter()

    spawn_started = time.perf_counter()

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    spawn_ms = (
        time.perf_counter()
        - spawn_started
    ) * 1000.0

    job = None
    memory_handle = None

    setup_started = time.perf_counter()

    try:
        job = _create_job()

        _assign_process(
            job=job,
            process_handle=process._handle,
        )

        memory_handle = _open_process_for_memory(
            process.pid
        )

        if process.stdin is None:
            raise RuntimeError(
                "Worker stdin pipe is unavailable."
            )

        process.stdin.write("GO\n")
        process.stdin.flush()
        process.stdin.close()

        parent_setup_ms = (
            time.perf_counter()
            - setup_started
        ) * 1000.0

        timed_out = False

        try:
            process.wait(
                timeout=timeout_seconds
            )

        except subprocess.TimeoutExpired:
            timed_out = True

            kernel32.TerminateJobObject(
                job,
                1,
            )

            process.wait(
                timeout=5.0
            )

        teardown_started = time.perf_counter()

        peak_rss = _peak_rss(
            memory_handle
        )

        payload = _load_worker_result(
            Path(command[
                command.index("--result-path")
                + 1
            ])
        )

        teardown_ms = (
            time.perf_counter()
            - teardown_started
        ) * 1000.0

        total_ms = (
            time.perf_counter()
            - total_started
        ) * 1000.0

        return _RawWorkerRun(
            payload=payload,
            return_code=process.returncode,
            timed_out=timed_out,
            worker_spawn_ms=spawn_ms,
            parent_setup_ms=parent_setup_ms,
            worker_teardown_ms=teardown_ms,
            end_to_end_ms=total_ms,
            worker_peak_rss_bytes=peak_rss,
        )

    finally:
        if (
            process.poll() is None
            and job is not None
        ):
            kernel32.TerminateJobObject(
                job,
                1,
            )

            try:
                process.wait(
                    timeout=5.0
                )

            except subprocess.TimeoutExpired:
                process.kill()

        if memory_handle:
            kernel32.CloseHandle(
                memory_handle
            )

        if job:
            kernel32.CloseHandle(job)


def _payload_number(
    payload: dict[str, object] | None,
    key: str,
) -> float | None:
    if payload is None:
        return None

    value = payload.get(key)

    if isinstance(value, (int, float)):
        return float(value)

    return None


def _payload_text(
    payload: dict[str, object] | None,
    key: str,
) -> str | None:
    if payload is None:
        return None

    value = payload.get(key)

    if isinstance(value, str):
        return value

    return None


def run_control_worker(
    *,
    mode: str,
    timeout_seconds: float,
    sleep_seconds: float = 0.0,
    allocation_bytes: int = 0,
) -> WorkerRunResult:
    if mode not in {
        "sleep",
        "memory",
    }:
        raise ValueError(
            f"Unsupported mode: {mode}"
        )

    with tempfile.TemporaryDirectory(
        prefix="govknowledge-parser-worker-"
    ) as temporary_directory:
        result_path = (
            Path(temporary_directory)
            / "result.json"
        )

        command = [
            sys.executable,
            "-m",
            (
                "app.infrastructure.parsing."
                "parser_worker_entry"
            ),
            "--mode",
            mode,
            "--result-path",
            str(result_path),
            "--sleep-seconds",
            str(sleep_seconds),
            "--allocation-bytes",
            str(allocation_bytes),
        ]

        raw = _execute_worker(
            command=command,
            timeout_seconds=timeout_seconds,
        )

        if raw.timed_out:
            status = ParsingStatus.FAILED
            failure = ParserFailureClass.TIMEOUT
            execution_ms = None

        elif (
            raw.payload is not None
            and raw.payload.get("status")
            == "FAILED"
            and raw.payload.get(
                "failure_class"
            )
            == "OOM"
        ):
            status = ParsingStatus.FAILED
            failure = ParserFailureClass.OOM
            execution_ms = _payload_number(
                raw.payload,
                "execution_ms",
            )

        elif (
            raw.return_code == 0
            and raw.payload is not None
            and raw.payload.get("status")
            == "SUCCESS"
        ):
            status = ParsingStatus.SUCCESS
            failure = None
            execution_ms = _payload_number(
                raw.payload,
                "execution_ms",
            )

        else:
            status = ParsingStatus.FAILED
            failure = (
                ParserFailureClass.NATIVE_CRASH
            )
            execution_ms = None

        return WorkerRunResult(
            parsing_status=status,
            failure_class=failure,
            return_code=raw.return_code,
            worker_spawn_ms=(
                raw.worker_spawn_ms
            ),
            worker_setup_ms=(
                raw.parent_setup_ms
            ),
            parser_execution_ms=execution_ms,
            worker_teardown_ms=(
                raw.worker_teardown_ms
            ),
            end_to_end_document_processing_ms=(
                raw.end_to_end_ms
            ),
            worker_peak_rss_bytes=(
                raw.worker_peak_rss_bytes
            ),
        )


def _deserialize_parsed_document(
    payload: dict[str, object],
) -> object:
    from app.application.ports.document_parser import (
        ParsedChunk,
        ParsedDocument,
        ParsedPage,
    )

    raw_document = payload.get(
        "parsed_document"
    )

    if not isinstance(raw_document, dict):
        raise ValueError(
            "Worker success payload has no "
            "parsed_document object."
        )

    raw_pages = raw_document.get("pages")
    raw_chunks = raw_document.get("chunks")

    if not isinstance(raw_pages, list):
        raise ValueError(
            "Worker pages payload is invalid."
        )

    if not isinstance(raw_chunks, list):
        raise ValueError(
            "Worker chunks payload is invalid."
        )

    pages = []

    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            raise ValueError(
                "Worker page payload is invalid."
            )

        pages.append(
            ParsedPage(
                page_number=int(
                    raw_page["page_number"]
                ),
                text=str(
                    raw_page["text"]
                ),
                metadata=dict(
                    raw_page["metadata"]
                ),
            )
        )

    chunks = []

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            raise ValueError(
                "Worker chunk payload is invalid."
            )

        section_title = raw_chunk.get(
            "section_title"
        )

        chunks.append(
            ParsedChunk(
                page_number=int(
                    raw_chunk["page_number"]
                ),
                chunk_index=int(
                    raw_chunk["chunk_index"]
                ),
                content=str(
                    raw_chunk["content"]
                ),
                section_title=(
                    str(section_title)
                    if section_title is not None
                    else None
                ),
                language=str(
                    raw_chunk["language"]
                ),
                metadata=dict(
                    raw_chunk["metadata"]
                ),
            )
        )

    return ParsedDocument(
        pages=tuple(pages),
        chunks=tuple(chunks),
    )


def run_document_worker(
    *,
    file_path: Path,
    language: str,
    timeout_seconds: float,
    artifacts_path: Path | None = None,
) -> DocumentWorkerRunResult:
    path = file_path.resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    if not language:
        raise ValueError(
            "language cannot be empty."
        )

    with tempfile.TemporaryDirectory(
        prefix="govknowledge-parser-worker-"
    ) as temporary_directory:
        result_path = (
            Path(temporary_directory)
            / "result.json"
        )

        command = [
            sys.executable,
            "-m",
            (
                "app.infrastructure.parsing."
                "parser_worker_entry"
            ),
            "--mode",
            "parse",
            "--result-path",
            str(result_path),
            "--file-path",
            str(path),
            "--language",
            language,
        ]

        if artifacts_path is not None:
            command.extend(
                [
                    "--artifacts-path",
                    str(
                        artifacts_path.resolve()
                    ),
                ]
            )

        raw = _execute_worker(
            command=command,
            timeout_seconds=timeout_seconds,
        )

        payload = raw.payload

        if raw.timed_out:
            return DocumentWorkerRunResult(
                parsing_status=(
                    ParsingStatus.FAILED
                ),
                failure_class=(
                    ParserFailureClass.TIMEOUT
                ),
                parsed_document=None,
                return_code=raw.return_code,
                worker_spawn_ms=(
                    raw.worker_spawn_ms
                ),
                worker_setup_ms=None,
                parser_execution_ms=None,
                worker_teardown_ms=(
                    raw.worker_teardown_ms
                ),
                end_to_end_document_processing_ms=(
                    raw.end_to_end_ms
                ),
                worker_peak_rss_bytes=(
                    raw.worker_peak_rss_bytes
                ),
                exception_type=None,
                exception_message=None,
            )

        if (
            raw.return_code == 0
            and payload is not None
            and payload.get("status")
            == "SUCCESS"
        ):
            parsed = (
                _deserialize_parsed_document(
                    payload
                )
            )

            return DocumentWorkerRunResult(
                parsing_status=(
                    ParsingStatus.SUCCESS
                ),
                failure_class=None,
                parsed_document=parsed,
                return_code=raw.return_code,
                worker_spawn_ms=(
                    raw.worker_spawn_ms
                ),
                worker_setup_ms=(
                    _payload_number(
                        payload,
                        "worker_setup_ms",
                    )
                ),
                parser_execution_ms=(
                    _payload_number(
                        payload,
                        "parser_execution_ms",
                    )
                ),
                worker_teardown_ms=(
                    raw.worker_teardown_ms
                ),
                end_to_end_document_processing_ms=(
                    raw.end_to_end_ms
                ),
                worker_peak_rss_bytes=(
                    raw.worker_peak_rss_bytes
                ),
                exception_type=None,
                exception_message=None,
            )

        if (
            payload is not None
            and payload.get("status")
            == "FAILED"
        ):
            raw_failure = payload.get(
                "failure_class"
            )

            try:
                failure = (
                    ParserFailureClass(
                        str(raw_failure)
                    )
                )

            except ValueError:
                failure = (
                    ParserFailureClass.UNKNOWN
                )

            return DocumentWorkerRunResult(
                parsing_status=(
                    ParsingStatus.FAILED
                ),
                failure_class=failure,
                parsed_document=None,
                return_code=raw.return_code,
                worker_spawn_ms=(
                    raw.worker_spawn_ms
                ),
                worker_setup_ms=(
                    _payload_number(
                        payload,
                        "worker_setup_ms",
                    )
                ),
                parser_execution_ms=(
                    _payload_number(
                        payload,
                        "parser_execution_ms",
                    )
                ),
                worker_teardown_ms=(
                    raw.worker_teardown_ms
                ),
                end_to_end_document_processing_ms=(
                    raw.end_to_end_ms
                ),
                worker_peak_rss_bytes=(
                    raw.worker_peak_rss_bytes
                ),
                exception_type=(
                    _payload_text(
                        payload,
                        "exception_type",
                    )
                ),
                exception_message=(
                    _payload_text(
                        payload,
                        "exception_message",
                    )
                ),
            )

        return DocumentWorkerRunResult(
            parsing_status=(
                ParsingStatus.FAILED
            ),
            failure_class=(
                ParserFailureClass.NATIVE_CRASH
            ),
            parsed_document=None,
            return_code=raw.return_code,
            worker_spawn_ms=(
                raw.worker_spawn_ms
            ),
            worker_setup_ms=None,
            parser_execution_ms=None,
            worker_teardown_ms=(
                raw.worker_teardown_ms
            ),
            end_to_end_document_processing_ms=(
                raw.end_to_end_ms
            ),
            worker_peak_rss_bytes=(
                raw.worker_peak_rss_bytes
            ),
            exception_type=None,
            exception_message=None,
        )