from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


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

        # Touch every 4 KiB page so the allocation
        # cannot remain merely virtual/reserved.
        for offset in range(
            0,
            len(block),
            4096,
        ):
            block[offset] = 1

        blocks.append(block)
        allocated += current


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "sleep",
            "memory",
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

    args = parser.parse_args()

    _wait_for_parent_signal()

    started = time.perf_counter()

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

        else:
            raise AssertionError(
                f"Unsupported worker mode: {args.mode}"
            )

    except MemoryError:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        _write_result(
            args.result_path,
            {
                "status": "FAILED",
                "failure_class": "OOM",
                "execution_ms": elapsed_ms,
            },
        )

        return 42

    except BaseException as exc:
        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000.0

        _write_result(
            args.result_path,
            {
                "status": "FAILED",
                "failure_class": "UNKNOWN",
                "execution_ms": elapsed_ms,
                "exception_type": (
                    type(exc).__name__
                ),
            },
        )

        return 1

    elapsed_ms = (
        time.perf_counter() - started
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
