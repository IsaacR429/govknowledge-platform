from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path


DEFAULT_ARTIFACT_ROOT = (
    Path.home()
    / ".govknowledge"
    / "artifacts"
    / "docling-p0"
)

ARTIFACT_ROOT_ENV = (
    "GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT"
)

OCR_MODEL_FAMILY = "PP-OCRv6"
OCR_BACKEND = "onnxruntime"

CHUNKING_STRATEGY = (
    "paragraph_greedy_char_limit"
)


def _write_result(
    path: Path,
    payload: dict[str, object],
) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _wait_for_parent_signal() -> None:
    signal = input()

    if signal != "GO":
        raise RuntimeError(
            "Worker did not receive expected GO signal."
        )


def _sha256_file(
    path: Path,
) -> str:
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


def _resolve_artifact_root() -> Path:
    configured = os.environ.get(
        ARTIFACT_ROOT_ENV
    )

    if configured:
        return Path(
            configured
        ).expanduser().resolve()

    return DEFAULT_ARTIFACT_ROOT.resolve()


def _resolve_ocr_models(
    artifact_root: Path,
) -> tuple[
    Path,
    Path,
    Path,
]:
    root = artifact_root.resolve()

    if not root.is_dir():
        raise FileNotFoundError(
            f"Docling artifact root missing: {root}"
        )

    model_root = (
        root
        / "rapidocr"
        / "ppocr-v6"
    )

    detector = (
        model_root
        / "PP-OCRv6_det_small.onnx"
    )

    classifier = (
        model_root
        / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    )

    recognizer = (
        model_root
        / "PP-OCRv6_rec_small.onnx"
    )

    for path in (
        detector,
        classifier,
        recognizer,
    ):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required OCR model missing: {path}"
            )

    return (
        detector,
        classifier,
        recognizer,
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
        if (
            len(paragraph)
            > max_chunk_chars
        ):
            flush()

            remaining = paragraph

            while remaining:
                if (
                    len(remaining)
                    <= max_chunk_chars
                ):
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
        raise FileNotFoundError(
            source
        )

    if not language:
        raise ValueError(
            "language cannot be empty."
        )

    artifact_root = (
        _resolve_artifact_root()
    )

    (
        det_model,
        cls_model,
        rec_model,
    ) = _resolve_ocr_models(
        artifact_root
    )

    setup_started = time.perf_counter()

    from docling.datamodel.base_models import (
        InputFormat,
    )

    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
    )

    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
    )

    runtime = {
        "docling_version":
            version("docling"),

        "docling_core_version":
            version("docling-core"),

        "docling_ibm_models_version":
            version("docling-ibm-models"),

        "docling_parse_version":
            version("docling-parse"),

        "rapidocr_version":
            version("rapidocr"),

        "onnxruntime_version":
            version("onnxruntime"),

        "torch_version":
            version("torch"),
    }

    model_hashes = {
        "ocr_det_model_sha256":
            _sha256_file(
                det_model
            ),

        "ocr_cls_model_sha256":
            _sha256_file(
                cls_model
            ),

        "ocr_rec_model_sha256":
            _sha256_file(
                rec_model
            ),
    }

    pipeline_options = (
        PdfPipelineOptions(
            artifacts_path=(
                artifact_root
            ),
            do_ocr=True,
            ocr_options=RapidOcrOptions(
                backend=OCR_BACKEND,
                det_model_path=str(
                    det_model
                ),
                cls_model_path=str(
                    cls_model
                ),
                rec_model_path=str(
                    rec_model
                ),
                rapidocr_params={
                    "Rec.rec_keys_path":
                        None,
                },
            ),
        )
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF:
                PdfFormatOption(
                    pipeline_options=(
                        pipeline_options
                    )
                )
        }
    )

    worker_setup_ms = (
        time.perf_counter()
        - setup_started
    ) * 1000.0

    execution_started = (
        time.perf_counter()
    )

    conversion = converter.convert(
        source,
        raises_on_error=True,
    )

    document = conversion.document

    raw_pages = getattr(
        document,
        "pages",
        None,
    )

    if not isinstance(
        raw_pages,
        dict,
    ):
        raise ValueError(
            "Modern Docling document.pages "
            "is not a mapping."
        )

    page_numbers = sorted(
        int(page_number)
        for page_number
        in raw_pages.keys()
    )

    if not page_numbers:
        raise ValueError(
            "Modern Docling produced no pages."
        )

    pages: list[
        dict[str, object]
    ] = []

    chunks: list[
        dict[str, object]
    ] = []

    chunk_index = 0

    for page_number in page_numbers:
        page_text = (
            document.export_to_text(
                page_no=page_number,
                traverse_pictures=True,
            )
            .strip()
        )

        metadata: dict[
            str,
            object,
        ] = {
            "parser":
                "docling",

            "parser_variant":
                "modern_fallback",

            "docling_version":
                runtime[
                    "docling_version"
                ],

            "docling_core_version":
                runtime[
                    "docling_core_version"
                ],

            "docling_ibm_models_version":
                runtime[
                    "docling_ibm_models_version"
                ],

            "docling_parse_version":
                runtime[
                    "docling_parse_version"
                ],

            "rapidocr_version":
                runtime[
                    "rapidocr_version"
                ],

            "onnxruntime_version":
                runtime[
                    "onnxruntime_version"
                ],

            "torch_version":
                runtime[
                    "torch_version"
                ],

            "max_chunk_chars":
                max_chunk_chars,

            "page_number":
                page_number,

            "ocr_requested":
                True,

            "ocr_backend":
                OCR_BACKEND,

            "ocr_model_family":
                OCR_MODEL_FAMILY,

            "chunking_strategy":
                CHUNKING_STRATEGY,

            **model_hashes,
        }

        pages.append(
            {
                "page_number":
                    page_number,

                "text":
                    page_text,

                "metadata":
                    metadata.copy(),
            }
        )

        for chunk_text in _split_page(
            page_text,
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
                "status":
                    "SUCCESS",

                "failure_class":
                    None,

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
                "status":
                    "FAILED",

                "failure_class":
                    "OOM",

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
                "status":
                    "FAILED",

                "failure_class":
                    "UNKNOWN",

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