from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ARTIFACT_ROOT_ENV = "GOVKNOWLEDGE_DOCLING_ARTIFACT_ROOT"

DEFAULT_ARTIFACT_ROOT = (
    Path(__file__).resolve().parents[4]
    / ".runtime"
    / "artifacts"
    / "docling-p0"
)


def _write_result(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-path", type=Path)
    source.add_argument("--pdf-path", type=Path)
    parser.add_argument("--page-number", type=int, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.page_number < 1:
            raise ValueError("page-number must be positive")

        if args.pdf_path is not None:
            import numpy as np
            import pypdfium2 as pdfium

            pdf_path = args.pdf_path.resolve()
            if not pdf_path.is_file():
                raise FileNotFoundError(pdf_path)

            pdf = pdfium.PdfDocument(str(pdf_path))
            try:
                if args.page_number > len(pdf):
                    raise ValueError(
                        f"Page {args.page_number} exceeds "
                        f"PDF page count {len(pdf)}."
                    )

                page = pdf[args.page_number - 1]
                try:
                    bitmap = page.render(scale=2)
                    image = np.array(
                        bitmap.to_pil().convert("RGB")
                    )
                finally:
                    page.close()
            finally:
                pdf.close()
        else:
            image = args.image_path.resolve()
            if not image.is_file():
                raise FileNotFoundError(image)

        artifact_root = Path(
            os.environ.get(
                ARTIFACT_ROOT_ENV,
                str(DEFAULT_ARTIFACT_ROOT),
            )
        ).expanduser().resolve()

        models = artifact_root / "rapidocr" / "ppocr-v6"

        detector = models / "PP-OCRv6_det_small.onnx"
        classifier = models / "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
        recognizer = models / "PP-OCRv6_rec_small.onnx"

        for model in (detector, classifier, recognizer):
            if not model.is_file():
                raise FileNotFoundError(model)

        from rapidocr import EngineType, RapidOCR

        ocr = RapidOCR(
            params={
                "Det.engine_type": EngineType.ONNXRUNTIME,
                "Det.model_path": str(detector),
                "Cls.engine_type": EngineType.ONNXRUNTIME,
                "Cls.model_path": str(classifier),
                "Rec.engine_type": EngineType.ONNXRUNTIME,
                "Rec.model_path": str(recognizer),
                "Rec.rec_keys_path": None,
            }
        )

        result = ocr(
            image,
            use_det=True,
            use_cls=True,
            use_rec=True,
        )

        lines = [
            line.strip()
            for line in (result.txts or ())
            if isinstance(line, str) and line.strip()
        ]

        if not lines:
            _write_result(
                args.result_path,
                {
                    "status": "FAILED",
                    "error_code": "OCR_NO_TEXT",
                    "page_number": args.page_number,
                },
            )
            return 2

        _write_result(
            args.result_path,
            {
                "status": "SUCCESS",
                "page_number": args.page_number,
                "text": "\n".join(lines),
                "text_lines": lines,
                "extraction_method": "rapidocr_page_recovery",
                "ocr_backend": "onnxruntime",
            },
        )
        return 0

    except Exception as exc:
        _write_result(
            args.result_path,
            {
                "status": "FAILED",
                "error_code": "OCR_RUNTIME_ERROR",
                "page_number": args.page_number,
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:1000],
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
