from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium


class PdfNormalizationError(RuntimeError):
    """The PDF could not be normalized safely."""


class PostNormalizationMismatchError(
    PdfNormalizationError
):
    """The derived PDF failed integrity validation."""


@dataclass(frozen=True)
class PdfNormalizationResult:
    original_path: Path
    derived_path: Path
    original_sha256: str
    derived_sha256: str
    expected_page_count: int
    source_page_count: int
    derived_page_count: int
    first_page_text_sha256: str
    last_page_text_sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _page_text_sha256(
    pdf: pdfium.PdfDocument,
    index: int,
) -> str:
    page = pdf[index]

    try:
        text_page = page.get_textpage()

        try:
            text = (
                text_page.get_text_bounded()
                or ""
            )

        finally:
            text_page.close()

    finally:
        page.close()

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def normalize_pdf_copy(
    *,
    source_path: Path,
    derived_path: Path,
    expected_page_count: int,
) -> PdfNormalizationResult:
    """Create and validate one derived PDF rewrite.

    The source is never modified.

    expected_page_count must come from the
    independent structural preflight layer.
    """

    source = source_path.resolve()
    derived = derived_path.resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if expected_page_count < 1:
        raise ValueError(
            "expected_page_count must be positive."
        )

    if source == derived:
        raise ValueError(
            "derived_path must differ from source_path."
        )

    derived.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_hash_before = _sha256_file(
        source
    )

    try:
        source_pdf = pdfium.PdfDocument(
            source
        )

        try:
            source_page_count = len(
                source_pdf
            )

            if (
                source_page_count
                != expected_page_count
            ):
                raise (
                    PostNormalizationMismatchError(
                        "Source page count does not "
                        "match independent preflight: "
                        f"{source_page_count} != "
                        f"{expected_page_count}"
                    )
                )

            first_text_before = (
                _page_text_sha256(
                    source_pdf,
                    0,
                )
            )

            last_text_before = (
                _page_text_sha256(
                    source_pdf,
                    source_page_count - 1,
                )
            )

            derived_pdf = (
                pdfium.PdfDocument.new()
            )

            try:
                derived_pdf.import_pages(
                    source_pdf
                )

                derived_pdf.save(
                    derived
                )

            finally:
                derived_pdf.close()

        finally:
            source_pdf.close()

    except PostNormalizationMismatchError:
        raise

    except Exception as exc:
        raise PdfNormalizationError(
            "PDF rewrite failed."
        ) from exc

    original_hash_after = _sha256_file(
        source
    )

    if (
        original_hash_after
        != original_hash_before
    ):
        raise PdfNormalizationError(
            "Original PDF changed during "
            "normalization."
        )

    if not derived.is_file():
        raise PdfNormalizationError(
            "Derived PDF was not created."
        )

    derived_hash = _sha256_file(
        derived
    )

    try:
        check_pdf = pdfium.PdfDocument(
            derived
        )

        try:
            derived_page_count = len(
                check_pdf
            )

            if (
                derived_page_count
                != expected_page_count
            ):
                raise (
                    PostNormalizationMismatchError(
                        "Derived page count changed: "
                        f"{derived_page_count} != "
                        f"{expected_page_count}"
                    )
                )

            for index in range(
                derived_page_count
            ):
                page = check_pdf[index]

                try:
                    box = page.get_mediabox(
                        fallback_ok=False
                    )

                finally:
                    page.close()

                if box is None:
                    raise (
                        PostNormalizationMismatchError(
                            "Derived page "
                            f"{index + 1} has no "
                            "explicit MediaBox."
                        )
                    )

                left, bottom, right, top = box

                if (
                    right <= left
                    or top <= bottom
                ):
                    raise (
                        PostNormalizationMismatchError(
                            "Derived page "
                            f"{index + 1} has "
                            "invalid dimensions."
                        )
                    )

            first_text_after = (
                _page_text_sha256(
                    check_pdf,
                    0,
                )
            )

            last_text_after = (
                _page_text_sha256(
                    check_pdf,
                    derived_page_count - 1,
                )
            )

        finally:
            check_pdf.close()

    except PostNormalizationMismatchError:
        raise

    except Exception as exc:
        raise PdfNormalizationError(
            "Derived PDF validation failed."
        ) from exc

    if first_text_after != first_text_before:
        raise PostNormalizationMismatchError(
            "First-page text identity changed."
        )

    if last_text_after != last_text_before:
        raise PostNormalizationMismatchError(
            "Last-page text identity changed."
        )

    return PdfNormalizationResult(
        original_path=source,
        derived_path=derived,
        original_sha256=(
            original_hash_before
        ),
        derived_sha256=derived_hash,
        expected_page_count=(
            expected_page_count
        ),
        source_page_count=(
            source_page_count
        ),
        derived_page_count=(
            derived_page_count
        ),
        first_page_text_sha256=(
            first_text_before
        ),
        last_page_text_sha256=(
            last_text_before
        ),
    )
