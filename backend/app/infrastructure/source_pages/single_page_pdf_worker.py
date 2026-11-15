"""Extract one PDF page in the isolated PyMuPDF environment.

Usage:
    python single_page_pdf_worker.py SOURCE EXPECTED_SHA256 PAGE_NUMBER

Successful stdout contains PDF bytes only.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        return 2

    source_path = Path(sys.argv[1])
    expected_hash = sys.argv[2].lower()
    page_number = int(sys.argv[3])

    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        return 1

    if page_number < 1 or not source_path.is_file():
        return 1

    # Check the original bytes against the database's file hash.
    digest = hashlib.sha256()
    with source_path.open("rb") as source_file:
        while block := source_file.read(1024 * 1024):
            digest.update(block)

    if not hmac.compare_digest(digest.hexdigest(), expected_hash):
        return 1

    import pymupdf

    with pymupdf.open(str(source_path)) as original:
        if original.needs_pass:
            return 1

        if page_number > original.page_count:
            return 1

        with pymupdf.open() as single_page:
            single_page.insert_pdf(
                original,
                from_page=page_number - 1,
                to_page=page_number - 1,
                links=False,
                annots=False,
            )
            pdf_bytes = single_page.tobytes(
                garbage=4,
                deflate=True,
            )

    if not pdf_bytes.startswith(b"%PDF-"):
        return 1

    sys.stdout.buffer.write(pdf_bytes)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Do not expose local paths or parser exceptions to clients.
        print("Source page preparation failed.", file=sys.stderr)
        raise SystemExit(1)
