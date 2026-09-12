from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
)


DEFAULT_SOURCE = Path(
    "documents/sample/test-policy.pdf"
)

INTERESTING_RE = re.compile(
    r"\b(?:ARTICLE\s+\d+|CHAPTER\s+[A-Z0-9IVXLC]+)\b",
    re.IGNORECASE,
)


def label_name(item: object) -> str:
    label = getattr(
        item,
        "label",
        None,
    )

    if label is None:
        return "<none>"

    return str(
        getattr(
            label,
            "value",
            label,
        )
    )


def page_numbers(
    item: object,
) -> tuple[int, ...]:
    provenance = (
        getattr(
            item,
            "prov",
            None,
        )
        or ()
    )

    pages = []

    for entry in provenance:
        page = getattr(
            entry,
            "page_no",
            None,
        )

        if page is not None:
            pages.append(
                int(page)
            )

    return tuple(
        sorted(
            set(pages)
        )
    )


def clean(
    value: str,
    limit: int = 500,
) -> str:
    value = " ".join(
        value.split()
    )

    if len(value) <= limit:
        return value

    return value[:limit] + "..."


def main() -> None:
    source = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_SOURCE
    )

    if not source.exists():
        raise SystemExit(
            f"Source not found: {source}"
        )

    print()
    print(
        "=== Docling structural inspection ==="
    )

    print(
        "Source:",
        source,
    )

    converter = (
        DocumentConverter()
    )

    result = converter.convert(
        source,
        raises_on_error=True,
    )

    document = result.document

    label_counts = Counter()
    interesting = []

    total_items = 0

    for item, level in (
        document.iterate_items()
    ):
        total_items += 1

        label = label_name(
            item
        )

        label_counts[
            label
        ] += 1

        text = getattr(
            item,
            "text",
            None,
        )

        if not isinstance(
            text,
            str,
        ):
            continue

        if not INTERESTING_RE.search(
            text
        ):
            continue

        interesting.append(
            {
                "level": level,
                "label": label,
                "pages": page_numbers(
                    item
                ),
                "text": text,
            }
        )

    print()
    print(
        "Total document items:",
        total_items,
    )

    print()
    print(
        "=== Label distribution ==="
    )

    for label, count in (
        label_counts.most_common()
    ):
        print(
            f"{label:<25} {count}"
        )

    print()
    print(
        "=== Items containing Article/Chapter ==="
    )

    print(
        "Matching items:",
        len(interesting),
    )

    for index, item in enumerate(
        interesting[:80],
        start=1,
    ):
        pages = (
            ",".join(
                str(page)
                for page
                in item["pages"]
            )
            or "-"
        )

        print()
        print(
            f"[{index}] "
            f"level={item['level']} "
            f"label={item['label']} "
            f"pages={pages}"
        )

        print(
            clean(
                item["text"]
            )
        )


if __name__ == "__main__":
    main()
