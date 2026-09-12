from pathlib import Path

from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)


PDF_PATH = Path(
    r"C:\Users\isaac\govknowledge\documents"
    r"\qualification"
    r"\unicef-emergency-procedures-fr.pdf"
)

PAGE_NUMBER = 12


def preview(text: str, start: int, end: int) -> str:
    start = max(0, start)
    end = min(len(text), end)

    return text[start:end]


def main() -> None:
    parser = DoclingDocumentParser()

    conversion = parser._converter.convert(
        PDF_PATH.resolve(),
        raises_on_error=True,
    )

    document = conversion.document

    raw = document.export_to_text(
        page_no=PAGE_NUMBER,
        traverse_pictures=True,
    ).strip()

    print("=" * 72)
    print("RAW DOCLING PAGE 12")
    print("=" * 72)

    print("Total characters:", len(raw))

    paragraphs = [
        part.strip()
        for part in raw.split("\n\n")
        if part.strip()
    ]

    print("Paragraph count:", len(paragraphs))
    print(
        "Longest paragraph:",
        max(
            (len(p) for p in paragraphs),
            default=0,
        ),
    )

    print()
    print("Occurrence counts:")
    print(
        "500 000:",
        raw.count("500 000"),
    )
    print(
        "750 000:",
        raw.count("750 000"),
    )
    print(
        "PEAS:",
        raw.count("PEAS"),
    )
    print(
        "En cas d'urgence de niveau 2:",
        raw.count(
            "En cas d'urgence de niveau 2"
        ),
    )
    print(
        "En cas d'urgence de niveau 3:",
        raw.count(
            "En cas d'urgence de niveau 3"
        ),
    )

    target = raw.find("500 000")

    print()
    print("=" * 72)
    print("RAW CALLOUT REGION")
    print("=" * 72)

    if target == -1:
        print("500 000 not found.")
    else:
        print(
            preview(
                raw,
                target - 500,
                target + 1800,
            )
        )

    pieces = parser._split_page(raw)

    print()
    print("=" * 72)
    print("CHUNK OUTPUT")
    print("=" * 72)

    print("Number of page-12 chunks:", len(pieces))

    for index, piece in enumerate(pieces):
        interesting = (
            "500 000" in piece
            or "750 000" in piece
            or "En cas d'urgen" in piece
            or piece.startswith("ce de niveau 3")
        )

        if not interesting:
            continue

        print()
        print("-" * 72)
        print(
            f"LOCAL PIECE {index} — "
            f"{len(piece)} chars"
        )
        print("-" * 72)

        print("FIRST 250 CHARS:")
        print(repr(piece[:250]))

        print()
        print("LAST 300 CHARS:")
        print(repr(piece[-300:]))

        print()
        print("COUNTS:")
        print(
            "500 000:",
            piece.count("500 000"),
        )
        print(
            "750 000:",
            piece.count("750 000"),
        )
        print(
            "PEAS:",
            piece.count("PEAS"),
        )

    print()
    print("=" * 72)
    print("BOUNDARY CHECK")
    print("=" * 72)

    for index in range(
        len(pieces) - 1
    ):
        left = pieces[index]
        right = pieces[index + 1]

        if (
            "En cas d'urgen" in left[-100:]
            or right.startswith("ce de niveau 3")
        ):
            print(
                f"Boundary between local pieces "
                f"{index} and {index + 1}"
            )
            print()
            print("LEFT END:")
            print(repr(left[-250:]))
            print()
            print("RIGHT START:")
            print(repr(right[:250]))


if __name__ == "__main__":
    main()
