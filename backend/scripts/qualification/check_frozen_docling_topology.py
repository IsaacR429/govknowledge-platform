from collections import Counter
from pathlib import Path
from uuid import UUID

from sqlalchemy import select

from app.infrastructure.database.models.documents import (
    DocumentChunk,
)
from app.infrastructure.database.session import (
    SessionLocal,
)
from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    PROJECT_ROOT
    / "documents"
    / "sample"
    / "test-policy.pdf"
)

VERSION_ID = UUID(
    "86aa0794-a065-4819-a4fe-01694fad6122"
)

EXPECTED_PAGES = 54
EXPECTED_CHUNKS = 107


def main() -> None:
    print("=" * 72)
    print("FROZEN DOCLING TOPOLOGY CHECK")
    print("=" * 72)

    parser = DoclingDocumentParser()

    print()
    print("Parsing frozen fixture...")

    parsed = parser.parse(
        SOURCE,
        language="en",
    )

    print()
    print("New parse:")
    print("  pages :", len(parsed.pages))
    print("  chunks:", len(parsed.chunks))

    db = SessionLocal()

    try:
        stored = tuple(
            db.scalars(
                select(DocumentChunk)
                .where(
                    DocumentChunk.document_version_id
                    == VERSION_ID
                )
                .order_by(
                    DocumentChunk.chunk_index
                )
            )
        )

        print()
        print("Historical canonical:")
        print("  chunks:", len(stored))

        problems = []

        if len(parsed.pages) != EXPECTED_PAGES:
            problems.append(
                f"Page count changed: "
                f"{len(parsed.pages)} != {EXPECTED_PAGES}"
            )

        if len(parsed.chunks) != EXPECTED_CHUNKS:
            problems.append(
                f"Chunk count changed: "
                f"{len(parsed.chunks)} != {EXPECTED_CHUNKS}"
            )

        if len(stored) != EXPECTED_CHUNKS:
            problems.append(
                f"Historical DB does not contain "
                f"{EXPECTED_CHUNKS} chunks: "
                f"found {len(stored)}"
            )

        new_distribution = Counter(
            chunk.page_number
            for chunk in parsed.chunks
        )

        old_distribution = Counter(
            chunk.page_start
            for chunk in stored
        )

        changed_pages = sorted(
            set(new_distribution)
            | set(old_distribution)
        )

        changed_pages = [
            page
            for page in changed_pages
            if (
                new_distribution[page]
                != old_distribution[page]
            )
        ]

        print()
        print(
            "Pages with changed chunk counts:",
            changed_pages or "NONE",
        )

        if changed_pages:
            print()
            print("Per-page differences:")

            for page in changed_pages:
                print(
                    f"  page {page}: "
                    f"old={old_distribution[page]} "
                    f"new={new_distribution[page]}"
                )

        exact_mismatches = []

        for index, (
            new_chunk,
            old_chunk,
        ) in enumerate(
            zip(
                parsed.chunks,
                stored,
            )
        ):
            same = (
                new_chunk.chunk_index
                == old_chunk.chunk_index
                and new_chunk.page_number
                == old_chunk.page_start
                and new_chunk.content
                == old_chunk.content
            )

            if not same:
                exact_mismatches.append(
                    (
                        index,
                        new_chunk,
                        old_chunk,
                    )
                )

        if len(parsed.chunks) != len(stored):
            problems.append(
                "New and historical chunk sequences "
                "have different lengths."
            )

        if exact_mismatches:
            problems.append(
                f"{len(exact_mismatches)} canonical "
                "chunk(s) changed."
            )

            print()
            print(
                "First exact chunk differences:"
            )

            for (
                index,
                new_chunk,
                old_chunk,
            ) in exact_mismatches[:5]:
                print()
                print("-" * 72)
                print(
                    f"POSITION {index}"
                )
                print(
                    "OLD:",
                    f"index={old_chunk.chunk_index}",
                    f"page={old_chunk.page_start}",
                    f"length={len(old_chunk.content)}",
                )
                print(
                    repr(
                        old_chunk.content[:300]
                    )
                )

                print()
                print(
                    "NEW:",
                    f"index={new_chunk.chunk_index}",
                    f"page={new_chunk.page_number}",
                    f"length={len(new_chunk.content)}",
                )
                print(
                    repr(
                        new_chunk.content[:300]
                    )
                )

        print()
        print("=" * 72)

        if problems:
            print(
                "FROZEN DOCLING TOPOLOGY: CHANGED"
            )
            print("=" * 72)

            for problem in problems:
                print(
                    " -",
                    problem,
                )

            raise SystemExit(1)

        print(
            "FROZEN DOCLING TOPOLOGY: PASS"
        )
        print("=" * 72)

        print()
        print(
            "54 pages reproduced."
        )
        print(
            "107 chunks reproduced."
        )
        print(
            "Chunk page assignments and contents "
            "match the historical canonical version."
        )

    finally:
        db.close()


if __name__ == "__main__":
    main()
