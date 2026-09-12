from __future__ import annotations

import re
from collections import Counter
from uuid import UUID

from sqlalchemy import select

from app.infrastructure.database.models.documents import (
    DocumentChunk,
)


VERSION_ID = UUID(
    "86aa0794-a065-4819-a4fe-01694fad6122"
)


ARTICLE_RE = re.compile(
    r"\bArticle\s+\d+\b",
    re.IGNORECASE,
)

CHAPTER_RE = re.compile(
    r"\bChapter\s+[A-Z0-9IVXLC]+\b",
    re.IGNORECASE,
)


def preview(
    text: str,
    limit: int = 350,
) -> str:
    text = " ".join(
        text.split()
    )

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def main() -> None:
    # Import here so this diagnostic remains
    # isolated from application startup.
    from app.infrastructure.database.session import (
        SessionLocal,
    )

    db = SessionLocal()

    try:
        chunks = tuple(
            db.scalars(
                select(
                    DocumentChunk
                )
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
        print(
            "=== Chunk-boundary diagnostic ==="
        )

        print(
            "Canonical chunks:",
            len(chunks),
        )

        distribution = Counter()

        multi_article = []
        multi_structure = []

        for chunk in chunks:
            article_matches = tuple(
                ARTICLE_RE.findall(
                    chunk.content
                )
            )

            chapter_matches = tuple(
                CHAPTER_RE.findall(
                    chunk.content
                )
            )

            article_count = len(
                article_matches
            )

            chapter_count = len(
                chapter_matches
            )

            distribution[
                article_count
            ] += 1

            if article_count >= 2:
                multi_article.append(
                    (
                        chunk,
                        article_matches,
                    )
                )

            if (
                article_count >= 2
                or (
                    article_count >= 1
                    and chapter_count >= 1
                )
            ):
                multi_structure.append(
                    (
                        chunk,
                        article_matches,
                        chapter_matches,
                    )
                )

        print()
        print(
            "Article-boundary distribution:"
        )

        for count in sorted(
            distribution
        ):
            print(
                f"  {count} article markers: "
                f"{distribution[count]} chunks"
            )

        print()
        print(
            "Chunks containing >=2 Article markers:",
            len(multi_article),
        )

        percentage = (
            100.0
            * len(multi_article)
            / len(chunks)
            if chunks
            else 0.0
        )

        print(
            "Percentage:",
            f"{percentage:.1f}%",
        )

        print()
        print(
            "=== Examples ==="
        )

        for (
            chunk,
            articles,
            chapters,
        ) in multi_structure[:10]:
            print()
            print(
                f"chunk={chunk.chunk_index} "
                f"page={chunk.page_start}"
            )

            print(
                "Articles:",
                ", ".join(
                    articles
                )
                if articles
                else "-"
            )

            print(
                "Chapters:",
                ", ".join(
                    chapters
                )
                if chapters
                else "-"
            )

            print(
                preview(
                    chunk.content
                )
            )

    finally:
        db.rollback()
        db.close()


if __name__ == "__main__":
    main()
