from uuid import UUID

from sqlalchemy import select

from app.infrastructure.database.models.documents import (
    DocumentChunk,
)
from app.infrastructure.database.session import SessionLocal


VERSION_ID = UUID(
    "7326d7cb-2fad-400c-84aa-526073914849"
)

PAGES = (3, 5, 18, 21, 26)


def main() -> None:
    db = SessionLocal()

    try:
        for page_number in PAGES:
            print()
            print("=" * 70)
            print(f"PAGE {page_number}")
            print("=" * 70)

            chunks = list(
                db.scalars(
                    select(DocumentChunk)
                    .where(
                        DocumentChunk.document_version_id
                        == VERSION_ID,
                        DocumentChunk.page_start
                        == page_number,
                    )
                    .order_by(
                        DocumentChunk.chunk_index
                    )
                )
            )

            print(
                f"Chunks on page: {len(chunks)}"
            )

            for chunk in chunks:
                print()
                print(
                    f"[chunk {chunk.chunk_index}]"
                )
                print(chunk.content)

    finally:
        db.close()


if __name__ == "__main__":
    main()
