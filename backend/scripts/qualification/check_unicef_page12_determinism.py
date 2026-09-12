from hashlib import sha256
from pathlib import Path

from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    PROJECT_ROOT
    / "documents"
    / "qualification"
    / "unicef-emergency-procedures-fr.pdf"
)


def digest(text: str) -> str:
    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = DoclingDocumentParser()

    parsed = parser.parse(
        SOURCE,
        language="fr",
    )

    chunks = tuple(
        chunk
        for chunk in parsed.chunks
        if chunk.page_number == 12
    )

    print("=" * 72)
    print("UNICEF PAGE 12 DETERMINISM")
    print("=" * 72)
    print("page12_chunks:", len(chunks))

    for chunk in chunks:
        print()
        print(
            f"index={chunk.chunk_index} "
            f"length={len(chunk.content)}"
        )
        print(
            "sha256=",
            digest(chunk.content),
        )
        print(
            "first=",
            repr(chunk.content[:100]),
        )
        print(
            "last=",
            repr(chunk.content[-100:]),
        )


if __name__ == "__main__":
    main()
