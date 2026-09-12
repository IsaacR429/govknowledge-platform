from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import numpy as np
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer

from app.application.ports.authorization import AuthorizationScope
from app.domain.enums.permission import PermissionAction
from app.infrastructure.database.models.documents import DocumentVersion
from app.infrastructure.database.session import SessionLocal
from research.parsing.experimental_structure_chunker import (
    ExperimentalStructureChunker,
)
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)


VERSION_ID = UUID(
    "86aa0794-a065-4819-a4fe-01694fad6122"
)

MODEL_NAME = "intfloat/multilingual-e5-base"

MODEL_REVISION = (
    "d128750597153bb5987e10b1c3493a34e5a4502a"
)

REPO_ROOT = Path(__file__).resolve().parents[3]

SOURCE = (
    REPO_ROOT
    / "documents"
    / "sample"
    / "test-policy.pdf"
)

DATASET = Path(
    "tests/fixtures/evidence_calibration_v1.json"
)


def clean(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    cases = {
        item["id"]: item
        for item in json.loads(
            DATASET.read_text(encoding="utf-8")
        )
    }

    question = cases["A02"]["question"]

    model = SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )

    query = model.encode(
        ["query: " + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    parsed = (
        DocumentConverter()
        .convert(
            SOURCE,
            raises_on_error=True,
        )
    )

    chunks = (
        ExperimentalStructureChunker(
            max_chars=2200
        )
        .chunk(parsed.document)
    )

    vectors = model.encode(
        [
            "passage: " + item.content
            for item in chunks
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    order = np.argsort(
        -(vectors @ query)
    )

    db = SessionLocal()

    try:
        version = db.get(
            DocumentVersion,
            VERSION_ID,
        )

        scope = AuthorizationScope(
            user_id=UUID(
                "00000000-0000-0000-0000-000000000001"
            ),
            action=PermissionAction.READ,
            document_ids=frozenset(
                {version.document_id}
            ),
        )

        baseline = (
            PostgresVectorRetrievalAdapter(db)
        )

        current = baseline.search_authorized(
            scope=scope,
            query_vector=query.tolist(),
            model_name=MODEL_NAME,
            model_revision=MODEL_REVISION,
            top_k=5,
        )

        print()
        print("=== A02 baseline top 5 ===")

        for rank, item in enumerate(
            current,
            start=1,
        ):
            print()
            print(
                rank,
                "page=",
                item.page_number,
                "score=",
                f"{item.score:.4f}",
            )
            print(
                clean(item.content)[:800]
            )

        print()
        print("=== A02 structure top 5 ===")

        scores = vectors @ query

        for rank, index in enumerate(
            order[:5],
            start=1,
        ):
            item = chunks[int(index)]

            print()
            print(
                rank,
                "page=",
                item.page_number,
                "article=",
                item.article,
                "score=",
                f"{scores[index]:.4f}",
            )
            print(
                clean(item.content)[:800]
            )

    finally:
        db.rollback()
        db.close()


if __name__ == "__main__":
    main()
