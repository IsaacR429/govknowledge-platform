from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import delete

from app.application.ports.authorization import (
    AuthorizationScope,
)
from app.domain.enums.permission import (
    PermissionAction,
)
from app.infrastructure.database.models.documents import (
    DocumentVersion,
)
from app.infrastructure.database.models.identity import (
    User,
)
from app.infrastructure.database.models.permissions import (
    DocumentPermission,
)
from app.infrastructure.database.session import (
    SessionLocal,
)
from app.infrastructure.retrieval.postgres_vector_retrieval import (
    PostgresVectorRetrievalAdapter,
)
from app.infrastructure.retrieval.sentence_transformer_embeddings import (
    SentenceTransformerEmbeddingProvider,
)


DATASET = Path(
    "tests/fixtures/evidence_calibration_v1.json"
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m "
            "scripts.calibrate_evidence_policy "
            "<version-uuid>"
        )

    version_id = UUID(
        sys.argv[1]
    )

    cases = json.loads(
        DATASET.read_text(
            encoding="utf-8"
        )
    )

    db = SessionLocal()

    user_id = uuid4()
    permission_id = uuid4()

    try:
        version = db.get(
            DocumentVersion,
            version_id,
        )

        if version is None:
            raise RuntimeError(
                "Document version not found."
            )

        document_id = version.document_id

        username = (
            "evidence-calibration-"
            + user_id.hex[:12]
        )

        db.add(
            User(
                id=user_id,
                username=username,
                display_name=(
                    "Evidence Calibration"
                ),
                email=None,
                department="Evaluation",
                language_preference="fr",
                is_active=True,
            )
        )

        db.flush()

        db.add(
            DocumentPermission(
                id=permission_id,
                user_id=user_id,
                document_id=document_id,
                action=(
                    PermissionAction.READ.value
                ),
            )
        )

        db.commit()

        scope = AuthorizationScope(
            user_id=user_id,
            action=PermissionAction.READ,
            document_ids=frozenset(
                {document_id}
            ),
        )

        provider = (
            SentenceTransformerEmbeddingProvider()
        )

        retrieval = (
            PostgresVectorRetrievalAdapter(
                db
            )
        )

        measurements = []

        print()
        print(
            "=== Evidence calibration v1 ==="
        )
        print(
            "Cases:",
            len(cases),
        )
        print(
            "Model:",
            provider.model_name,
        )
        print(
            "Revision:",
            provider.model_revision,
        )
        print()

        for case in cases:
            embedded = provider.embed_query(
                case["question"]
            )

            results = retrieval.search_authorized(
                scope=scope,
                query_vector=embedded.values,
                model_name=embedded.model_name,
                model_revision=(
                    embedded.model_revision
                ),
                top_k=5,
            )

            scores = [
                item.score
                for item in results
            ]

            top1 = (
                scores[0]
                if scores
                else None
            )

            top2 = (
                scores[1]
                if len(scores) > 1
                else None
            )

            mean_top3 = (
                statistics.mean(
                    scores[:3]
                )
                if scores
                else None
            )

            gap = (
                top1 - top2
                if (
                    top1 is not None
                    and top2 is not None
                )
                else None
            )

            measurements.append(
                {
                    "id": case["id"],
                    "label": case["label"],
                    "top1": top1,
                    "top2": top2,
                    "mean_top3": mean_top3,
                    "gap": gap,
                }
            )

            print(
                f"{case['id']} "
                f"{case['label']:<12} "
                f"top1={top1:.4f} "
                f"top2={top2:.4f} "
                f"mean3={mean_top3:.4f} "
                f"gap={gap:.4f}"
            )

            print(
                "   ",
                case["question"],
            )

        print()
        print(
            "=== Distribution summary ==="
        )

        for label in (
            "ANSWERABLE",
            "UNANSWERABLE",
        ):
            group = [
                item["top1"]
                for item in measurements
                if (
                    item["label"] == label
                    and item["top1"]
                    is not None
                )
            ]

            print(
                f"{label}: "
                f"min={min(group):.4f} "
                f"median={statistics.median(group):.4f} "
                f"max={max(group):.4f}"
            )

        answerable = [
            item["top1"]
            for item in measurements
            if item["label"] == "ANSWERABLE"
        ]

        unanswerable = [
            item["top1"]
            for item in measurements
            if item["label"] == "UNANSWERABLE"
        ]

        margin = (
            min(answerable)
            - max(unanswerable)
        )

        print()
        print(
            "Top-score separation margin:",
            f"{margin:.4f}",
        )

        if margin > 0:
            print(
                "Top-score-only separation:",
                "OBSERVED"
            )
        else:
            print(
                "Top-score-only separation:",
                "NOT OBSERVED"
            )

    finally:
        db.rollback()

        db.execute(
            delete(
                DocumentPermission
            ).where(
                DocumentPermission.id
                == permission_id
            )
        )

        db.execute(
            delete(
                User
            ).where(
                User.id == user_id
            )
        )

        db.commit()
        db.close()


if __name__ == "__main__":
    main()
