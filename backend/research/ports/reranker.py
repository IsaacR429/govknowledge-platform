from dataclasses import dataclass
from typing import Protocol, Sequence

from app.application.ports.retrieval import RetrievedChunk


@dataclass(frozen=True)
class RerankedChunk:
    chunk: RetrievedChunk
    rerank_score: float


class RerankerPort(Protocol):
    @property
    def model_name(self) -> str:
        ...

    @property
    def model_revision(self) -> str:
        ...

    def rerank(
        self,
        *,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_k: int,
    ) -> tuple[RerankedChunk, ...]:
        ...
