from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(frozen=True)
class EmbeddingResult:
    values: tuple[float, ...]
    model_name: str
    model_revision: str
    dimension: int


class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str:
        ...

    @property
    def model_revision(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_query(
        self,
        text: str,
    ) -> EmbeddingResult:
        ...

    def embed_passages(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingResult, ...]:
        ...
