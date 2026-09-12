from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


ARTICLE_HEADER_RE = re.compile(
    r"^Article\s+\d+\s*$",
    re.IGNORECASE,
)

CHAPTER_HEADER_RE = re.compile(
    r"^Chapter\s+.+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExperimentalChunk:
    page_number: int
    content: str
    article: str | None
    chapter: str | None
    section: str | None

    @property
    def structural_label(self) -> str:
        parts = [
            value
            for value in (
                self.chapter,
                self.section,
                self.article,
            )
            if value
        ]

        return " > ".join(parts)


def _label_name(
    item: object,
) -> str:
    label = getattr(
        item,
        "label",
        None,
    )

    if label is None:
        return ""

    return str(
        getattr(
            label,
            "value",
            label,
        )
    )


def _pages(
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

    values = []

    for entry in provenance:
        page_no = getattr(
            entry,
            "page_no",
            None,
        )

        if page_no is not None:
            values.append(
                int(page_no)
            )

    return tuple(
        sorted(
            set(values)
        )
    )


def _clean(
    text: str,
) -> str:
    return " ".join(
        text.split()
    )


class ExperimentalStructureChunker:
    """Detachable Docling structure-aware chunker.

    Canonical GovKnowledge chunks are not modified.
    """

    def __init__(
        self,
        *,
        max_chars: int = 2200,
    ) -> None:
        if max_chars < 200:
            raise ValueError(
                "max_chars is too small."
            )

        self._max_chars = max_chars

    def chunk(
        self,
        document: object,
    ) -> tuple[ExperimentalChunk, ...]:
        chunks: list[ExperimentalChunk] = []

        current_article: str | None = None
        current_chapter: str | None = None
        current_section: str | None = None
        current_page: int | None = None

        buffer: list[str] = []

        def flush() -> None:
            nonlocal buffer

            if (
                current_page is None
                or not buffer
            ):
                buffer = []
                return

            text = _clean(
                " ".join(buffer)
            )

            if text:
                chunks.extend(
                    self._split_large(
                        page_number=current_page,
                        content=text,
                        article=current_article,
                        chapter=current_chapter,
                        section=current_section,
                    )
                )

            buffer = []

        for item, _level in (
            document.iterate_items()
        ):
            label = _label_name(
                item
            )

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

            text = _clean(
                text
            )

            if not text:
                continue

            pages = _pages(
                item
            )

            if not pages:
                continue

            page = pages[0]

            # Preserve the current GovKnowledge
            # single-page evidence invariant.
            if (
                current_page is not None
                and page != current_page
            ):
                flush()

            current_page = page

            if label == "section_header":
                # Every structural header closes
                # the preceding evidence unit.
                flush()

                if ARTICLE_HEADER_RE.fullmatch(
                    text
                ):
                    current_article = text
                    continue

                # A new non-article heading means
                # the previous Article no longer
                # owns subsequent content.
                current_article = None

                if CHAPTER_HEADER_RE.fullmatch(
                    text
                ):
                    current_chapter = text
                    current_section = None
                    continue

                # Preserve other Docling headings
                # as contextual metadata.
                current_section = text
                continue

            if label in {
                "text",
                "list_item",
            }:
                buffer.append(
                    text
                )

        flush()

        return tuple(
            chunks
        )

    def _split_large(
        self,
        *,
        page_number: int,
        content: str,
        article: str | None,
        chapter: str | None,
        section: str | None,
    ) -> Iterable[ExperimentalChunk]:
        if len(content) <= self._max_chars:
            yield ExperimentalChunk(
                page_number=page_number,
                content=content,
                article=article,
                chapter=chapter,
                section=section,
            )
            return

        start = 0

        while start < len(content):
            end = min(
                start + self._max_chars,
                len(content),
            )

            if end < len(content):
                boundary = content.rfind(
                    ". ",
                    start,
                    end,
                )

                if boundary > start + 500:
                    end = boundary + 1

            piece = (
                content[
                    start:end
                ]
                .strip()
            )

            if piece:
                yield ExperimentalChunk(
                    page_number=page_number,
                    content=piece,
                    article=article,
                    chapter=chapter,
                    section=section,
                )

            start = end
