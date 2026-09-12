from importlib.metadata import version
from pathlib import Path

from app.infrastructure.parsing.docling_parser import (
    DoclingDocumentParser,
)


class FakeDocument:
    def __init__(self):
        self.pages = {
            1: object(),
            2: object(),
        }

    def export_to_text(
        self,
        *,
        page_no,
        traverse_pictures,
    ):
        assert traverse_pictures is True

        if page_no == 1:
            return (
                "Premier paragraphe.\n\n"
                "Deuxième paragraphe."
            )

        if page_no == 2:
            return (
                "Contenu de la deuxième page."
            )

        raise AssertionError(
            f"Unexpected page: {page_no}"
        )


class FakeConversion:
    def __init__(self):
        self.document = FakeDocument()


class FakeConverter:
    def convert(
        self,
        path,
        *,
        raises_on_error,
    ):
        assert isinstance(path, Path)
        assert raises_on_error is True

        return FakeConversion()


def test_docling_parser_preserves_page_provenance(
    tmp_path,
):
    file_path = tmp_path / "policy.pdf"

    file_path.write_bytes(
        b"fake-pdf"
    )

    parser = DoclingDocumentParser(
        converter=FakeConverter(),
        max_chunk_chars=100,
    )

    parsed = parser.parse(
        file_path,
        language="fr",
    )

    assert len(parsed.pages) == 2

    assert [
        page.page_number
        for page in parsed.pages
    ] == [1, 2]

    assert (
        parsed.pages[0].text
        == (
            "Premier paragraphe.\n\n"
            "Deuxième paragraphe."
        )
    )

    assert len(parsed.chunks) == 2

    assert [
        chunk.page_number
        for chunk in parsed.chunks
    ] == [1, 2]

    assert [
        chunk.chunk_index
        for chunk in parsed.chunks
    ] == [0, 1]

    assert all(
        chunk.language == "fr"
        for chunk in parsed.chunks
    )

    assert all(
        chunk.metadata["parser"]
        == "docling"
        for chunk in parsed.chunks
    )


    metadata = parsed.pages[0].metadata

    assert (
        metadata["docling_version"]
        == version("docling")
    )
    assert (
        metadata["docling_core_version"]
        == version("docling-core")
    )
    assert (
        metadata[
            "docling_ibm_models_version"
        ]
        == version("docling-ibm-models")
    )
    assert (
        metadata["docling_parse_version"]
        == version("docling-parse")
    )


def test_docling_parser_never_crosses_pages(
    tmp_path,
):
    file_path = tmp_path / "policy.pdf"

    file_path.write_bytes(
        b"fake-pdf"
    )

    parser = DoclingDocumentParser(
        converter=FakeConverter(),
        max_chunk_chars=20,
    )

    parsed = parser.parse(
        file_path,
        language="fr",
    )

    assert parsed.chunks

    assert all(
        chunk.page_number in {1, 2}
        for chunk in parsed.chunks
    )

    assert all(
        len(chunk.content) <= 20
        for chunk in parsed.chunks
    )

    page_one_chunks = [
        chunk
        for chunk in parsed.chunks
        if chunk.page_number == 1
    ]

    page_two_chunks = [
        chunk
        for chunk in parsed.chunks
        if chunk.page_number == 2
    ]

    assert page_one_chunks
    assert page_two_chunks

    assert all(
        chunk.page_number == 1
        for chunk in page_one_chunks
    )

    assert all(
        chunk.page_number == 2
        for chunk in page_two_chunks
    )


def test_docling_parser_rejects_missing_file(
    tmp_path,
):
    import pytest

    parser = DoclingDocumentParser(
        converter=FakeConverter()
    )

    missing = (
        tmp_path / "missing.pdf"
    )

    with pytest.raises(
        FileNotFoundError
    ):
        parser.parse(
            missing,
            language="fr",
        )


def test_oversized_paragraph_does_not_split_inside_word() -> None:
    parser = DoclingDocumentParser(
        converter=object(),
        max_chunk_chars=100,
    )

    # 95 chars + one space means "urgence"
    # starts at position 96.
    #
    # A hard cut at character 100 produces:
    #
    #   ... "urge" | "nce ..."
    #
    # which deliberately forces a mid-word boundary.
    text = (
        ("A" * 95)
        + " urgence de niveau 3, "
        + "le montant est de 750 000 dollars. "
        + ("B" * 100)
    )

    chunks = parser._split_page(text)

    assert len(chunks) >= 2

    for left, right in zip(
        chunks,
        chunks[1:],
    ):
        if left and right and left[-1].isalnum() and right[0].isalnum():
            last_word_of_left = left.split()[-1]
            first_word_of_right = right.split()[0]

            assert (
                f"{last_word_of_left} {first_word_of_right}" in text
                or f"{last_word_of_left}\n{first_word_of_right}" in text
            ), (
                "Chunk boundary split a word: "
                f"{left[-20:]!r} | {right[:20]!r}"
            )


def test_oversized_paragraph_prefers_sentence_boundary() -> None:
    parser = DoclingDocumentParser(
        converter=object(),
        max_chunk_chars=100,
    )

    first_sentence = (
        ("alpha " * 10)
        + "fin."
    )

    second_sentence = (
        "Deuxieme phrase "
        + ("beta " * 25)
    )

    text = (
        first_sentence
        + " "
        + second_sentence
    )

    chunks = parser._split_page(text)

    assert len(chunks) >= 2

    assert chunks[0] == first_sentence

    assert chunks[1].startswith(
        "Deuxieme phrase"
    )
