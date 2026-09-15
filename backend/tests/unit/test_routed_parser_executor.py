from pathlib import Path
from typing import cast

from app.application.ports.document_parser import (
    ParsedDocument,
)
from app.application.ports.parser_execution import (
    ParserExecutionResult,
)
from app.domain.enums.processing import (
    ParserFailureClass,
    ParsingStatus,
)
from app.infrastructure.parsing.routed_parser_executor import (
    RoutedParserExecutor,
)


class StubParserExecutor:
    def __init__(
        self,
        result: ParserExecutionResult,
    ) -> None:
        self.result = result
        self.calls = 0

    def execute(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParserExecutionResult:
        self.calls += 1
        return self.result


def successful_result(
    parser: str,
) -> ParserExecutionResult:
    return ParserExecutionResult(
        parsing_status=ParsingStatus.SUCCESS,
        failure_class=None,
        parsed_document=cast(
            ParsedDocument,
            object(),
        ),
        lineage={
            "parser_family": parser,
        },
    )


def failed_result(
    parser: str,
    failure: ParserFailureClass,
) -> ParserExecutionResult:
    return ParserExecutionResult(
        parsing_status=ParsingStatus.FAILED,
        failure_class=failure,
        parsed_document=None,
        lineage={
            "parser_family": parser,
        },
    )


def test_primary_success_does_not_call_fallback() -> None:
    primary = StubParserExecutor(
        successful_result("primary")
    )
    fallback = StubParserExecutor(
        successful_result("fallback")
    )

    executor = RoutedParserExecutor(
        primary=primary,
        fallback=fallback,
        primary_name="pymupdf4llm",
        fallback_name="docling-modern",
    )

    result = executor.execute(
        Path("document.pdf"),
        language="en",
    )

    assert result.parsing_status is ParsingStatus.SUCCESS
    assert primary.calls == 1
    assert fallback.calls == 0

    routing = result.lineage["parser_routing"]

    assert routing["fallback_attempted"] is False
    assert routing["fallback_reason"] is None
    assert routing["selected_parser"] == "pymupdf4llm"


def test_primary_failure_calls_fallback_once() -> None:
    primary = StubParserExecutor(
        failed_result(
            "primary",
            ParserFailureClass.TIMEOUT,
        )
    )
    fallback = StubParserExecutor(
        successful_result("fallback")
    )

    executor = RoutedParserExecutor(
        primary=primary,
        fallback=fallback,
        primary_name="pymupdf4llm",
        fallback_name="docling-modern",
    )

    result = executor.execute(
        Path("document.pdf"),
        language="en",
    )

    assert result.parsing_status is ParsingStatus.SUCCESS
    assert primary.calls == 1
    assert fallback.calls == 1

    routing = result.lineage["parser_routing"]

    assert routing["fallback_attempted"] is True
    assert routing["fallback_reason"] == "PRIMARY_FAILED"
    assert routing["selected_parser"] == "docling-modern"
    assert routing["primary_failure_class"] == "TIMEOUT"


def test_policy_can_request_fallback_after_primary_success() -> None:
    primary = StubParserExecutor(
        successful_result("primary")
    )
    fallback = StubParserExecutor(
        successful_result("fallback")
    )

    predicate_calls = 0

    def requires_fallback(
        file_path: Path,
        result: ParserExecutionResult,
    ) -> bool:
        nonlocal predicate_calls
        predicate_calls += 1

        assert file_path == Path(
            "document.pdf"
        )
        assert (
            result.parsing_status
            is ParsingStatus.SUCCESS
        )

        return True

    executor = RoutedParserExecutor(
        primary=primary,
        fallback=fallback,
        primary_name="pymupdf4llm",
        fallback_name="docling-modern",
        fallback_predicate=requires_fallback,
    )

    result = executor.execute(
        Path("document.pdf"),
        language="en",
    )

    assert result.parsing_status is ParsingStatus.SUCCESS
    assert primary.calls == 1
    assert fallback.calls == 1
    assert predicate_calls == 1

    routing = result.lineage["parser_routing"]

    assert routing["fallback_attempted"] is True
    assert routing["fallback_reason"] == "POLICY_TRIGGERED"
    assert routing["selected_parser"] == "docling-modern"


def test_both_fail_without_recursive_retry() -> None:
    primary = StubParserExecutor(
        failed_result(
            "primary",
            ParserFailureClass.TIMEOUT,
        )
    )
    fallback = StubParserExecutor(
        failed_result(
            "fallback",
            ParserFailureClass.NATIVE_CRASH,
        )
    )

    executor = RoutedParserExecutor(
        primary=primary,
        fallback=fallback,
        primary_name="pymupdf4llm",
        fallback_name="docling-modern",
    )

    result = executor.execute(
        Path("document.pdf"),
        language="en",
    )

    assert result.parsing_status is ParsingStatus.FAILED
    assert (
        result.failure_class
        is ParserFailureClass.NATIVE_CRASH
    )

    assert primary.calls == 1
    assert fallback.calls == 1

    routing = result.lineage["parser_routing"]

    assert routing["fallback_attempted"] is True
    assert routing["fallback_reason"] == "PRIMARY_FAILED"
    assert routing["selected_parser"] is None
    assert routing["primary_failure_class"] == "TIMEOUT"
    assert routing["fallback_failure_class"] == "NATIVE_CRASH"