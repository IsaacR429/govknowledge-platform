from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from app.application.ports.parser_execution import (
    ParserExecutionResult,
    ParserExecutor,
)
from app.domain.enums.processing import ParsingStatus


FallbackPredicate = Callable[
    [Path, ParserExecutionResult],
    bool,
]


class RoutedParserExecutor(ParserExecutor):
    """
    Execute one primary parser and at most one fallback.

    Fallback is attempted when:
    - the primary parser fails, or
    - an explicit fallback predicate requests it.

    The predicate allows later OCR routing without coupling this
    executor to a particular PDF library.
    """

    def __init__(
        self,
        *,
        primary: ParserExecutor,
        fallback: ParserExecutor,
        primary_name: str,
        fallback_name: str,
        fallback_predicate: FallbackPredicate | None = None,
    ) -> None:
        if not primary_name.strip():
            raise ValueError(
                "primary_name must not be empty."
            )

        if not fallback_name.strip():
            raise ValueError(
                "fallback_name must not be empty."
            )

        self._primary = primary
        self._fallback = fallback
        self._primary_name = primary_name
        self._fallback_name = fallback_name
        self._fallback_predicate = fallback_predicate

    @staticmethod
    def _failure_value(
        result: ParserExecutionResult,
    ) -> str | None:
        if result.failure_class is None:
            return None

        return result.failure_class.value

    def _with_routing_lineage(
        self,
        *,
        result: ParserExecutionResult,
        primary_result: ParserExecutionResult,
        fallback_result: ParserExecutionResult | None,
        fallback_attempted: bool,
        fallback_reason: str | None,
        selected_parser: str | None,
    ) -> ParserExecutionResult:
        lineage = dict(result.lineage)

        lineage["parser_routing"] = {
            "strategy": "PRIMARY_THEN_SINGLE_FALLBACK",
            "primary_parser": self._primary_name,
            "fallback_parser": self._fallback_name,
            "fallback_attempted": fallback_attempted,
            "fallback_reason": fallback_reason,
            "selected_parser": selected_parser,
            "primary_status": (
                primary_result.parsing_status.value
            ),
            "primary_failure_class": (
                self._failure_value(
                    primary_result
                )
            ),
            "primary_lineage": dict(
                primary_result.lineage
            ),
            "fallback_status": (
                fallback_result.parsing_status.value
                if fallback_result is not None
                else None
            ),
            "fallback_failure_class": (
                self._failure_value(
                    fallback_result
                )
                if fallback_result is not None
                else None
            ),
            "fallback_lineage": (
                dict(
                    fallback_result.lineage
                )
                if fallback_result is not None
                else None
            ),
        }

        return ParserExecutionResult(
            parsing_status=result.parsing_status,
            failure_class=result.failure_class,
            parsed_document=result.parsed_document,
            lineage=lineage,
        )

    def execute(
        self,
        file_path: Path,
        *,
        language: str,
    ) -> ParserExecutionResult:
        primary_result = self._primary.execute(
            file_path,
            language=language,
        )

        fallback_reason: str | None = None

        if (
            primary_result.parsing_status
            is ParsingStatus.FAILED
        ):
            fallback_reason = "PRIMARY_FAILED"

        elif (
            self._fallback_predicate is not None
            and self._fallback_predicate(
                file_path,
                primary_result,
            )
        ):
            fallback_reason = "POLICY_TRIGGERED"

        if fallback_reason is None:
            return self._with_routing_lineage(
                result=primary_result,
                primary_result=primary_result,
                fallback_result=None,
                fallback_attempted=False,
                fallback_reason=None,
                selected_parser=self._primary_name,
            )

        fallback_result = self._fallback.execute(
            file_path,
            language=language,
        )

        selected_parser = (
            self._fallback_name
            if (
                fallback_result.parsing_status
                is ParsingStatus.SUCCESS
            )
            else None
        )

        return self._with_routing_lineage(
            result=fallback_result,
            primary_result=primary_result,
            fallback_result=fallback_result,
            fallback_attempted=True,
            fallback_reason=fallback_reason,
            selected_parser=selected_parser,
        )