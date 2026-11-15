from enum import StrEnum


class TableRowRole(StrEnum):
    """Semantic role assigned to a canonical table row."""

    HEADER = "HEADER"
    DATA = "DATA"
    SECTION = "SECTION"
    SUMMARY = "SUMMARY"


class TableContinuationDecision(StrEnum):
    """Decision about whether two physical tables form one logical table."""

    CONTINUE = "CONTINUE"
    STOP = "STOP"
    AMBIGUOUS = "AMBIGUOUS"


class TableSpanKind(StrEnum):
    """What a source span contributes to canonical evidence."""

    TEXT = "TEXT"
    STRUCTURE = "STRUCTURE"
    GEOMETRY = "GEOMETRY"


class TableCellValueSource(StrEnum):
    """How the canonical cell value was established."""

    SOURCE_TEXT = "SOURCE_TEXT"
    INHERITED = "INHERITED"
    DERIVED = "DERIVED"


class TableEvidenceLayer(StrEnum):
    """Nature of normalized parser evidence."""

    PHYSICAL = "PHYSICAL"
    STRUCTURAL = "STRUCTURAL"
    FUSED = "FUSED"


class TableColumnMappingKind(StrEnum):
    """Relationship between physical and logical columns."""

    EXACT = "EXACT"
    EQUAL_GROUPS = "EQUAL_GROUPS"


class TableCellTextStatus(StrEnum):
    """Whether cell text was actually observed.

    OBSERVED with text="" means a real observed blank.

    UNAVAILABLE means the adapter could not determine cell text.
    It must never be interpreted as a blank value.
    """

    OBSERVED = "OBSERVED"
    UNAVAILABLE = "UNAVAILABLE"


class TableRowAlignmentKind(StrEnum):
    ONE_TO_ONE = "ONE_TO_ONE"
    WRAPPED_CONTINUATION = "WRAPPED_CONTINUATION"
    STRUCTURAL_GROUP = "STRUCTURAL_GROUP"
