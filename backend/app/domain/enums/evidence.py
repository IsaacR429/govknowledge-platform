from enum import Enum


class EvidenceDecision(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    NO_EVIDENCE = "NO_EVIDENCE"
    TOP_SCORE_TOO_LOW = "TOP_SCORE_TOO_LOW"
    INSUFFICIENT_SUPPORT = "INSUFFICIENT_SUPPORT"


class EntailmentLabel(str, Enum):
    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"
