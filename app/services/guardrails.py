import re
from dataclasses import dataclass

from app.services.normalizer import normalize_text

COMPLIANCE_MESSAGE = (
    "This is not really what I was trained for, therefore I cannot answer. Try again."
)

TIER1_INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"reveal\s+system\s+prompt",
    r"show\s+me\s+your\s+system\s+prompt",
    r"bypass\s+guardrails",
    r"developer\s+message",
    r"jailbreak",
    r"prompt\s+injection",
]

EXFIL_VERB_RE = re.compile(
    r"\b(reveal|show|give|print|dump|leak|expose|share|tell\s+me)\b",
    flags=re.IGNORECASE,
)
EXFIL_TARGET_RE = re.compile(
    r"\b(api[\s_-]*key|secret[\s_-]*key|system\s+prompt|credentials?|token)\b",
    flags=re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(r"[.!?;\n]+")

COMPILED_TIER1_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE) for pattern in TIER1_INJECTION_PATTERNS
]


@dataclass(frozen=True)
class InjectionCheckResult:
    is_injection: bool
    reason: str


def detect_prompt_injection(user_question: str) -> InjectionCheckResult:
    normalized = normalize_text(user_question)
    for pattern in COMPILED_TIER1_PATTERNS:
        if pattern.search(normalized):
            return InjectionCheckResult(
                is_injection=True,
                reason=f"tier1_pattern:{pattern.pattern}",
            )

    for sentence in SENTENCE_SPLIT_RE.split(normalized):
        sentence = sentence.strip()
        if not sentence:
            continue
        if EXFIL_VERB_RE.search(sentence) and EXFIL_TARGET_RE.search(sentence):
            return InjectionCheckResult(
                is_injection=True,
                reason="tier2_exfil_intent",
            )
    return InjectionCheckResult(is_injection=False, reason="none")
