import re
from dataclasses import dataclass

from app.services.normalizer import normalize_text

COMPLIANCE_MESSAGE = (
    "This is not really what I was trained for, therefore I cannot answer. Try again."
)

INJECTION_PATTERNS = [
    r"ignore\s+previous\s+instructions",
    r"reveal\s+system\s+prompt",
    r"show\s+me\s+your\s+system\s+prompt",
    r"api[\s_-]*key",
    r"bypass\s+guardrails",
    r"developer\s+message",
    r"jailbreak",
]

COMPILED_INJECTION_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE) for pattern in INJECTION_PATTERNS
]


@dataclass(frozen=True)
class InjectionCheckResult:
    is_injection: bool
    reason: str


def detect_prompt_injection(user_question: str) -> InjectionCheckResult:
    normalized = normalize_text(user_question)
    for pattern in COMPILED_INJECTION_PATTERNS:
        if pattern.search(normalized):
            return InjectionCheckResult(
                is_injection=True,
                reason=f"matched_pattern:{pattern.pattern}",
            )
    return InjectionCheckResult(is_injection=False, reason="none")
