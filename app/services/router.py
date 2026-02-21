from dataclasses import dataclass

from app.services.normalizer import normalize_text

IN_SCOPE_DOMAINS = (
    "account",
    "profile",
    "security",
    "billing",
    "subscription",
    "notifications",
    "privacy",
    "troubleshooting",
    "developer",
)

CRUCIAL_FUZZY_KEYWORDS: set[str] = {"password", "subscription", "renewal", "invoice", "refund"}


DOMAIN_TOKEN_PATTERNS: dict[str, set[str]] = {
    "account": {"account", "login", "locked", "deactivate", "delete"},
    "profile": {"profile", "display", "avatar", "photo", "email", "username"},
    "security": {
        "password",
        "pass",
        "pwd",
        "forgot",
        "reset",
        "recover",
        "2fa",
        "authenticator",
        "passkey",
        "phishing",
        "session",
        "compromised",
    },
    "billing": {"billing", "invoice", "payment", "refund", "card", "paypal", "receipt"},
    "subscription": {
        "plan",
        "subscription",
        "cancel",
        "renewal",
        "renew",
        "upgrade",
        "downgrade",
        "billing",
        "access",
    },
    "notifications": {"notification", "notifications", "email", "push", "alerts", "dnd"},
    "privacy": {"privacy", "export", "download", "delete", "gdpr", "personal", "data"},
    "troubleshooting": {"crash", "slow", "bug", "error", "help", "issue", "fix"},
    "developer": {"api", "key", "developer", "sdk", "token", "integration", "webhook"},
}


DOMAIN_PHRASE_PATTERNS: dict[str, list[set[str]]] = {
    "security": [
        {"forgot", "password"},
        {"reset", "password"},
        {"recover", "password"},
        {"two", "factor"},
    ],
    "privacy": [
        {"export", "data"},
        {"download", "data"},
        {"delete", "data"},
        {"get", "my", "data"},
    ],
    "subscription": [
        {"stop", "renewal"},
        {"stop", "billing"},
        {"cancel", "renewal"},
        {"keep", "access"},
    ],
    "developer": [
        {"api", "key"},
        {"developer", "token"},
    ],
}


DOMAIN_SYNONYM_GROUPS: dict[str, list[list[set[str]]]] = {
    "security": [
        [
            {"pass", "password", "pwd"},
            {"reset", "recover", "forgot"},
        ]
    ],
    "privacy": [
        [
            {"export", "download", "get"},
            {"data"},
        ]
    ],
    "subscription": [
        [
            {"stop", "cancel"},
            {"renewal", "billing"},
        ],
        [
            {"keep"},
            {"access"},
        ],
    ],
}


@dataclass(frozen=True)
class DomainDecision:
    in_domain: bool
    category: str
    reason: str


def _tokenize(text: str) -> set[str]:
    return {token for token in text.split(" ") if token}


def _is_edit_distance_leq_one(source: str, target: str) -> bool:
    if source == target:
        return True
    if abs(len(source) - len(target)) > 1:
        return False
    if len(source) > len(target):
        source, target = target, source

    i = 0
    j = 0
    edits = 0
    while i < len(source) and j < len(target):
        if source[i] == target[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(source) == len(target):
            i += 1
            j += 1
        else:
            j += 1
    if j < len(target) or i < len(source):
        edits += 1
    return edits <= 1


class DomainRouter:
    def route_domain(self, user_question: str) -> DomainDecision:
        normalized = normalize_text(user_question)
        if not normalized:
            return DomainDecision(in_domain=False, category="N/A", reason="empty_input")

        tokens = _tokenize(normalized)
        domain_scores: dict[str, int] = {domain: 0 for domain in IN_SCOPE_DOMAINS}
        domain_reasons: dict[str, list[str]] = {domain: [] for domain in IN_SCOPE_DOMAINS}

        for domain in IN_SCOPE_DOMAINS:
            for keyword in DOMAIN_TOKEN_PATTERNS.get(domain, set()):
                if keyword in tokens:
                    domain_scores[domain] += 1
                    domain_reasons[domain].append(f"token:{keyword}")
                    continue
                if keyword in CRUCIAL_FUZZY_KEYWORDS and any(
                    _is_edit_distance_leq_one(token, keyword) for token in tokens
                ):
                    domain_scores[domain] += 1
                    domain_reasons[domain].append(f"fuzzy:{keyword}")

            for phrase_tokens in DOMAIN_PHRASE_PATTERNS.get(domain, []):
                if phrase_tokens.issubset(tokens):
                    domain_scores[domain] += 2
                    domain_reasons[domain].append(
                        "phrase:" + "_".join(sorted(phrase_tokens))
                    )

            for synonym_groups in DOMAIN_SYNONYM_GROUPS.get(domain, []):
                if all(group.intersection(tokens) for group in synonym_groups):
                    domain_scores[domain] += 2
                    domain_reasons[domain].append("synonym_combo")

        ranked = sorted(
            domain_scores.items(),
            key=lambda item: (item[1], len(domain_reasons[item[0]])),
            reverse=True,
        )
        best_domain, best_score = ranked[0]
        if best_score >= 1:
            reason = ",".join(domain_reasons[best_domain][:4]) or "token_match"
            return DomainDecision(in_domain=True, category=best_domain, reason=reason)

        return DomainDecision(in_domain=False, category="N/A", reason="no_domain_match")
