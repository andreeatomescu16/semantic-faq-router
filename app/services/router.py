from dataclasses import dataclass

from app.services.normalizer import normalize_text

IN_SCOPE_DOMAINS = {
    "account",
    "profile",
    "security",
    "billing",
    "subscription",
    "notifications",
    "privacy",
    "troubleshooting",
    "developer",
}

DOMAIN_KEYWORDS: dict[str, set[str]] = {
    "account": {"account", "login", "locked", "deactivate", "delete account"},
    "profile": {"profile", "display name", "avatar", "email", "edit profile"},
    "security": {
        "password",
        "2fa",
        "two factor",
        "authenticator",
        "passkey",
        "phishing",
        "session",
        "compromised",
    },
    "billing": {"billing", "invoice", "payment", "refund", "card", "paypal"},
    "subscription": {"plan", "subscription", "cancel", "upgrade", "downgrade"},
    "notifications": {"notification", "email notifications", "push", "alerts", "dnd"},
    "privacy": {"privacy", "export data", "delete data", "gdpr", "personal data"},
    "troubleshooting": {"crash", "slow", "bug", "error", "help", "issue"},
    "developer": {"api key", "developer", "sdk", "token", "integration"},
}


@dataclass(frozen=True)
class DomainDecision:
    in_domain: bool
    category: str
    reason: str


class DomainRouter:
    def route_domain(self, user_question: str) -> DomainDecision:
        normalized = normalize_text(user_question)
        if not normalized:
            return DomainDecision(in_domain=False, category="N/A", reason="empty_input")

        scores: dict[str, int] = {domain: 0 for domain in IN_SCOPE_DOMAINS}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                if keyword in normalized:
                    scores[domain] += 1

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]
        if best_score >= 1:
            return DomainDecision(in_domain=True, category=best_domain, reason="keyword_match")

        return DomainDecision(in_domain=False, category="N/A", reason="no_domain_match")
