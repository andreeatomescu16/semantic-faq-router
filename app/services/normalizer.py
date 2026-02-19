import re
import unicodedata

WHITESPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^a-z0-9\s@\-_./]")

CATEGORY_ALIAS: dict[str, str] = {
    "settings": "profile",
    "data_recovery": "troubleshooting",
    "account_lifecycle": "account",
    "security_incident": "security",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    clean = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    clean = clean.lower().strip()
    clean = NON_WORD_RE.sub(" ", clean)
    clean = WHITESPACE_RE.sub(" ", clean)
    return clean


def normalize_category(category: str) -> str:
    key = normalize_text(category).replace(" ", "_")
    return CATEGORY_ALIAS.get(key, key)
