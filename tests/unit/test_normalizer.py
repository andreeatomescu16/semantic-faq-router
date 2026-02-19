from app.services.normalizer import normalize_category, normalize_text


def test_normalize_text_removes_noise_and_emoji() -> None:
    raw = "help!!! 😭😭😭 my account is locked   "
    assert normalize_text(raw) == "help my account is locked"


def test_normalize_category_applies_alias() -> None:
    assert normalize_category("security_incident") == "security"
    assert normalize_category("data recovery") == "troubleshooting"
