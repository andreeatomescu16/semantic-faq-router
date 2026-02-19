from app.services.hashing import compute_content_hash


def test_content_hash_is_deterministic() -> None:
    first = compute_content_hash("change login email", "profile")
    second = compute_content_hash("change login email", "profile")
    assert first == second
    assert len(first) == 64


def test_content_hash_changes_when_content_changes() -> None:
    first = compute_content_hash("change login email", "profile")
    second = compute_content_hash("change login email", "security")
    assert first != second
