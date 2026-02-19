import hashlib


def compute_content_hash(question_norm: str, category_norm: str) -> str:
    payload = f"{question_norm}|{category_norm}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
