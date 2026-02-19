import argparse
import json
from pathlib import Path

from app.db.repository import upsert_kb_item
from app.db.session import SessionLocal
from app.services.normalizer import normalize_category, normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest FAQ KB items into Postgres.")
    parser.add_argument("--kb-path", type=str, default="data/kb_items.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kb_path = Path(args.kb_path)
    if not kb_path.exists():
        raise FileNotFoundError(f"KB file not found: {kb_path}")

    payload = json.loads(kb_path.read_text(encoding="utf-8"))
    items = payload.get("knowledge_base_items", [])
    if not isinstance(items, list):
        raise ValueError("Invalid KB format. Expected knowledge_base_items list.")

    with SessionLocal() as db:
        for item in items:
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            category = normalize_category(str(item.get("category", "unknown")))
            if not question or not answer:
                continue

            upsert_kb_item(
                db,
                question=question,
                answer=answer,
                category=category,
                question_norm=normalize_text(question),
                answer_norm=normalize_text(answer),
                raw_json=item,
            )
        db.commit()

    print(f"Ingested/updated {len(items)} KB records from {kb_path}.")


if __name__ == "__main__":
    main()
