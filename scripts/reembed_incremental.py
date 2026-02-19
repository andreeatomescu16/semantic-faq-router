from app.core.config import get_settings
from app.db.repository import get_all_items, get_existing_hashes, insert_embedding
from app.db.session import SessionLocal
from app.services.embeddings import OpenAIEmbeddingClient
from app.services.hashing import compute_content_hash


def main() -> None:
    settings = get_settings()
    embedder = OpenAIEmbeddingClient(settings=settings)

    with SessionLocal() as db:
        existing_hashes = get_existing_hashes(
            db,
            model_name=settings.openai_embedding_model,
            embedding_version=settings.embedding_version,
        )
        items = get_all_items(db)

        inserted = 0
        for item in items:
            content_hash = compute_content_hash(item.question_norm, item.category)
            if content_hash in existing_hashes:
                continue

            embedding = embedder.embed_query(item.question_norm)
            if len(embedding) != settings.embedding_dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {settings.embedding_dimensions}, got {len(embedding)}"
                )

            insert_embedding(
                db,
                kb_item_id=item.id,
                embedding=embedding,
                model_name=settings.openai_embedding_model,
                embedding_version=settings.embedding_version,
                content_hash=content_hash,
            )
            inserted += 1
        db.commit()

    print(f"Incremental embedding complete. Inserted {inserted} new vectors.")


if __name__ == "__main__":
    main()
