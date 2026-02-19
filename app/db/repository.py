from dataclasses import dataclass

from sqlalchemy import Select, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import KBEmbedding, KBItem


@dataclass(frozen=True)
class RetrievalResult:
    kb_item_id: int
    question: str
    answer: str
    category: str
    score: float
    question_norm: str | None = None


def upsert_kb_item(
    db: Session,
    *,
    question: str,
    answer: str,
    category: str,
    question_norm: str,
    answer_norm: str,
    raw_json: dict,
) -> None:
    stmt = insert(KBItem).values(
        question=question,
        answer=answer,
        category=category,
        question_norm=question_norm,
        answer_norm=answer_norm,
        raw_json=raw_json,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_qnorm_category",
        set_={
            "question": question,
            "answer": answer,
            "answer_norm": answer_norm,
            "raw_json": raw_json,
        },
    )
    db.execute(stmt)


def get_all_items(db: Session) -> list[KBItem]:
    stmt: Select[tuple[KBItem]] = select(KBItem).order_by(KBItem.id.asc())
    return list(db.scalars(stmt).all())


def get_existing_hashes(
    db: Session,
    *,
    model_name: str,
    embedding_version: str,
) -> set[str]:
    stmt = select(KBEmbedding.content_hash).where(
        KBEmbedding.model_name == model_name,
        KBEmbedding.embedding_version == embedding_version,
    )
    return set(db.scalars(stmt).all())


def insert_embedding(
    db: Session,
    *,
    kb_item_id: int,
    embedding: list[float],
    model_name: str,
    embedding_version: str,
    content_hash: str,
) -> None:
    stmt = insert(KBEmbedding).values(
        kb_item_id=kb_item_id,
        embedding=embedding,
        model_name=model_name,
        embedding_version=embedding_version,
        content_hash=content_hash,
    )
    stmt = stmt.on_conflict_do_nothing(constraint="uq_hash_model_version")
    db.execute(stmt)


def get_top_k_matches(
    db: Session,
    *,
    query_embedding: list[float],
    model_name: str,
    embedding_version: str,
    top_k: int,
) -> list[RetrievalResult]:
    distance_expr = KBEmbedding.embedding.cosine_distance(query_embedding)
    stmt = (
        select(
            KBItem.id,
            KBItem.question,
            KBItem.answer,
            KBItem.category,
            KBItem.question_norm,
            distance_expr.label("distance"),
        )
        .join(KBEmbedding, KBEmbedding.kb_item_id == KBItem.id)
        .where(
            KBEmbedding.model_name == model_name,
            KBEmbedding.embedding_version == embedding_version,
        )
        .order_by(distance_expr.asc())
        .limit(top_k)
    )
    rows = db.execute(stmt).all()
    if not rows:
        return []

    results: list[RetrievalResult] = []
    for row in rows:
        similarity_score = max(0.0, 1.0 - float(row.distance))
        results.append(
            RetrievalResult(
                kb_item_id=row.id,
                question=row.question,
                answer=row.answer,
                category=row.category,
                score=round(similarity_score, 4),
                question_norm=row.question_norm,
            )
        )
    return results


def get_best_match(
    db: Session,
    *,
    query_embedding: list[float],
    model_name: str,
    embedding_version: str,
    top_k: int,
) -> RetrievalResult | None:
    results = get_top_k_matches(
        db,
        query_embedding=query_embedding,
        model_name=model_name,
        embedding_version=embedding_version,
        top_k=top_k,
    )
    return results[0] if results else None
