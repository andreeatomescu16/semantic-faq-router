from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.repository import RetrievalResult, get_top_k_matches
from app.services.embeddings import EmbeddingClient
from app.services.normalizer import normalize_text


@dataclass(frozen=True)
class RetrievalDecision:
    query_norm: str
    candidates: list[RetrievalResult]


class RetrievalService:
    def __init__(
        self,
        embedder: EmbeddingClient,
        model_name: str,
        embedding_version: str,
        top_k: int,
    ) -> None:
        self.embedder = embedder
        self.model_name = model_name
        self.embedding_version = embedding_version
        self.top_k = top_k

    def retrieve_candidates(self, db: Session, user_question: str) -> RetrievalDecision:
        query_norm = normalize_text(user_question)
        query_embedding = self.embedder.embed_query(query_norm)
        candidates = get_top_k_matches(
            db,
            query_embedding=query_embedding,
            model_name=self.model_name,
            embedding_version=self.embedding_version,
            top_k=self.top_k,
        )
        return RetrievalDecision(query_norm=query_norm, candidates=candidates)

    def retrieve_candidates_from_embedding(
        self,
        db: Session,
        *,
        query_norm: str,
        query_embedding: list[float],
    ) -> RetrievalDecision:
        candidates = get_top_k_matches(
            db,
            query_embedding=query_embedding,
            model_name=self.model_name,
            embedding_version=self.embedding_version,
            top_k=self.top_k,
        )
        return RetrievalDecision(query_norm=query_norm, candidates=candidates)

    def find_best_match(self, db: Session, user_question: str) -> RetrievalDecision:
        return self.retrieve_candidates(db=db, user_question=user_question)
