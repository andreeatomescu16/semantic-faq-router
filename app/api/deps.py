from collections.abc import Generator

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.db.session import SessionLocal
from app.services.embeddings import OpenAIEmbeddingClient
from app.services.llm_client import OpenAIChatClient
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalService
from app.services.router import DomainRouter
from app.services.scoring import build_scoring_strategy
import secrets


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_token(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.require_auth:
        return
    if not settings.api_auth_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_AUTH_TOKEN is not configured.",
        )
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token.",
        )
    token = authorization.split(" ", maxsplit=1)[1].strip()
    if not secrets.compare_digest(token, settings.api_auth_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        )


def get_qa_service(settings: Settings = Depends(get_settings)) -> QAService:
    embedder = OpenAIEmbeddingClient(settings=settings)
    llm_client = OpenAIChatClient(settings=settings)
    retrieval = RetrievalService(
        embedder=embedder,
        model_name=settings.openai_embedding_model,
        embedding_version=settings.embedding_version,
        top_k=settings.top_k,
    )
    return QAService(
        router=DomainRouter(),
        retrieval=retrieval,
        scoring_strategy=build_scoring_strategy(
            settings.scoring_strategy,
            margin=settings.similarity_margin,
            alpha=settings.hybrid_alpha,
            beta=settings.hybrid_beta,
            gamma=settings.hybrid_gamma,
        ),
        llm_client=llm_client,
        threshold=settings.similarity_threshold,
    )


def get_trace_id(request: Request) -> str:
    return getattr(request.state, "trace_id", "N/A")
