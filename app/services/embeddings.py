from collections.abc import Sequence
from typing import Protocol

from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings


class EmbeddingClient(Protocol):
    def embed_query(self, text: str) -> list[float]:
        """Return an embedding vector for a query."""


class OpenAIEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for embeddings.")
        self.model_name = settings.openai_embedding_model
        self._client = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            request_timeout=settings.openai_timeout_seconds,
        )

    def embed_query(self, text: str) -> list[float]:
        embedding: Sequence[float] = self._client.embed_query(text)
        return list(map(float, embedding))
