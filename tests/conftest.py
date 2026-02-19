from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from app.api import deps
from app.main import app
from app.services.qa_service import QAService
from app.services.retrieval import RetrievalDecision
from app.services.router import DomainRouter
from app.services.scoring import CosineThresholdStrategy


class FakeRetrievalResult:
    def __init__(
        self,
        score: float,
        question: str,
        answer: str,
        category: str,
        question_norm: str | None = None,
    ) -> None:
        self.score = score
        self.question = question
        self.answer = answer
        self.category = category
        self.question_norm = question_norm


class FakeRetrievalService:
    def retrieve_candidates(self, _db, user_question: str) -> RetrievalDecision:
        lowered = user_question.lower()
        if "profile" in lowered:
            result = FakeRetrievalResult(
                score=0.95,
                question="How do I change my profile information?",
                answer="Navigate to your profile page, click on 'Edit Profile', and make the desired changes.",
                category="profile",
            )
            return RetrievalDecision(candidates=[result], query_norm=user_question)
        if "password" in lowered:
            first = FakeRetrievalResult(
                score=0.45,
                question="What steps do I take to reset my password?",
                answer="Go to account settings and reset password.",
                category="security",
            )
            second = FakeRetrievalResult(
                score=0.44,
                question="Are there any guidelines on setting a strong password?",
                answer="Use a combination of letters and symbols.",
                category="security",
            )
            return RetrievalDecision(candidates=[first, second], query_norm=user_question)
        return RetrievalDecision(candidates=[], query_norm=user_question)


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, user_question: str) -> str:
        self.calls += 1
        return f"mocked-openai-answer:{user_question}"


@pytest.fixture
def fake_llm_client() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def client(fake_llm_client: FakeLLMClient) -> Generator[TestClient, None, None]:
    qa_service = QAService(
        router=DomainRouter(),
        retrieval=FakeRetrievalService(),  # type: ignore[arg-type]
        scoring_strategy=CosineThresholdStrategy(),
        llm_client=fake_llm_client,
        threshold=0.82,
    )

    app.dependency_overrides[deps.get_qa_service] = lambda: qa_service
    app.dependency_overrides[deps.get_db] = lambda: object()
    app.dependency_overrides[deps.get_token] = lambda: None

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
