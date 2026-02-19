import logging
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.services.guardrails import COMPLIANCE_MESSAGE, detect_prompt_injection
from app.services.llm_client import ChatClient
from app.services.retrieval import RetrievalService
from app.services.router import DomainRouter
from app.services.scoring import ScoringStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QAResult:
    source: str
    matched_question: str
    category: str
    confidence: float | None
    answer: str


class QAService:
    def __init__(
        self,
        *,
        router: DomainRouter,
        retrieval: RetrievalService,
        scoring_strategy: ScoringStrategy,
        llm_client: ChatClient,
        threshold: float,
    ) -> None:
        self.router = router
        self.retrieval = retrieval
        self.scoring_strategy = scoring_strategy
        self.llm_client = llm_client
        self.threshold = threshold

    def ask(
        self,
        db: Session,
        user_question: str,
        trace_id: str,
        *,
        dry_run_openai: bool = False,
    ) -> QAResult:
        injection = detect_prompt_injection(user_question)
        if injection.is_injection:
            logger.warning(
                "Prompt injection blocked",
                extra={"trace_id": trace_id, "reason": injection.reason},
            )
            return QAResult(
                source="compliance",
                matched_question="N/A",
                category="N/A",
                confidence=None,
                answer=COMPLIANCE_MESSAGE,
            )

        domain = self.router.route_domain(user_question)
        if not domain.in_domain:
            return QAResult(
                source="compliance",
                matched_question="N/A",
                category="N/A",
                confidence=None,
                answer=COMPLIANCE_MESSAGE,
            )

        retrieval = self.retrieval.retrieve_candidates(db, user_question)
        selection = self.scoring_strategy.score_candidates(
            query_norm=retrieval.query_norm,
            candidates=retrieval.candidates,
            predicted_category=domain.category,
        )
        logger.info(
            "Scoring strategy selection",
            extra={
                "trace_id": trace_id,
                "strategy": self.scoring_strategy.name,
                "final_confidence": selection.final_confidence,
            },
        )

        if self.scoring_strategy.accepts_local(selection, self.threshold) and selection.best:
            return QAResult(
                source="local",
                matched_question=selection.best.question,
                category=selection.best.category,
                confidence=selection.final_confidence,
                answer=selection.best.answer,
            )

        if dry_run_openai:
            return QAResult(
                source="openai",
                matched_question="N/A",
                category=domain.category,
                confidence=selection.final_confidence,
                answer="[dry_run_openai] OpenAI fallback would be used.",
            )

        try:
            fallback = self.llm_client.answer(user_question)
            return QAResult(
                source="openai",
                matched_question="N/A",
                category=domain.category,
                confidence=selection.final_confidence,
                answer=fallback,
            )
        except Exception:
            logger.exception("OpenAI fallback failed", extra={"trace_id": trace_id})
            return QAResult(
                source="compliance",
                matched_question="N/A",
                category="N/A",
                confidence=None,
                answer="I cannot answer that right now. Please try again later.",
            )
