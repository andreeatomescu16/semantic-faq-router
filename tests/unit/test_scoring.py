from app.db.repository import RetrievalResult
from app.services.scoring import (
    CosineMarginStrategy,
    CosineThresholdStrategy,
    HybridRerankStrategy,
)


def _candidate(
    question: str,
    score: float,
    category: str = "profile",
    question_norm: str | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        kb_item_id=1,
        question=question,
        answer="answer",
        category=category,
        score=score,
        question_norm=question_norm,
    )


def test_cosine_threshold_vs_margin_acceptance_logic() -> None:
    candidates = [
        _candidate("best", 0.84),
        _candidate("second", 0.81),
    ]
    threshold = CosineThresholdStrategy()
    margin = CosineMarginStrategy(margin=0.05)

    base_selection = threshold.score_candidates("query", candidates)
    margin_selection = margin.score_candidates("query", candidates)

    assert threshold.accepts_local(base_selection, threshold=0.82) is True
    assert margin.accepts_local(margin_selection, threshold=0.82) is False


def test_hybrid_rerank_changes_order_with_lexical_overlap() -> None:
    candidates = [
        _candidate(
            question="password reset flow steps",
            score=0.80,
            category="security",
            question_norm="password reset flow steps",
        ),
        _candidate(
            question="profile avatar upload",
            score=0.92,
            category="profile",
            question_norm="profile avatar upload",
        ),
    ]
    hybrid = HybridRerankStrategy(alpha=0.3, beta=0.7, gamma=0.0)
    selection = hybrid.score_candidates("password reset email", candidates, predicted_category="security")
    assert selection.best is not None
    assert selection.best.question == "password reset flow steps"


def test_hybrid_category_boost_and_score_cap() -> None:
    candidates = [
        _candidate(
            question="billing invoice download",
            score=0.98,
            category="billing",
            question_norm="billing invoice download",
        )
    ]
    hybrid = HybridRerankStrategy(alpha=0.9, beta=0.05, gamma=0.5)
    selection = hybrid.score_candidates(
        "billing invoice",
        candidates,
        predicted_category="billing",
    )
    assert selection.final_confidence is not None
    assert 0.0 <= selection.final_confidence <= 1.0


def test_hybrid_rerank_prefers_reset_email_candidate_for_q020_like_query() -> None:
    candidates = [
        _candidate(
            question="Pause email notifications",
            score=0.84,
            category="notifications",
            question_norm="pause email notifications",
        ),
        _candidate(
            question="I didn't get the password reset email",
            score=0.78,
            category="security",
            question_norm="i didnt get the password reset email",
        ),
    ]
    hybrid = HybridRerankStrategy(alpha=0.6, beta=0.3, gamma=0.1)
    selection = hybrid.score_candidates(
        "reset email not received after 10 mins",
        candidates,
        predicted_category="security",
    )
    assert selection.best is not None
    assert selection.best.question == "I didn't get the password reset email"
