from dataclasses import dataclass
from typing import Protocol

from app.db.repository import RetrievalResult
from app.services.normalizer import normalize_text


@dataclass(frozen=True)
class ScoredSelection:
    best: RetrievalResult | None
    best_score: float | None
    second_score: float | None
    final_confidence: float | None
    debug: dict[str, object]


class ScoringStrategy(Protocol):
    name: str

    def score_candidates(
        self,
        query_norm: str,
        candidates: list[RetrievalResult],
        predicted_category: str | None = None,
    ) -> ScoredSelection:
        """Score candidates and return best selection details."""

    def accepts_local(self, selection: ScoredSelection, threshold: float) -> bool:
        """Determine local routing acceptance."""


def _empty_selection() -> ScoredSelection:
    return ScoredSelection(
        best=None,
        best_score=None,
        second_score=None,
        final_confidence=None,
        debug={},
    )


def _tokenize(text: str) -> set[str]:
    normalized = normalize_text(text)
    return {token for token in normalized.split(" ") if token}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))


LEXICAL_BOOST_TOKENS: set[str] = {"reset", "password", "email", "invoice", "refund"}


def _weighted_overlap(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    common = query_tokens.intersection(candidate_tokens)
    base = len(common) / len(query_tokens)
    keyword_boost = 0.0
    for token in LEXICAL_BOOST_TOKENS:
        if token in common:
            keyword_boost += 0.12
    return min(1.0, base + keyword_boost)


class CosineThresholdStrategy:
    name = "cosine_threshold"

    def score_candidates(
        self,
        query_norm: str,
        candidates: list[RetrievalResult],
        predicted_category: str | None = None,
    ) -> ScoredSelection:
        del query_norm, predicted_category
        if not candidates:
            return _empty_selection()
        best = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else None
        return ScoredSelection(
            best=best,
            best_score=best.score,
            second_score=second_score,
            final_confidence=best.score,
            debug={
                "strategy": self.name,
                "ranked_questions": [candidate.question for candidate in candidates],
                "ranked_scores": [candidate.score for candidate in candidates],
            },
        )

    def accepts_local(self, selection: ScoredSelection, threshold: float) -> bool:
        return bool(
            selection.best is not None
            and selection.final_confidence is not None
            and selection.final_confidence >= threshold
        )


class CosineMarginStrategy(CosineThresholdStrategy):
    name = "cosine_margin"

    def __init__(self, margin: float) -> None:
        self.margin = margin

    def accepts_local(self, selection: ScoredSelection, threshold: float) -> bool:
        if not super().accepts_local(selection, threshold):
            return False
        second = selection.second_score if selection.second_score is not None else 0.0
        return (selection.final_confidence - second) >= self.margin  # type: ignore[operator]


class HybridRerankStrategy:
    name = "hybrid"

    def __init__(self, alpha: float, beta: float, gamma: float) -> None:
        weight_sum = alpha + beta + gamma
        if weight_sum <= 0:
            self.alpha, self.beta, self.gamma = 0.7, 0.2, 0.1
        else:
            self.alpha = alpha / weight_sum
            self.beta = beta / weight_sum
            self.gamma = gamma / weight_sum

    def score_candidates(
        self,
        query_norm: str,
        candidates: list[RetrievalResult],
        predicted_category: str | None = None,
    ) -> ScoredSelection:
        if not candidates:
            return _empty_selection()

        query_tokens = _tokenize(query_norm)
        scored: list[tuple[RetrievalResult, float, float, float]] = []
        for candidate in candidates:
            candidate_tokens = _tokenize(candidate.question_norm or candidate.question)
            lexical = _weighted_overlap(query_tokens, candidate_tokens)
            category_boost = 0.0
            if predicted_category and predicted_category != "N/A":
                category_boost = 1.0 if candidate.category == predicted_category else -0.35
            final_score = (
                self.alpha * candidate.score
                + self.beta * lexical
                + self.gamma * category_boost
            )
            final_score = min(1.0, max(0.0, final_score))
            scored.append((candidate, final_score, lexical, category_boost))

        scored.sort(key=lambda item: item[1], reverse=True)
        best, best_score, best_lexical, best_boost = scored[0]
        second_score = scored[1][1] if len(scored) > 1 else None
        return ScoredSelection(
            best=best,
            best_score=best.score,
            second_score=second_score,
            final_confidence=best_score,
            debug={
                "strategy": self.name,
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "best_lexical": round(best_lexical, 4),
                "best_category_boost": best_boost,
                "lexical_mode": "weighted_overlap",
                "ranked_questions": [item[0].question for item in scored],
                "ranked_scores": [round(item[1], 4) for item in scored],
            },
        )

    def accepts_local(self, selection: ScoredSelection, threshold: float) -> bool:
        return bool(
            selection.best is not None
            and selection.final_confidence is not None
            and selection.final_confidence >= threshold
        )


class HybridMarginStrategy(HybridRerankStrategy):
    name = "hybrid_margin"

    def __init__(self, alpha: float, beta: float, gamma: float, margin: float) -> None:
        super().__init__(alpha=alpha, beta=beta, gamma=gamma)
        self.margin = margin

    def accepts_local(self, selection: ScoredSelection, threshold: float) -> bool:
        if not super().accepts_local(selection, threshold):
            return False
        second = selection.second_score if selection.second_score is not None else 0.0
        return (selection.final_confidence - second) >= self.margin  # type: ignore[operator]


def build_scoring_strategy(
    strategy_name: str,
    *,
    margin: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> ScoringStrategy:
    key = strategy_name.strip().lower()
    if key == "cosine_margin":
        return CosineMarginStrategy(margin=margin)
    if key == "hybrid":
        return HybridRerankStrategy(alpha=alpha, beta=beta, gamma=gamma)
    if key == "hybrid_margin":
        return HybridMarginStrategy(alpha=alpha, beta=beta, gamma=gamma, margin=margin)
    return CosineThresholdStrategy()
