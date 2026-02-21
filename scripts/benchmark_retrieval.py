import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.embeddings import OpenAIEmbeddingClient
from app.services.guardrails import detect_prompt_injection
from app.services.normalizer import normalize_text
from app.services.retrieval import RetrievalDecision, RetrievalService
from app.services.router import DomainRouter
from app.services.scoring import (
    CosineMarginStrategy,
    CosineThresholdStrategy,
    HybridMarginStrategy,
    HybridRerankStrategy,
    ScoringStrategy,
)

BENCHMARK_DATASET = Path("data/benchmark_queries.jsonl")
EMBEDDING_CACHE_PATH = Path("data/benchmark_query_embeddings_cache.json")
RESULTS_PATH = Path("benchmark_results.md")


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError(f"Embedding unexpectedly requested for query: {text}")


@dataclass(frozen=True)
class BenchmarkRow:
    benchmark_id: str
    query: str
    label_type: str
    expected_source: str
    expected_match: str | None


@dataclass
class StrategyMetrics:
    source_total: int = 0
    source_correct: int = 0
    expected_local_total: int = 0
    expected_not_local_total: int = 0
    false_positive_local: int = 0
    false_negative_local: int = 0
    positive_total: int = 0
    top1_hits: int = 0
    mrr_sum: float = 0.0

    def as_result(self) -> dict[str, float]:
        source_accuracy = self.source_correct / self.source_total if self.source_total else 0.0
        top1_accuracy = self.top1_hits / self.positive_total if self.positive_total else 0.0
        mrr = self.mrr_sum / self.positive_total if self.positive_total else 0.0
        fpr_local = (
            self.false_positive_local / self.expected_not_local_total
            if self.expected_not_local_total
            else 0.0
        )
        fnr_local = (
            self.false_negative_local / self.expected_local_total if self.expected_local_total else 0.0
        )
        return {
            "source_accuracy": source_accuracy,
            "top1_accuracy": top1_accuracy,
            "mrr": mrr,
            "fpr_local": fpr_local,
            "fnr_local": fnr_local,
        }


def _load_benchmark_rows(path: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                BenchmarkRow(
                    benchmark_id=str(payload["id"]),
                    query=str(payload["query"]),
                    label_type=str(payload["label_type"]),
                    expected_source=str(payload["expected_source"]),
                    expected_match=payload.get("expected_match"),
                )
            )
    return rows


def _load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_embedding_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _get_query_embedding(
    query_norm: str,
    cache: dict[str, list[float]],
    embedder: OpenAIEmbeddingClient | None,
) -> list[float]:
    if query_norm in cache:
        return cache[query_norm]
    if embedder is None:
        raise RuntimeError(
            "Missing cached embedding and OPENAI_API_KEY is not configured for benchmark embedding generation."
        )
    embedding = embedder.embed_query(query_norm)
    cache[query_norm] = embedding
    return embedding


def _ranked_questions(retrieval: RetrievalDecision, debug: dict[str, object]) -> list[str]:
    ranked = debug.get("ranked_questions")
    if isinstance(ranked, list) and all(isinstance(item, str) for item in ranked):
        return list(ranked)
    return [candidate.question for candidate in retrieval.candidates]


def _reciprocal_rank(expected_match: str, ranked_questions: list[str]) -> float:
    for index, question in enumerate(ranked_questions, start=1):
        if question == expected_match:
            return 1.0 / index
    return 0.0


def _format_markdown_table(rows: list[tuple[str, dict[str, float]]]) -> str:
    lines = [
        "| Strategy | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in rows:
        lines.append(
            "| {name} | {source:.3f} | {top1:.3f} | {mrr:.3f} | {fpr:.3f} | {fnr:.3f} |".format(
                name=name,
                source=result["source_accuracy"],
                top1=result["top1_accuracy"],
                mrr=result["mrr"],
                fpr=result["fpr_local"],
                fnr=result["fnr_local"],
            )
        )
    return "\n".join(lines)


def main() -> None:
    settings = get_settings()
    rows = _load_benchmark_rows(BENCHMARK_DATASET)

    strategies: list[ScoringStrategy] = [
        CosineThresholdStrategy(),
        CosineMarginStrategy(margin=settings.similarity_margin),
        HybridRerankStrategy(
            alpha=settings.hybrid_alpha,
            beta=settings.hybrid_beta,
            gamma=settings.hybrid_gamma,
        ),
        HybridMarginStrategy(
            alpha=settings.hybrid_alpha,
            beta=settings.hybrid_beta,
            gamma=settings.hybrid_gamma,
            margin=settings.similarity_margin,
        ),
    ]
    metrics = {strategy.name: StrategyMetrics() for strategy in strategies}

    embedder = OpenAIEmbeddingClient(settings=settings) if settings.openai_api_key else None
    retrieval = RetrievalService(
        embedder=embedder if embedder is not None else _NoopEmbedder(),
        model_name=settings.openai_embedding_model,
        embedding_version=settings.embedding_version,
        top_k=settings.top_k,
    )
    router = DomainRouter()

    cache = _load_embedding_cache(EMBEDDING_CACHE_PATH)
    with SessionLocal() as db:
        for row in rows:
            query_norm = normalize_text(row.query)
            if detect_prompt_injection(row.query).is_injection:
                for strategy in strategies:
                    metric = metrics[strategy.name]
                    metric.source_total += 1
                    metric.expected_not_local_total += 1
                    predicted_source = "compliance"
                    if predicted_source == row.expected_source:
                        metric.source_correct += 1
                continue

            domain = router.route_domain(row.query)

            query_embedding = _get_query_embedding(query_norm=query_norm, cache=cache, embedder=embedder)
            retrieval_decision = retrieval.retrieve_candidates_from_embedding(
                db,
                query_norm=query_norm,
                query_embedding=query_embedding,
            )

            for strategy in strategies:
                selection = strategy.score_candidates(
                    query_norm=query_norm,
                    candidates=retrieval_decision.candidates,
                    predicted_category=domain.category,
                )
                predicted_source = (
                    "local"
                    if strategy.accepts_local(selection=selection, threshold=settings.similarity_threshold)
                    else ("openai" if domain.in_domain else "compliance")
                )
                metric = metrics[strategy.name]
                metric.source_total += 1
                if row.expected_source == "local":
                    metric.expected_local_total += 1
                else:
                    metric.expected_not_local_total += 1
                if predicted_source == row.expected_source:
                    metric.source_correct += 1
                if predicted_source == "local" and row.expected_source != "local":
                    metric.false_positive_local += 1
                if predicted_source != "local" and row.expected_source == "local":
                    metric.false_negative_local += 1

                if row.label_type == "positive" and row.expected_match:
                    metric.positive_total += 1
                    if selection.best and selection.best.question == row.expected_match:
                        metric.top1_hits += 1
                    ranked_questions = _ranked_questions(retrieval_decision, selection.debug)
                    metric.mrr_sum += _reciprocal_rank(row.expected_match, ranked_questions)

    _save_embedding_cache(EMBEDDING_CACHE_PATH, cache)

    result_rows: list[tuple[str, dict[str, float]]] = []
    for strategy in strategies:
        result_rows.append((strategy.name, metrics[strategy.name].as_result()))

    print("\nBenchmark complete")
    table = _format_markdown_table(result_rows)
    print(table)

    timestamp = datetime.now(timezone.utc).isoformat()
    config_block = {
        "similarity_threshold": settings.similarity_threshold,
        "similarity_margin": settings.similarity_margin,
        "top_k": settings.top_k,
        "hybrid_alpha": settings.hybrid_alpha,
        "hybrid_beta": settings.hybrid_beta,
        "hybrid_gamma": settings.hybrid_gamma,
        "embedding_model": settings.openai_embedding_model,
        "embedding_version": settings.embedding_version,
    }
    interpretation = (
        "- Higher Top1/MRR means better retrieval quality.\n"
        "- Lower FPR(local) means fewer wrong local answers.\n"
        "- Margin and hybrid strategies often trade some recall for safer routing."
    )
    RESULTS_PATH.write_text(
        "\n".join(
            [
                "# Benchmark Results",
                "",
                f"- Timestamp: `{timestamp}`",
                f"- Config: `{json.dumps(config_block)}`",
                "",
                table,
                "",
                "## Interpretation Notes",
                interpretation,
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
