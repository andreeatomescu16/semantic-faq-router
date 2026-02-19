import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import get_settings
from app.db.repository import RetrievalResult
from app.db.session import SessionLocal
from app.services.calibration import (
    EvaluationMetrics,
    compute_local_metrics,
    objective_f1_local,
    objective_routing_cost,
    objective_source_accuracy,
)
from app.services.guardrails import detect_prompt_injection
from app.services.normalizer import normalize_text
from app.services.retrieval import RetrievalDecision, RetrievalService
from app.services.router import DomainRouter
from app.services.scoring import ScoringStrategy, build_scoring_strategy

BENCHMARK_DATASET = Path("data/benchmark_queries.jsonl")
EMBEDDING_CACHE_PATH = Path("data/benchmark_query_embeddings_cache.json")
CSV_OUTPUT_PATH = Path("calibration_results.csv")
MD_OUTPUT_PATH = Path("calibration_results.md")

STRATEGIES = ["cosine_threshold", "cosine_margin", "hybrid", "hybrid_margin"]


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError(
            f"OpenAI calls are disabled in calibration. Missing cached embedding for: {text}"
        )


@dataclass(frozen=True)
class BenchmarkRow:
    benchmark_id: str
    query: str
    label_type: str
    expected_source: str
    expected_match: str | None


@dataclass(frozen=True)
class PrecomputedSample:
    row: BenchmarkRow
    query_norm: str
    in_domain: bool
    predicted_category: str
    retrieval: RetrievalDecision


@dataclass(frozen=True)
class StrategyConfig:
    strategy_name: str
    threshold: float
    margin: float
    alpha: float
    beta: float
    gamma: float


@dataclass(frozen=True)
class CalibrationRecord:
    strategy_name: str
    threshold: float
    margin: float
    alpha: float
    beta: float
    gamma: float
    metrics: EvaluationMetrics
    objective_source_accuracy: float
    objective_f1_local: float
    objective_cost: float


def _float_range(start: float, end: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= end + 1e-9:
        values.append(round(current, 2))
        current += step
    return values


def _load_rows(path: Path) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
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
        raise FileNotFoundError(
            "Missing data/benchmark_query_embeddings_cache.json. "
            "Run benchmark once to populate cache without repeated embedding calls."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _reciprocal_rank(expected_match: str, ranked_questions: list[str]) -> float:
    for idx, question in enumerate(ranked_questions, start=1):
        if question == expected_match:
            return 1.0 / idx
    return 0.0


def _ranked_questions(selection_debug: dict[str, object], candidates: list[RetrievalResult]) -> list[str]:
    ranked = selection_debug.get("ranked_questions")
    if isinstance(ranked, list) and all(isinstance(item, str) for item in ranked):
        return list(ranked)
    return [candidate.question for candidate in candidates]


def _precompute_samples(
    rows: list[BenchmarkRow],
    embedding_cache: dict[str, list[float]],
) -> list[PrecomputedSample]:
    settings = get_settings()
    retrieval = RetrievalService(
        embedder=_NoopEmbedder(),
        model_name=settings.openai_embedding_model,
        embedding_version=settings.embedding_version,
        top_k=settings.top_k,
    )
    router = DomainRouter()

    samples: list[PrecomputedSample] = []
    with SessionLocal() as db:
        for row in rows:
            query_norm = normalize_text(row.query)

            # Guardrail first; calibration is routing-only, no OpenAI call.
            if detect_prompt_injection(row.query).is_injection:
                samples.append(
                    PrecomputedSample(
                        row=row,
                        query_norm=query_norm,
                        in_domain=False,
                        predicted_category="N/A",
                        retrieval=RetrievalDecision(query_norm=query_norm, candidates=[]),
                    )
                )
                continue

            domain = router.route_domain(row.query)
            if not domain.in_domain:
                samples.append(
                    PrecomputedSample(
                        row=row,
                        query_norm=query_norm,
                        in_domain=False,
                        predicted_category="N/A",
                        retrieval=RetrievalDecision(query_norm=query_norm, candidates=[]),
                    )
                )
                continue

            if query_norm not in embedding_cache:
                raise ValueError(
                    f"Query embedding missing from cache for normalized query: '{query_norm}'. "
                    "Run scripts.benchmark_retrieval once to populate cache."
                )

            retrieval_decision = retrieval.retrieve_candidates_from_embedding(
                db,
                query_norm=query_norm,
                query_embedding=embedding_cache[query_norm],
            )
            samples.append(
                PrecomputedSample(
                    row=row,
                    query_norm=query_norm,
                    in_domain=domain.in_domain,
                    predicted_category=domain.category,
                    retrieval=retrieval_decision,
                )
            )
    return samples


def _evaluate_config(
    samples: list[PrecomputedSample],
    strategy: ScoringStrategy,
    *,
    threshold: float,
    cost_fp_local: float,
    cost_fn_local: float,
    config: StrategyConfig,
) -> CalibrationRecord:
    source_total = 0
    source_correct = 0

    tp_local = 0
    fp_local = 0
    fn_local = 0
    tn_local = 0

    positive_total = 0
    top1_hits = 0
    mrr_sum = 0.0

    for sample in samples:
        expected_source = sample.row.expected_source
        source_total += 1

        if not sample.in_domain:
            predicted_source = "compliance"
        else:
            selection = strategy.score_candidates(
                query_norm=sample.query_norm,
                candidates=sample.retrieval.candidates,
                predicted_category=sample.predicted_category,
            )
            predicted_source = (
                "local"
                if strategy.accepts_local(selection=selection, threshold=threshold)
                else "openai"
            )

            if sample.row.label_type == "positive" and sample.row.expected_match:
                positive_total += 1
                if selection.best and selection.best.question == sample.row.expected_match:
                    top1_hits += 1
                ranked_questions = _ranked_questions(selection.debug, sample.retrieval.candidates)
                mrr_sum += _reciprocal_rank(sample.row.expected_match, ranked_questions)

        if predicted_source == expected_source:
            source_correct += 1

        expected_local = expected_source == "local"
        predicted_local = predicted_source == "local"

        if expected_local and predicted_local:
            tp_local += 1
        elif (not expected_local) and predicted_local:
            fp_local += 1
        elif expected_local and (not predicted_local):
            fn_local += 1
        else:
            tn_local += 1

    metrics = compute_local_metrics(
        source_correct=source_correct,
        source_total=source_total,
        positive_total=positive_total,
        top1_hits=top1_hits,
        mrr_sum=mrr_sum,
        tp_local=tp_local,
        fp_local=fp_local,
        fn_local=fn_local,
        tn_local=tn_local,
    )

    return CalibrationRecord(
        strategy_name=config.strategy_name,
        threshold=config.threshold,
        margin=config.margin,
        alpha=config.alpha,
        beta=config.beta,
        gamma=config.gamma,
        metrics=metrics,
        objective_source_accuracy=objective_source_accuracy(metrics),
        objective_f1_local=objective_f1_local(metrics),
        objective_cost=objective_routing_cost(
            metrics,
            cost_fp_local=cost_fp_local,
            cost_fn_local=cost_fn_local,
        ),
    )


def _build_strategy_configs() -> list[StrategyConfig]:
    settings = get_settings()
    thresholds = _float_range(0.50, 0.95, 0.01)
    margins = _float_range(0.00, 0.10, 0.01)

    configs: list[StrategyConfig] = []

    for threshold in thresholds:
        configs.append(
            StrategyConfig(
                strategy_name="cosine_threshold",
                threshold=threshold,
                margin=0.0,
                alpha=settings.hybrid_alpha,
                beta=settings.hybrid_beta,
                gamma=settings.hybrid_gamma,
            )
        )

    for threshold in thresholds:
        for margin in margins:
            configs.append(
                StrategyConfig(
                    strategy_name="cosine_margin",
                    threshold=threshold,
                    margin=margin,
                    alpha=settings.hybrid_alpha,
                    beta=settings.hybrid_beta,
                    gamma=settings.hybrid_gamma,
                )
            )

    for threshold in thresholds:
        configs.append(
            StrategyConfig(
                strategy_name="hybrid",
                threshold=threshold,
                margin=0.0,
                alpha=settings.hybrid_alpha,
                beta=settings.hybrid_beta,
                gamma=settings.hybrid_gamma,
            )
        )

    for threshold in thresholds:
        for margin in margins:
            configs.append(
                StrategyConfig(
                    strategy_name="hybrid_margin",
                    threshold=threshold,
                    margin=margin,
                    alpha=settings.hybrid_alpha,
                    beta=settings.hybrid_beta,
                    gamma=settings.hybrid_gamma,
                )
            )

    return configs


def _write_csv(records: list[CalibrationRecord]) -> None:
    fieldnames = [
        "strategy",
        "threshold",
        "margin",
        "alpha",
        "beta",
        "gamma",
        "source_accuracy",
        "top1_accuracy",
        "mrr",
        "fpr_local",
        "fnr_local",
        "precision_local",
        "recall_local",
        "f1_local",
        "tp_local",
        "fp_local",
        "fn_local",
        "tn_local",
        "objective_source_accuracy",
        "objective_f1_local",
        "objective_cost",
    ]
    with CSV_OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "strategy": record.strategy_name,
                    "threshold": f"{record.threshold:.2f}",
                    "margin": f"{record.margin:.2f}",
                    "alpha": f"{record.alpha:.2f}",
                    "beta": f"{record.beta:.2f}",
                    "gamma": f"{record.gamma:.2f}",
                    "source_accuracy": f"{record.metrics.source_accuracy:.4f}",
                    "top1_accuracy": f"{record.metrics.top1_accuracy:.4f}",
                    "mrr": f"{record.metrics.mrr:.4f}",
                    "fpr_local": f"{record.metrics.fpr_local:.4f}",
                    "fnr_local": f"{record.metrics.fnr_local:.4f}",
                    "precision_local": f"{record.metrics.precision_local:.4f}",
                    "recall_local": f"{record.metrics.recall_local:.4f}",
                    "f1_local": f"{record.metrics.f1_local:.4f}",
                    "tp_local": record.metrics.tp_local,
                    "fp_local": record.metrics.fp_local,
                    "fn_local": record.metrics.fn_local,
                    "tn_local": record.metrics.tn_local,
                    "objective_source_accuracy": f"{record.objective_source_accuracy:.4f}",
                    "objective_f1_local": f"{record.objective_f1_local:.4f}",
                    "objective_cost": f"{record.objective_cost:.4f}",
                }
            )


def _best_by(records: list[CalibrationRecord], key_name: str, minimize: bool) -> CalibrationRecord:
    if minimize:
        return min(records, key=lambda r: getattr(r, key_name))
    return max(records, key=lambda r: getattr(r, key_name))


def _summary_row(record: CalibrationRecord, objective_name: str, objective_value: float) -> str:
    return (
        f"| {record.strategy_name} | {record.threshold:.2f} | {record.margin:.2f} | "
        f"{record.metrics.source_accuracy:.3f} | {record.metrics.top1_accuracy:.3f} | "
        f"{record.metrics.mrr:.3f} | {record.metrics.fpr_local:.3f} | "
        f"{record.metrics.fnr_local:.3f} | {record.metrics.precision_local:.3f} | "
        f"{record.metrics.recall_local:.3f} | {record.metrics.f1_local:.3f} | "
        f"{objective_name}={objective_value:.3f} |"
    )


def _print_table(title: str, rows: list[str]) -> None:
    print(f"\n{title}")
    print(
        "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | "
        "FNR(local) | Precision(local) | Recall(local) | F1(local) | Objective |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(row)


def _write_markdown(
    records: list[CalibrationRecord],
    best_source: dict[str, CalibrationRecord],
    best_f1: dict[str, CalibrationRecord],
    best_cost: dict[str, CalibrationRecord],
    *,
    cost_fp_local: float,
    cost_fn_local: float,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    settings = get_settings()

    lines: list[str] = [
        "# Calibration Results",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Dataset: `{BENCHMARK_DATASET}`",
        f"- Cost config: `COST_FP_LOCAL={cost_fp_local}`, `COST_FN_LOCAL={cost_fn_local}`",
        f"- Hybrid weights: `alpha={settings.hybrid_alpha}`, `beta={settings.hybrid_beta}`, `gamma={settings.hybrid_gamma}`",
        "",
        "## Best by Source Accuracy",
        "",
        "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | Objective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for strategy in STRATEGIES:
        rec = best_source[strategy]
        lines.append(_summary_row(rec, "src_acc", rec.objective_source_accuracy))

    lines.extend(
        [
            "",
            "## Best by F1(local)",
            "",
            "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | Objective |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in STRATEGIES:
        rec = best_f1[strategy]
        lines.append(_summary_row(rec, "f1_local", rec.objective_f1_local))

    lines.extend(
        [
            "",
            "## Best by Routing Cost (minimize)",
            "",
            "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | Objective |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in STRATEGIES:
        rec = best_cost[strategy]
        lines.append(_summary_row(rec, "cost", rec.objective_cost))

    lines.extend(
        [
            "",
            "## Recommended .env settings (cost-optimized)",
            "",
        ]
    )

    for strategy in STRATEGIES:
        rec = best_cost[strategy]
        lines.extend(
            [
                f"### {strategy}",
                "```env",
                f"SCORING_STRATEGY={strategy}",
                f"SIMILARITY_THRESHOLD={rec.threshold:.2f}",
                f"SIMILARITY_MARGIN={rec.margin:.2f}",
                f"HYBRID_ALPHA={rec.alpha:.2f}",
                f"HYBRID_BETA={rec.beta:.2f}",
                f"HYBRID_GAMMA={rec.gamma:.2f}",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "- Thresholds are empirical; recalibrate when KB content/distribution changes.",
            "- Margin strategies usually reduce false-positive local answers.",
            "- Cost objective is useful when wrong local answers are much more expensive than fallback.",
            "",
            f"Raw sweep results are in `{CSV_OUTPUT_PATH}`.",
        ]
    )

    MD_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cost_fp_local = float(os.getenv("COST_FP_LOCAL", "5"))
    cost_fn_local = float(os.getenv("COST_FN_LOCAL", "1"))

    rows = _load_rows(BENCHMARK_DATASET)
    embedding_cache = _load_embedding_cache(EMBEDDING_CACHE_PATH)
    samples = _precompute_samples(rows, embedding_cache)

    all_configs = _build_strategy_configs()
    all_records: list[CalibrationRecord] = []

    for config in all_configs:
        strategy = build_scoring_strategy(
            config.strategy_name,
            margin=config.margin,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
        )
        record = _evaluate_config(
            samples=samples,
            strategy=strategy,
            threshold=config.threshold,
            cost_fp_local=cost_fp_local,
            cost_fn_local=cost_fn_local,
            config=config,
        )
        all_records.append(record)

    _write_csv(all_records)

    grouped: dict[str, list[CalibrationRecord]] = {name: [] for name in STRATEGIES}
    for record in all_records:
        grouped[record.strategy_name].append(record)

    best_source = {name: _best_by(grouped[name], "objective_source_accuracy", minimize=False) for name in STRATEGIES}
    best_f1 = {name: _best_by(grouped[name], "objective_f1_local", minimize=False) for name in STRATEGIES}
    best_cost = {name: _best_by(grouped[name], "objective_cost", minimize=True) for name in STRATEGIES}

    source_rows = [
        _summary_row(best_source[name], "src_acc", best_source[name].objective_source_accuracy) for name in STRATEGIES
    ]
    f1_rows = [_summary_row(best_f1[name], "f1_local", best_f1[name].objective_f1_local) for name in STRATEGIES]
    cost_rows = [_summary_row(best_cost[name], "cost", best_cost[name].objective_cost) for name in STRATEGIES]

    _print_table("Best configs by Source Accuracy", source_rows)
    _print_table("Best configs by F1(local)", f1_rows)
    _print_table("Best configs by Routing Cost", cost_rows)

    _write_markdown(
        all_records,
        best_source,
        best_f1,
        best_cost,
        cost_fp_local=cost_fp_local,
        cost_fn_local=cost_fn_local,
    )

    print(f"\nWrote {CSV_OUTPUT_PATH} and {MD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
