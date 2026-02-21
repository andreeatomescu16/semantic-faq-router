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
from app.services.scoring import ScoredSelection, ScoringStrategy, build_scoring_strategy

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
    is_injection: bool


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
            injection = detect_prompt_injection(row.query)
            if injection.is_injection:
                samples.append(
                    PrecomputedSample(
                        row=row,
                        query_norm=query_norm,
                        in_domain=False,
                        predicted_category="N/A",
                        retrieval=RetrievalDecision(query_norm=query_norm, candidates=[]),
                        is_injection=True,
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
                        is_injection=False,
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
                    in_domain=True,
                    predicted_category=domain.category,
                    retrieval=retrieval_decision,
                    is_injection=False,
                )
            )
    return samples


def _predict_source_for_sample(
    sample: PrecomputedSample,
    *,
    strategy: ScoringStrategy,
    threshold: float,
) -> tuple[str, ScoredSelection | None]:
    if sample.is_injection:
        return "compliance", None
    if not sample.in_domain:
        return "compliance", None

    selection = strategy.score_candidates(
        query_norm=sample.query_norm,
        candidates=sample.retrieval.candidates,
        predicted_category=sample.predicted_category,
    )
    predicted_source = "local" if strategy.accepts_local(selection, threshold) else "openai"
    return predicted_source, selection


def _evaluate_config(
    samples: list[PrecomputedSample],
    strategy: ScoringStrategy,
    *,
    threshold: float,
    cost_fp_local: float,
    cost_fn_local: float,
    cost_openai_call: float,
    cost_false_compliance: float,
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

    predicted_local = 0
    predicted_openai = 0
    predicted_compliance = 0
    false_compliance = 0

    for sample in samples:
        expected_source = sample.row.expected_source
        source_total += 1
        predicted_source, selection = _predict_source_for_sample(
            sample,
            strategy=strategy,
            threshold=threshold,
        )

        if predicted_source == "local":
            predicted_local += 1
        elif predicted_source == "openai":
            predicted_openai += 1
        else:
            predicted_compliance += 1

        if predicted_source == expected_source:
            source_correct += 1

        expected_local = expected_source == "local"
        predicted_local_flag = predicted_source == "local"
        if expected_local and predicted_local_flag:
            tp_local += 1
        elif (not expected_local) and predicted_local_flag:
            fp_local += 1
        elif expected_local and (not predicted_local_flag):
            fn_local += 1
        else:
            tn_local += 1

        if predicted_source == "compliance" and expected_source == "local":
            false_compliance += 1

        if (
            sample.row.label_type == "positive"
            and sample.row.expected_match
            and selection is not None
        ):
            positive_total += 1
            if selection.best and selection.best.question == sample.row.expected_match:
                top1_hits += 1
            ranked_questions = _ranked_questions(selection.debug, sample.retrieval.candidates)
            mrr_sum += _reciprocal_rank(sample.row.expected_match, ranked_questions)
        elif sample.row.label_type == "positive" and sample.row.expected_match and selection is None:
            positive_total += 1

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
        predicted_local=predicted_local,
        predicted_openai=predicted_openai,
        predicted_compliance=predicted_compliance,
        false_compliance=false_compliance,
        cost_fp_local=cost_fp_local,
        cost_fn_local=cost_fn_local,
        cost_openai_call=cost_openai_call,
        cost_false_compliance=cost_false_compliance,
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
        objective_cost=objective_routing_cost(metrics),
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
        "kb_utilization",
        "local_rate",
        "openai_rate",
        "openai_calls",
        "false_compliance",
        "total_cost",
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
            metric = record.metrics
            writer.writerow(
                {
                    "strategy": record.strategy_name,
                    "threshold": f"{record.threshold:.2f}",
                    "margin": f"{record.margin:.2f}",
                    "alpha": f"{record.alpha:.2f}",
                    "beta": f"{record.beta:.2f}",
                    "gamma": f"{record.gamma:.2f}",
                    "source_accuracy": f"{metric.source_accuracy:.4f}",
                    "top1_accuracy": f"{metric.top1_accuracy:.4f}",
                    "mrr": f"{metric.mrr:.4f}",
                    "fpr_local": f"{metric.fpr_local:.4f}",
                    "fnr_local": f"{metric.fnr_local:.4f}",
                    "precision_local": f"{metric.precision_local:.4f}",
                    "recall_local": f"{metric.recall_local:.4f}",
                    "f1_local": f"{metric.f1_local:.4f}",
                    "kb_utilization": f"{metric.kb_utilization:.4f}",
                    "local_rate": f"{metric.local_rate:.4f}",
                    "openai_rate": f"{metric.openai_rate:.4f}",
                    "openai_calls": metric.predicted_openai,
                    "false_compliance": metric.false_compliance,
                    "total_cost": f"{metric.total_cost:.4f}",
                    "tp_local": metric.tp_local,
                    "fp_local": metric.fp_local,
                    "fn_local": metric.fn_local,
                    "tn_local": metric.tn_local,
                    "objective_source_accuracy": f"{record.objective_source_accuracy:.4f}",
                    "objective_f1_local": f"{record.objective_f1_local:.4f}",
                    "objective_cost": f"{record.objective_cost:.4f}",
                }
            )


def _best_by(records: list[CalibrationRecord], key_name: str, minimize: bool) -> CalibrationRecord:
    if minimize:
        return min(records, key=lambda r: getattr(r, key_name))
    return max(records, key=lambda r: getattr(r, key_name))


def _objective_value(record: CalibrationRecord, objective: str) -> float:
    if objective == "source_acc":
        return record.objective_source_accuracy
    if objective == "f1_local":
        return record.objective_f1_local
    return record.objective_cost


def _summary_row(record: CalibrationRecord, objective_name: str, objective_value: float) -> str:
    metric = record.metrics
    return (
        f"| {record.strategy_name} | {record.threshold:.2f} | {record.margin:.2f} | "
        f"{metric.source_accuracy:.3f} | {metric.top1_accuracy:.3f} | {metric.mrr:.3f} | "
        f"{metric.fpr_local:.3f} | {metric.fnr_local:.3f} | {metric.precision_local:.3f} | "
        f"{metric.recall_local:.3f} | {metric.f1_local:.3f} | {metric.kb_utilization:.3f} | "
        f"{metric.local_rate:.3f} | {metric.openai_rate:.3f} | {metric.total_cost:.2f} | "
        f"{objective_name}={objective_value:.3f} |"
    )


def _print_table(title: str, rows: list[str]) -> None:
    print(f"\n{title}")
    print(
        "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | "
        "Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(row)


def _find_safety_first(records: list[CalibrationRecord]) -> CalibrationRecord:
    return min(
        records,
        key=lambda r: (
            r.metrics.fp_local,
            r.metrics.fpr_local,
            r.metrics.total_cost,
            -r.metrics.source_accuracy,
        ),
    )


def _write_markdown(
    records: list[CalibrationRecord],
    best_by_objective: dict[str, CalibrationRecord],
    best_source: dict[str, CalibrationRecord],
    best_f1: dict[str, CalibrationRecord],
    best_cost: dict[str, CalibrationRecord],
    *,
    routing_objective: str,
    cost_fp_local: float,
    cost_fn_local: float,
    cost_openai_call: float,
    cost_false_compliance: float,
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    settings = get_settings()
    objective_name = "cost" if routing_objective == "cost" else routing_objective
    safety_first = _find_safety_first(records)
    cost_optimized = _best_by(records, "objective_cost", minimize=True)

    lines: list[str] = [
        "# Calibration Results",
        "",
        f"- Timestamp: `{timestamp}`",
        f"- Dataset: `{BENCHMARK_DATASET}`",
        f"- ROUTING_OBJECTIVE: `{objective_name}`",
        (
            "- Cost config: "
            f"`COST_FP_LOCAL={cost_fp_local}`, "
            f"`COST_FN_LOCAL={cost_fn_local}`, "
            f"`COST_OPENAI_CALL={cost_openai_call}`, "
            f"`COST_FALSE_COMPLIANCE={cost_false_compliance}`"
        ),
        f"- Hybrid weights: `alpha={settings.hybrid_alpha}`, `beta={settings.hybrid_beta}`, `gamma={settings.hybrid_gamma}`",
        "",
        "## Best Config per Strategy (selected objective)",
        "",
        "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for strategy in STRATEGIES:
        rec = best_by_objective[strategy]
        lines.append(_summary_row(rec, objective_name, _objective_value(rec, routing_objective)))

    lines.extend(
        [
            "",
            "## Two Recommended Configs",
            "",
            f"- **Safety-first**: `{safety_first.strategy_name}` thr `{safety_first.threshold:.2f}` margin `{safety_first.margin:.2f}` (FP_local={safety_first.metrics.fp_local}, FPR={safety_first.metrics.fpr_local:.3f})",
            f"- **Cost-optimized**: `{cost_optimized.strategy_name}` thr `{cost_optimized.threshold:.2f}` margin `{cost_optimized.margin:.2f}` (TotalCost={cost_optimized.metrics.total_cost:.2f})",
            "",
            "## Best by Source Accuracy",
            "",
            "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in STRATEGIES:
        rec = best_source[strategy]
        lines.append(_summary_row(rec, "src_acc", rec.objective_source_accuracy))

    lines.extend(
        [
            "",
            "## Best by F1(local)",
            "",
            "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in STRATEGIES:
        rec = best_f1[strategy]
        lines.append(_summary_row(rec, "f1_local", rec.objective_f1_local))

    lines.extend(
        [
            "",
            "## Best by Total Cost",
            "",
            "| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for strategy in STRATEGIES:
        rec = best_cost[strategy]
        lines.append(_summary_row(rec, "cost", rec.objective_cost))

    lines.extend(["", "## Recommended .env settings (selected objective)", ""])
    for strategy in STRATEGIES:
        rec = best_by_objective[strategy]
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
                f"ROUTING_OBJECTIVE={objective_name}",
                "```",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "- Thresholds are empirical; recalibrate when KB content/distribution changes.",
            "- Margin strategies usually reduce false-positive local answers.",
            "- Cost objective balances wrong-local risk with unnecessary OpenAI usage.",
            "",
            f"Raw sweep results are in `{CSV_OUTPUT_PATH}`.",
        ]
    )

    MD_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    settings = get_settings()
    cost_fp_local = float(os.getenv("COST_FP_LOCAL", "8.0"))
    cost_fn_local = float(os.getenv("COST_FN_LOCAL", "2.0"))
    cost_openai_call = float(os.getenv("COST_OPENAI_CALL", "0.3"))
    cost_false_compliance = float(os.getenv("COST_FALSE_COMPLIANCE", "3.0"))
    routing_objective = os.getenv("ROUTING_OBJECTIVE", "cost").strip().lower()
    if routing_objective not in {"cost", "source_acc", "f1_local"}:
        routing_objective = "cost"

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
            cost_openai_call=cost_openai_call,
            cost_false_compliance=cost_false_compliance,
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
    if routing_objective == "source_acc":
        best_selected = best_source
    elif routing_objective == "f1_local":
        best_selected = best_f1
    else:
        best_selected = best_cost

    objective_label = "cost" if routing_objective == "cost" else routing_objective
    selected_rows = [
        _summary_row(best_selected[name], objective_label, _objective_value(best_selected[name], routing_objective))
        for name in STRATEGIES
    ]
    _print_table("Best configs by selected objective", selected_rows)

    source_rows = [
        _summary_row(best_source[name], "src_acc", best_source[name].objective_source_accuracy)
        for name in STRATEGIES
    ]
    f1_rows = [_summary_row(best_f1[name], "f1_local", best_f1[name].objective_f1_local) for name in STRATEGIES]
    cost_rows = [_summary_row(best_cost[name], "cost", best_cost[name].objective_cost) for name in STRATEGIES]
    _print_table("Best configs by Source Accuracy", source_rows)
    _print_table("Best configs by F1(local)", f1_rows)
    _print_table("Best configs by Total Cost", cost_rows)

    _write_markdown(
        all_records,
        best_selected,
        best_source,
        best_f1,
        best_cost,
        routing_objective=routing_objective,
        cost_fp_local=cost_fp_local,
        cost_fn_local=cost_fn_local,
        cost_openai_call=cost_openai_call,
        cost_false_compliance=cost_false_compliance,
    )
    print(f"\nWrote {CSV_OUTPUT_PATH} and {MD_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
