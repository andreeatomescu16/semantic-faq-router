"""Offline LLM-as-a-judge evaluation.

Smoke run:
1) docker compose exec -w /app api python -m scripts.benchmark_retrieval
2) docker compose exec -w /app api python -m scripts.llm_judge_eval
3) Inspect llm_judge_report.md:
   - Coverage > 0
   - Valid first pass > 0 (when JSON mode is active/supported)
   - Source alignment reflects expected_source.
"""

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.guardrails import detect_prompt_injection
from app.services.llm_judge import (
    EvalCandidate,
    EvalRecord,
    JudgeExecutionResult,
    JudgeOutputParseError,
    JudgeVerdict,
    LLMJudgeClient,
)
from app.services.normalizer import normalize_text
from app.services.retrieval import RetrievalDecision, RetrievalService
from app.services.router import DomainRouter
from app.services.scoring import build_scoring_strategy

BENCHMARK_DATASET = Path("data/benchmark_queries.jsonl")
EMBEDDING_CACHE_PATH = Path("data/benchmark_query_embeddings_cache.json")
JSONL_OUTPUT = Path("llm_judge_results.jsonl")
MD_OUTPUT = Path("llm_judge_report.md")
logger = logging.getLogger(__name__)


class _NoopEmbedder:
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError(
            f"OpenAI calls are disabled in judge eval precompute. Missing cached embedding for: {text}"
        )


class JudgeClientLike(Protocol):
    def judge_with_repair(self, record: EvalRecord) -> JudgeExecutionResult:
        """Return judge execution result for one eval record."""


@dataclass(frozen=True)
class BenchRow:
    benchmark_id: str
    query: str
    expected_source: str | None


@dataclass(frozen=True)
class JudgedCase:
    eval_record: EvalRecord
    verdict: JudgeVerdict
    computed_verdict: str
    is_gray: bool


@dataclass(frozen=True)
class JudgeRunMetrics:
    attempted_cases: int = 0
    valid_first_pass: int = 0
    repaired_successfully: int = 0
    still_invalid: int = 0

    @property
    def total_valid(self) -> int:
        return self.valid_first_pass + self.repaired_successfully

    @property
    def coverage(self) -> float:
        if self.attempted_cases == 0:
            return 0.0
        return self.total_valid / self.attempted_cases


def _load_rows() -> list[BenchRow]:
    rows: list[BenchRow] = []
    with BENCHMARK_DATASET.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(
                BenchRow(
                    benchmark_id=str(payload["id"]),
                    query=str(payload["query"]),
                    expected_source=payload.get("expected_source"),
                )
            )
    return rows


def _load_embedding_cache() -> dict[str, list[float]]:
    if not EMBEDDING_CACHE_PATH.exists():
        raise FileNotFoundError(
            "Missing benchmark embedding cache. Run `python -m scripts.benchmark_retrieval` first."
        )
    return json.loads(EMBEDDING_CACHE_PATH.read_text(encoding="utf-8"))


def _build_eval_records(
    rows: list[BenchRow],
    *,
    embedding_cache: dict[str, list[float]],
) -> list[EvalRecord]:
    settings = get_settings()
    router = DomainRouter()
    scoring = build_scoring_strategy(
        settings.scoring_strategy,
        margin=settings.similarity_margin,
        alpha=settings.hybrid_alpha,
        beta=settings.hybrid_beta,
        gamma=settings.hybrid_gamma,
    )
    retrieval = RetrievalService(
        embedder=_NoopEmbedder(),
        model_name=settings.openai_embedding_model,
        embedding_version=settings.embedding_version,
        top_k=settings.top_k,
    )

    records: list[EvalRecord] = []
    with SessionLocal() as db:
        for row in rows:
            query_norm = normalize_text(row.query)
            injection = detect_prompt_injection(row.query).is_injection
            if injection:
                records.append(
                    EvalRecord(
                        id=row.benchmark_id,
                        query=row.query,
                        expected_source=row.expected_source,
                        predicted_source="compliance",
                        confidence=None,
                        selected_index=None,
                        selected_question=None,
                        top_k=[],
                    )
                )
                continue

            domain = router.route_domain(row.query)
            if query_norm not in embedding_cache:
                raise ValueError(
                    f"Missing query embedding in cache for normalized query '{query_norm}'."
                )

            retrieval_decision: RetrievalDecision = retrieval.retrieve_candidates_from_embedding(
                db,
                query_norm=query_norm,
                query_embedding=embedding_cache[query_norm],
            )
            selection = scoring.score_candidates(
                query_norm=query_norm,
                candidates=retrieval_decision.candidates,
                predicted_category=domain.category,
            )
            if scoring.accepts_local(selection, settings.similarity_threshold):
                predicted_source = "local"
            elif domain.in_domain:
                predicted_source = "openai"
            else:
                predicted_source = "compliance"

            selected_index: int | None = None
            selected_question: str | None = None
            if selection.best:
                selected_question = selection.best.question
                for idx, candidate in enumerate(retrieval_decision.candidates):
                    if candidate.question == selection.best.question:
                        selected_index = idx
                        break

            top_k = [
                EvalCandidate(
                    index=idx,
                    question=candidate.question,
                    category=candidate.category,
                    score=candidate.score,
                )
                for idx, candidate in enumerate(retrieval_decision.candidates)
            ]
            records.append(
                EvalRecord(
                    id=row.benchmark_id,
                    query=row.query,
                    expected_source=row.expected_source,
                    predicted_source=predicted_source,
                    confidence=selection.final_confidence,
                    selected_index=selected_index,
                    selected_question=selected_question,
                    top_k=top_k,
                )
            )
    return records


def should_judge_case(
    record: EvalRecord,
    *,
    gray_low: float,
    gray_high: float,
    similarity_threshold: float,
) -> bool:
    case_1 = (
        record.expected_source is not None
        and record.predicted_source != record.expected_source
    )
    case_2 = (
        record.confidence is not None
        and gray_low <= record.confidence <= gray_high
    )
    case_3 = record.predicted_source == "compliance"
    case_4 = (
        record.predicted_source == "openai"
        and record.confidence is not None
        and abs(record.confidence - similarity_threshold) <= 0.03
    )
    return case_1 or case_2 or case_3 or case_4


def judge_records(
    records: list[EvalRecord],
    *,
    judge_client: JudgeClientLike,
    max_cases: int,
    gray_low: float,
    gray_high: float,
    similarity_threshold: float,
) -> tuple[list[JudgedCase], JudgeRunMetrics]:
    judged: list[JudgedCase] = []
    attempted_cases = 0
    valid_first_pass = 0
    repaired_successfully = 0
    still_invalid = 0

    for record in records:
        if attempted_cases >= max_cases:
            break
        if not should_judge_case(
            record,
            gray_low=gray_low,
            gray_high=gray_high,
            similarity_threshold=similarity_threshold,
        ):
            continue
        attempted_cases += 1
        try:
            execution = judge_client.judge_with_repair(record)
        except JudgeOutputParseError:
            still_invalid += 1
            logger.warning("Invalid judge output for record_id=%s", record.id)
            continue
        except Exception:
            still_invalid += 1
            logger.exception("Judge call failed for record_id=%s", record.id)
            continue

        verdict = execution.verdict
        if verdict is None:
            still_invalid += 1
            logger.warning("Judge output still invalid after repair for record_id=%s", record.id)
            continue
        if execution.valid_first_pass:
            valid_first_pass += 1
        elif execution.repaired:
            repaired_successfully += 1

        is_gray = (
            record.confidence is not None and gray_low <= record.confidence <= gray_high
        )
        judged.append(
            JudgedCase(
                eval_record=record,
                verdict=verdict,
                computed_verdict=execution.computed_verdict,
                is_gray=is_gray,
            )
        )
    metrics = JudgeRunMetrics(
        attempted_cases=attempted_cases,
        valid_first_pass=valid_first_pass,
        repaired_successfully=repaired_successfully,
        still_invalid=still_invalid,
    )
    return judged, metrics


def _write_jsonl(judged_cases: list[JudgedCase]) -> None:
    with JSONL_OUTPUT.open("w", encoding="utf-8") as handle:
        for case in judged_cases:
            payload = {
                "eval_record": case.eval_record.model_dump(),
                "judge_output": case.verdict.model_dump(),
                "is_gray": case.is_gray,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_markdown(
    judged_cases: list[JudgedCase],
    *,
    run_metrics: JudgeRunMetrics,
    max_cases: int,
) -> None:
    settings = get_settings()
    total = len(judged_cases)
    source_aligned_total = sum(
        1
        for case in judged_cases
        if case.verdict.preferred_source == case.eval_record.predicted_source
    )
    source_alignment_valid = (source_aligned_total / total) if total else 0.0
    source_alignment_overall = (
        source_aligned_total / run_metrics.attempted_cases
        if run_metrics.attempted_cases
        else 0.0
    )
    verdict_agree_total = sum(1 for case in judged_cases if case.computed_verdict == "agree")
    verdict_agree_rate = (verdict_agree_total / total) if total else 0.0

    gray_cases = [case for case in judged_cases if case.is_gray]
    gray_aligned = sum(
        1
        for case in gray_cases
        if case.verdict.preferred_source == case.eval_record.predicted_source
    )
    gray_source_alignment = (gray_aligned / len(gray_cases)) if gray_cases else 0.0

    breakdown_counts: dict[str, list[JudgedCase]] = defaultdict(list)
    for case in judged_cases:
        breakdown_counts[case.eval_record.predicted_source].append(case)

    rationale_counter = Counter(
        case.verdict.rationale.strip()
        for case in judged_cases
        if case.verdict.preferred_source != case.eval_record.predicted_source
    )
    top_rationales = rationale_counter.most_common(10)

    representative = judged_cases[:10]
    estimated_cost = total * settings.cost_openai_call

    lines: list[str] = [
        "# LLM Judge Report",
        "",
        f"- Timestamp: `{datetime.now(timezone.utc).isoformat()}`",
        f"- Judge model: `{settings.judge_model}`",
        f"- JUDGE_MAX_CASES: `{max_cases}`",
        f"- Attempted cases: `{run_metrics.attempted_cases}`",
        f"- Judged cases: `{total}`",
        f"- Skipped invalid judge outputs: `{run_metrics.still_invalid}`",
        f"- Estimated judge cost (calls * COST_OPENAI_CALL): `{estimated_cost:.2f}`",
        "",
        "## Coverage Metrics",
        f"- Attempted: `{run_metrics.attempted_cases}`",
        f"- Valid first pass: `{run_metrics.valid_first_pass}`",
        f"- Repaired successfully: `{run_metrics.repaired_successfully}`",
        f"- Still invalid: `{run_metrics.still_invalid}`",
        f"- Coverage: `{run_metrics.coverage:.3f}`",
        "",
        "## Source Alignment Metrics",
        f"- Source alignment (valid only): `{source_alignment_valid:.3f}`",
        f"- Source alignment (overall): `{source_alignment_overall:.3f}`",
        f"- Gray-zone source alignment: `{gray_source_alignment:.3f}`",
        f"- Judge verdict='agree' rate (valid only): `{verdict_agree_rate:.3f}`",
        "",
        "## Source Alignment by predicted_source",
    ]
    for source in ["local", "openai", "compliance"]:
        source_cases = breakdown_counts.get(source, [])
        aligned = sum(
            1
            for case in source_cases
            if case.verdict.preferred_source == case.eval_record.predicted_source
        )
        rate = (aligned / len(source_cases)) if source_cases else 0.0
        lines.append(f"- `{source}`: `{rate:.3f}` ({aligned}/{len(source_cases)})")

    lines.extend(["", "## Top Disagreement Reasons"])
    if not top_rationales:
        lines.append("- None.")
    else:
        for reason, count in top_rationales:
            lines.append(f"- `{reason}`: {count}")

    lines.extend(
        [
            "",
            "## Representative Examples (up to 10)",
            "",
            "| id | predicted | preferred | severity | confidence | rationale |",
            "|---|---|---|---|---:|---|",
        ]
    )
    for case in representative:
        conf = "N/A" if case.eval_record.confidence is None else f"{case.eval_record.confidence:.3f}"
        lines.append(
            "| {id} | {pred} | {pref} | {sev} | {conf} | {rat} |".format(
                id=case.eval_record.id,
                pred=case.eval_record.predicted_source,
                pref=case.verdict.preferred_source,
                sev=case.verdict.severity,
                conf=conf,
                rat=case.verdict.rationale.replace("|", "/"),
            )
        )

    lines.extend(
        [
            "",
            f"Raw judged outputs are in `{JSONL_OUTPUT}`.",
            "This judge is offline-only and not used in production serving.",
        ]
    )
    MD_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    settings = get_settings()
    rows = _load_rows()
    embedding_cache = _load_embedding_cache()
    eval_records = _build_eval_records(rows, embedding_cache=embedding_cache)

    if not settings.judge_enabled:
        _write_jsonl([])
        _write_markdown([], run_metrics=JudgeRunMetrics(), max_cases=settings.judge_max_cases)
        print("JUDGE_ENABLED=false; wrote empty judge artifacts.")
        return

    judge_client = LLMJudgeClient(settings=settings)
    judged_cases, run_metrics = judge_records(
        eval_records,
        judge_client=judge_client,
        max_cases=settings.judge_max_cases,
        gray_low=settings.judge_gray_low,
        gray_high=settings.judge_gray_high,
        similarity_threshold=settings.similarity_threshold,
    )
    _write_jsonl(judged_cases)
    _write_markdown(
        judged_cases,
        run_metrics=run_metrics,
        max_cases=settings.judge_max_cases,
    )
    print(
        "Attempted "
        f"{run_metrics.attempted_cases} cases; judged {len(judged_cases)} "
        f"(valid_first_pass={run_metrics.valid_first_pass}, "
        f"repaired={run_metrics.repaired_successfully}, invalid={run_metrics.still_invalid}). "
        f"Wrote {JSONL_OUTPUT} and {MD_OUTPUT}."
    )


if __name__ == "__main__":
    main()
