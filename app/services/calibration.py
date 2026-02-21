from dataclasses import dataclass


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class EvaluationMetrics:
    source_accuracy: float
    top1_accuracy: float
    mrr: float
    fpr_local: float
    fnr_local: float
    precision_local: float
    recall_local: float
    f1_local: float
    tp_local: int
    fp_local: int
    fn_local: int
    tn_local: int
    predicted_local: int
    predicted_openai: int
    predicted_compliance: int
    false_compliance: int
    kb_utilization: float
    local_rate: float
    openai_rate: float
    total_cost: float


def compute_local_metrics(
    *,
    source_correct: int,
    source_total: int,
    positive_total: int,
    top1_hits: int,
    mrr_sum: float,
    tp_local: int,
    fp_local: int,
    fn_local: int,
    tn_local: int,
    predicted_local: int,
    predicted_openai: int,
    predicted_compliance: int,
    false_compliance: int,
    cost_fp_local: float,
    cost_fn_local: float,
    cost_openai_call: float,
    cost_false_compliance: float,
) -> EvaluationMetrics:
    source_accuracy = _safe_div(source_correct, source_total)
    top1_accuracy = _safe_div(top1_hits, positive_total)
    mrr = _safe_div(mrr_sum, positive_total)

    expected_local_total = tp_local + fn_local
    expected_not_local_total = fp_local + tn_local

    fpr_local = _safe_div(fp_local, expected_not_local_total)
    fnr_local = _safe_div(fn_local, expected_local_total)

    precision_local = _safe_div(tp_local, tp_local + fp_local)
    recall_local = _safe_div(tp_local, tp_local + fn_local)
    f1_local = _safe_div(2 * precision_local * recall_local, precision_local + recall_local)
    kb_utilization = _safe_div(tp_local, positive_total)
    local_rate = _safe_div(predicted_local, source_total)
    openai_rate = _safe_div(predicted_openai, source_total)
    total_cost = (
        cost_fp_local * fp_local
        + cost_fn_local * fn_local
        + cost_openai_call * predicted_openai
        + cost_false_compliance * false_compliance
    )

    return EvaluationMetrics(
        source_accuracy=source_accuracy,
        top1_accuracy=top1_accuracy,
        mrr=mrr,
        fpr_local=fpr_local,
        fnr_local=fnr_local,
        precision_local=precision_local,
        recall_local=recall_local,
        f1_local=f1_local,
        tp_local=tp_local,
        fp_local=fp_local,
        fn_local=fn_local,
        tn_local=tn_local,
        predicted_local=predicted_local,
        predicted_openai=predicted_openai,
        predicted_compliance=predicted_compliance,
        false_compliance=false_compliance,
        kb_utilization=kb_utilization,
        local_rate=local_rate,
        openai_rate=openai_rate,
        total_cost=total_cost,
    )


def objective_source_accuracy(metrics: EvaluationMetrics) -> float:
    return metrics.source_accuracy


def objective_f1_local(metrics: EvaluationMetrics) -> float:
    return metrics.f1_local


def objective_routing_cost(
    metrics: EvaluationMetrics,
) -> float:
    return metrics.total_cost
