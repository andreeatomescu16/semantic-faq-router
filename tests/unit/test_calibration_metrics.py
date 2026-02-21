from app.services.calibration import (
    compute_local_metrics,
    objective_f1_local,
    objective_routing_cost,
    objective_source_accuracy,
)


def test_compute_local_metrics_deterministic() -> None:
    metrics = compute_local_metrics(
        source_correct=8,
        source_total=10,
        positive_total=4,
        top1_hits=3,
        mrr_sum=3.5,
        tp_local=3,
        fp_local=1,
        fn_local=1,
        tn_local=5,
        predicted_local=4,
        predicted_openai=3,
        predicted_compliance=3,
        false_compliance=1,
        cost_fp_local=8.0,
        cost_fn_local=2.0,
        cost_openai_call=0.3,
        cost_false_compliance=3.0,
    )

    assert metrics.source_accuracy == 0.8
    assert metrics.top1_accuracy == 0.75
    assert metrics.mrr == 0.875
    assert metrics.precision_local == 0.75
    assert metrics.recall_local == 0.75
    assert metrics.f1_local == 0.75
    assert metrics.fpr_local == 1 / 6
    assert metrics.fnr_local == 0.25
    assert metrics.kb_utilization == 0.75
    assert metrics.local_rate == 0.4
    assert metrics.openai_rate == 0.3
    assert round(metrics.total_cost, 3) == 13.9


def test_objectives_behaviour() -> None:
    metrics = compute_local_metrics(
        source_correct=7,
        source_total=10,
        positive_total=2,
        top1_hits=1,
        mrr_sum=1.0,
        tp_local=2,
        fp_local=1,
        fn_local=2,
        tn_local=5,
        predicted_local=3,
        predicted_openai=5,
        predicted_compliance=2,
        false_compliance=1,
        cost_fp_local=8.0,
        cost_fn_local=2.0,
        cost_openai_call=0.3,
        cost_false_compliance=3.0,
    )

    assert objective_source_accuracy(metrics) == 0.7
    assert round(objective_f1_local(metrics), 4) == 0.5714
    assert round(objective_routing_cost(metrics), 3) == 16.5
