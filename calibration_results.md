# Calibration Results

- Timestamp: `2026-02-19T14:03:50.945972+00:00`
- Dataset: `data/benchmark_queries.jsonl`
- Cost config: `COST_FP_LOCAL=5.0`, `COST_FN_LOCAL=1.0`
- Hybrid weights: `alpha=0.7`, `beta=0.2`, `gamma=0.1`

## Best by Source Accuracy

| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_threshold | 0.54 | 0.00 | 0.818 | 0.902 | 0.947 | 0.250 | 0.130 | 0.889 | 0.870 | 0.879 | src_acc=0.818 |
| cosine_margin | 0.54 | 0.00 | 0.818 | 0.902 | 0.947 | 0.250 | 0.130 | 0.889 | 0.870 | 0.879 | src_acc=0.818 |
| hybrid | 0.53 | 0.00 | 0.803 | 0.927 | 0.957 | 0.100 | 0.217 | 0.947 | 0.783 | 0.857 | src_acc=0.803 |
| hybrid_margin | 0.53 | 0.00 | 0.803 | 0.927 | 0.957 | 0.100 | 0.217 | 0.947 | 0.783 | 0.857 | src_acc=0.803 |

## Best by F1(local)

| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_threshold | 0.54 | 0.00 | 0.818 | 0.902 | 0.947 | 0.250 | 0.130 | 0.889 | 0.870 | 0.879 | f1_local=0.879 |
| cosine_margin | 0.54 | 0.00 | 0.818 | 0.902 | 0.947 | 0.250 | 0.130 | 0.889 | 0.870 | 0.879 | f1_local=0.879 |
| hybrid | 0.53 | 0.00 | 0.803 | 0.927 | 0.957 | 0.100 | 0.217 | 0.947 | 0.783 | 0.857 | f1_local=0.857 |
| hybrid_margin | 0.53 | 0.00 | 0.803 | 0.927 | 0.957 | 0.100 | 0.217 | 0.947 | 0.783 | 0.857 | f1_local=0.857 |

## Best by Routing Cost (minimize)

| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_threshold | 0.65 | 0.00 | 0.773 | 0.902 | 0.947 | 0.000 | 0.304 | 1.000 | 0.696 | 0.821 | cost=14.000 |
| cosine_margin | 0.65 | 0.00 | 0.773 | 0.902 | 0.947 | 0.000 | 0.304 | 1.000 | 0.696 | 0.821 | cost=14.000 |
| hybrid | 0.59 | 0.00 | 0.758 | 0.927 | 0.957 | 0.000 | 0.326 | 1.000 | 0.674 | 0.805 | cost=15.000 |
| hybrid_margin | 0.59 | 0.00 | 0.758 | 0.927 | 0.957 | 0.000 | 0.326 | 1.000 | 0.674 | 0.805 | cost=15.000 |

## Recommended .env settings (cost-optimized)

### cosine_threshold
```env
SCORING_STRATEGY=cosine_threshold
SIMILARITY_THRESHOLD=0.65
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
```

### cosine_margin
```env
SCORING_STRATEGY=cosine_margin
SIMILARITY_THRESHOLD=0.65
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
```

### hybrid
```env
SCORING_STRATEGY=hybrid
SIMILARITY_THRESHOLD=0.59
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
```

### hybrid_margin
```env
SCORING_STRATEGY=hybrid_margin
SIMILARITY_THRESHOLD=0.59
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
```

## Notes
- Thresholds are empirical; recalibrate when KB content/distribution changes.
- Margin strategies usually reduce false-positive local answers.
- Cost objective is useful when wrong local answers are much more expensive than fallback.

Raw sweep results are in `calibration_results.csv`.