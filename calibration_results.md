# Calibration Results

- Timestamp: `2026-02-21T14:02:54.231957+00:00`
- Dataset: `data/benchmark_queries.jsonl`
- ROUTING_OBJECTIVE: `cost`
- Cost config: `COST_FP_LOCAL=8.0`, `COST_FN_LOCAL=2.0`, `COST_OPENAI_CALL=0.3`, `COST_FALSE_COMPLIANCE=3.0`
- Hybrid weights: `alpha=0.7`, `beta=0.2`, `gamma=0.1`

## Best Config per Strategy (selected objective)

| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_threshold | 0.68 | 0.00 | 0.814 | 0.848 | 0.899 | 0.000 | 0.326 | 1.000 | 0.674 | 0.805 | 0.674 | 0.360 | 0.372 | 45.60 | cost=45.600 |
| cosine_margin | 0.68 | 0.00 | 0.814 | 0.848 | 0.899 | 0.000 | 0.326 | 1.000 | 0.674 | 0.805 | 0.674 | 0.360 | 0.372 | 45.60 | cost=45.600 |
| hybrid | 0.70 | 0.00 | 0.744 | 0.848 | 0.897 | 0.000 | 0.457 | 1.000 | 0.543 | 0.704 | 0.543 | 0.291 | 0.442 | 59.40 | cost=59.400 |
| hybrid_margin | 0.70 | 0.00 | 0.744 | 0.848 | 0.897 | 0.000 | 0.457 | 1.000 | 0.543 | 0.704 | 0.543 | 0.291 | 0.442 | 59.40 | cost=59.400 |

## Two Recommended Configs

- **Safety-first**: `cosine_threshold` thr `0.68` margin `0.00` (FP_local=0, FPR=0.000)
- **Cost-optimized**: `cosine_threshold` thr `0.68` margin `0.00` (TotalCost=45.60)

## Best by Source Accuracy

| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_threshold | 0.59 | 0.00 | 0.826 | 0.848 | 0.899 | 0.125 | 0.196 | 0.881 | 0.804 | 0.841 | 0.804 | 0.488 | 0.244 | 70.30 | src_acc=0.826 |
| cosine_margin | 0.59 | 0.00 | 0.826 | 0.848 | 0.899 | 0.125 | 0.196 | 0.881 | 0.804 | 0.841 | 0.804 | 0.488 | 0.244 | 70.30 | src_acc=0.826 |
| hybrid | 0.55 | 0.00 | 0.814 | 0.848 | 0.897 | 0.200 | 0.152 | 0.830 | 0.848 | 0.839 | 0.848 | 0.547 | 0.186 | 88.80 | src_acc=0.814 |
| hybrid_margin | 0.55 | 0.00 | 0.814 | 0.848 | 0.897 | 0.200 | 0.152 | 0.830 | 0.848 | 0.839 | 0.848 | 0.547 | 0.186 | 88.80 | src_acc=0.814 |

## Best by F1(local)

| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_threshold | 0.59 | 0.00 | 0.826 | 0.848 | 0.899 | 0.125 | 0.196 | 0.881 | 0.804 | 0.841 | 0.804 | 0.488 | 0.244 | 70.30 | f1_local=0.841 |
| cosine_margin | 0.59 | 0.00 | 0.826 | 0.848 | 0.899 | 0.125 | 0.196 | 0.881 | 0.804 | 0.841 | 0.804 | 0.488 | 0.244 | 70.30 | f1_local=0.841 |
| hybrid | 0.55 | 0.00 | 0.814 | 0.848 | 0.897 | 0.200 | 0.152 | 0.830 | 0.848 | 0.839 | 0.848 | 0.547 | 0.186 | 88.80 | f1_local=0.839 |
| hybrid_margin | 0.55 | 0.00 | 0.814 | 0.848 | 0.897 | 0.200 | 0.152 | 0.830 | 0.848 | 0.839 | 0.848 | 0.547 | 0.186 | 88.80 | f1_local=0.839 |

## Best by Total Cost

| Strategy | Thr | Margin | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) | Precision(local) | Recall(local) | F1(local) | KB util | Local rate | OpenAI rate | TotalCost | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cosine_threshold | 0.68 | 0.00 | 0.814 | 0.848 | 0.899 | 0.000 | 0.326 | 1.000 | 0.674 | 0.805 | 0.674 | 0.360 | 0.372 | 45.60 | cost=45.600 |
| cosine_margin | 0.68 | 0.00 | 0.814 | 0.848 | 0.899 | 0.000 | 0.326 | 1.000 | 0.674 | 0.805 | 0.674 | 0.360 | 0.372 | 45.60 | cost=45.600 |
| hybrid | 0.70 | 0.00 | 0.744 | 0.848 | 0.897 | 0.000 | 0.457 | 1.000 | 0.543 | 0.704 | 0.543 | 0.291 | 0.442 | 59.40 | cost=59.400 |
| hybrid_margin | 0.70 | 0.00 | 0.744 | 0.848 | 0.897 | 0.000 | 0.457 | 1.000 | 0.543 | 0.704 | 0.543 | 0.291 | 0.442 | 59.40 | cost=59.400 |

## Recommended .env settings (selected objective)

### cosine_threshold
```env
SCORING_STRATEGY=cosine_threshold
SIMILARITY_THRESHOLD=0.68
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
ROUTING_OBJECTIVE=cost
```

### cosine_margin
```env
SCORING_STRATEGY=cosine_margin
SIMILARITY_THRESHOLD=0.68
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
ROUTING_OBJECTIVE=cost
```

### hybrid
```env
SCORING_STRATEGY=hybrid
SIMILARITY_THRESHOLD=0.70
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
ROUTING_OBJECTIVE=cost
```

### hybrid_margin
```env
SCORING_STRATEGY=hybrid_margin
SIMILARITY_THRESHOLD=0.70
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
ROUTING_OBJECTIVE=cost
```

## Notes
- Thresholds are empirical; recalibrate when KB content/distribution changes.
- Margin strategies usually reduce false-positive local answers.
- Cost objective balances wrong-local risk with unnecessary OpenAI usage.

Raw sweep results are in `calibration_results.csv`.