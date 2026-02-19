# Benchmark Results

- Timestamp: `2026-02-19T13:43:38.568100+00:00`
- Config: `{"similarity_threshold": 0.82, "similarity_margin": 0.05, "top_k": 5, "hybrid_alpha": 0.7, "hybrid_beta": 0.2, "hybrid_gamma": 0.1, "embedding_model": "text-embedding-3-small", "embedding_version": "v1"}`

| Strategy | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) |
|---|---:|---:|---:|---:|---:|
| cosine_threshold | 0.439 | 0.902 | 0.947 | 0.000 | 0.756 |
| cosine_margin | 0.439 | 0.902 | 0.947 | 0.000 | 0.756 |
| hybrid | 0.364 | 0.902 | 0.945 | 0.000 | 0.878 |
| hybrid_margin | 0.364 | 0.902 | 0.945 | 0.000 | 0.878 |

## Interpretation Notes
- Higher Top1/MRR means better retrieval quality.
- Lower FPR(local) means fewer wrong local answers.
- Margin and hybrid strategies often trade some recall for safer routing.