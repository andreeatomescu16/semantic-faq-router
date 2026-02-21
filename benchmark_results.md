# Benchmark Results

- Timestamp: `2026-02-21T15:01:25.811424+00:00`
- Config: `{"similarity_threshold": 0.65, "similarity_margin": 0.05, "top_k": 5, "hybrid_alpha": 0.7, "hybrid_beta": 0.2, "hybrid_gamma": 0.1, "embedding_model": "text-embedding-3-small", "embedding_version": "v1"}`

| Strategy | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) |
|---|---:|---:|---:|---:|---:|
| cosine_threshold | 0.837 | 0.891 | 0.942 | 0.025 | 0.261 |
| cosine_margin | 0.802 | 0.891 | 0.942 | 0.025 | 0.326 |
| hybrid | 0.767 | 0.891 | 0.940 | 0.050 | 0.370 |
| hybrid_margin | 0.767 | 0.891 | 0.940 | 0.050 | 0.370 |

## Interpretation Notes
- Higher Top1/MRR means better retrieval quality.
- Lower FPR(local) means fewer wrong local answers.
- Margin and hybrid strategies often trade some recall for safer routing.