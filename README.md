# Semantic FAQ Assistant  
**FastAPI + pgvector + LangChain + Cost-Aware Routing + Offline LLM Judge**

Production-grade semantic FAQ system with deterministic routing, calibrated thresholds, cost-aware optimization, and offline LLM-based qualitative evaluation.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)  
2. [System Overview](#2-system-overview)  
3. [Core Design Principles](#3-core-design-principles)  
4. [Retrieval Layer](#4-retrieval-layer)  
5. [Benchmark Evaluation](#5-benchmark-evaluation)  
6. [Threshold Calibration](#6-threshold-calibration)  
7. [LLM-as-a-Judge (Offline)](#7-llm-as-a-judge-offline)  
8. [Guardrails](#8-guardrails)  
9. [API Contract](#9-api-contract)  
10. [Testing Strategy](#10-testing-strategy)  
11. [Operational Considerations](#11-operational-considerations)  
12. [Limitations](#12-limitations)  
13. [Future Improvements](#13-future-improvements)  
14. [Why This Architecture](#14-why-this-architecture)  
15. [Summary](#15-summary)  
16. [Appendix: Environment Variables](#appendix-environment-variables)  
17. [Appendix: Running Locally & Docker](#appendix-running-locally--docker)  
18. [Appendix: Evaluation Artifacts](#appendix-evaluation-artifacts)  

---

## 1. Problem Statement

The goal of this project is to build a **production-oriented semantic FAQ assistant** that:

- Uses a **local knowledge base** for high-confidence answers
- Falls back to an LLM only when necessary
- Minimizes unsafe or hallucinated answers
- Is explainable, measurable, and tunable
- Is robust against prompt injection
- Supports cost-aware operational optimization

This is not a demo chatbot.  
This is a routing-first architecture where retrieval quality and decision logic are treated as first-class citizens.

---

## 2. System Overview

### High-Level Architecture

```
Client
  ↓
FastAPI /ask-question
  ↓
Auth check (Bearer token)
  ↓
Trace middleware (trace_id)
  ↓
Guardrails (prompt injection patterns)
  ↓
Domain Router (deterministic gate)
    ├── out-of-domain → compliance refusal
    └── in-domain:
          ↓
          Normalize query
          ↓
          Embed query
          ↓
          pgvector topK retrieval
          ↓
          Scoring strategy
              ├── score ≥ threshold → local answer
              └── score < threshold → OpenAI fallback
```

### Project Structure

```
app/
  api/
  core/
  db/
  schemas/
  services/
data/
scripts/
tests/
Dockerfile
docker-compose.yml
requirements.txt
```

---

## 3. Core Design Principles

### 3.1 Deterministic Routing First

LLM usage is not the default; it is a fallback.

**Routing order:**
1. Prompt injection detection  
2. Domain gate (deterministic keyword heuristic)  
3. Retrieval scoring  
4. Threshold-based decision  
5. Guarded OpenAI fallback  

This prevents:
- unnecessary LLM cost
- hallucinated answers
- accidental leakage
- unpredictable responses

---

### 3.2 Local-First Safety Philosophy

Local answers are:
- deterministic
- auditable
- version-controlled
- explainable

OpenAI fallback is:
- guarded
- constrained by system prompt
- invoked only when confidence is insufficient

---

### 3.3 Cost-Aware Deployment Mindset

Routing is optimized not only for accuracy but also for:

- Wrong local answers (high risk)
- Missed local answers (UX degradation)
- OpenAI invocation cost
- False compliance refusals

This is enforced via calibration objective functions.

---

## 4. Retrieval Layer

### 4.1 Embeddings

Model:
```
text-embedding-3-small
embedding_version = v1
```

Embeddings are:
- Stored in PostgreSQL
- Indexed with pgvector
- Queried using cosine similarity

TopK default:
```
TOP_K = 5
```

---

### 4.2 Scoring Strategies

Supported strategies:

#### 1) `cosine_threshold`
```
confidence = top cosine score
accept if confidence >= SIMILARITY_THRESHOLD
```

#### 2) `cosine_margin`
Requires:
```
(best - second_best) >= SIMILARITY_MARGIN
```

#### 3) `hybrid`
Re-ranks topK:
```
score = alpha * cosine
      + beta  * lexical_jaccard
      + gamma * category_boost
```

#### 4) `hybrid_margin`
Hybrid + margin requirement.

---

## 5. Benchmark Evaluation

Run:

```bash
python scripts/benchmark_retrieval.py
```

This evaluates:
- **Top1 Accuracy** (in-domain positives)
- **MRR** (retrieval ranking quality)
- **Source Accuracy** (routing correctness across all samples)
- **FPR(local)** (wrong local answers; lower is safer)
- **FNR(local)** (missed local answers; lower improves KB utilization)

OpenAI fallback is disabled in benchmark mode (routing decision only).

---

### Benchmark Results (Provided Run)

**Config:**
```json
{
  "similarity_threshold": 0.65,
  "similarity_margin": 0.05,
  "top_k": 5,
  "hybrid_alpha": 0.7,
  "hybrid_beta": 0.2,
  "hybrid_gamma": 0.1,
  "embedding_model": "text-embedding-3-small",
  "embedding_version": "v1"
}
```

| Strategy | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) |
|---|---:|---:|---:|---:|---:|
| cosine_threshold | 0.837 | 0.891 | 0.942 | 0.025 | 0.261 |
| cosine_margin | 0.802 | 0.891 | 0.942 | 0.025 | 0.326 |
| hybrid | 0.767 | 0.891 | 0.940 | 0.050 | 0.370 |
| hybrid_margin | 0.767 | 0.891 | 0.940 | 0.050 | 0.370 |

---

### Interpretation

- Retrieval quality is strong (**Top1=0.891, MRR=0.942**).
- Margin strategies reduce risky local matches but increase missed local recall.
- Hybrid did not outperform cosine under this dataset distribution.
- The main bottleneck is **routing threshold calibration**, not retrieval ranking.

---

## 6. Threshold Calibration

Instead of manually picking thresholds:

```bash
python -m scripts.calibrate_routing
```

Calibration:
- loads `data/benchmark_queries.jsonl`
- precomputes domain routing + topK retrieval once per query
- sweeps thresholds/margins for each strategy
- optimizes selected objective (`cost` by default)

---

### 6.1 Cost Function

```
TotalCost =
  COST_FP_LOCAL * FP_local
+ COST_FN_LOCAL * FN_local
+ COST_OPENAI_CALL * OpenAI_calls
+ COST_FALSE_COMPLIANCE * False_compliance
```

Default cost config:

```
COST_FP_LOCAL=8.0
COST_FN_LOCAL=2.0
COST_OPENAI_CALL=0.3
COST_FALSE_COMPLIANCE=3.0
ROUTING_OBJECTIVE=cost
```

---

### 6.2 Calibration Results (Cost Objective)

**Best by Total Cost:**

| Strategy | Thr | Margin | Source Acc | FPR(local) | FNR(local) | TotalCost |
|---|---:|---:|---:|---:|---:|---:|
| cosine_threshold | 0.68 | 0.00 | 0.814 | 0.000 | 0.326 | 45.60 |
| cosine_margin | 0.68 | 0.00 | 0.814 | 0.000 | 0.326 | 45.60 |

**Recommended production `.env`:**
```env
SCORING_STRATEGY=cosine_threshold
SIMILARITY_THRESHOLD=0.68
SIMILARITY_MARGIN=0.00
HYBRID_ALPHA=0.70
HYBRID_BETA=0.20
HYBRID_GAMMA=0.10
ROUTING_OBJECTIVE=cost
```

---

### Why threshold 0.68?

This is a deliberate **safety-first** tradeoff:
- **FPR(local)=0.0** → no wrong local answers (high value in production)
- **FNR(local)=0.326** → some misses (acceptable, paid by fallback)
- Total operational cost minimized under the chosen cost weights

---

## 7. LLM-as-a-Judge (Offline)

This project includes an **offline-only** qualitative evaluator:

```bash
python -m scripts.llm_judge_eval
```

The judge sees:
- user query
- topK candidate metadata
- predicted route and selected index (if any)
- expected_source label (for evaluation)

The judge outputs:
- preferred_source
- preferred_index (only if local)
- severity
- rationale

---

### 7.1 Judge Report Summary (Provided Run)

- Judge model: `gpt-5-mini`
- Cases: 50/50
- Coverage: 1.000
- Source alignment (overall): 0.720
- Gray-zone alignment: 0.615

**Alignment by predicted_source**
- `local`: 0.800
- `openai`: 0.500
- `compliance`: 0.952

---

### 7.2 Interpretation

The dominant failure mode is:

> Over-routing to **OpenAI** for medium-confidence questions that already have a good local match.

This appears especially in:
- password reset / auth flows
- email changes
- invoices / refunds

Typical pattern:
- topK[0] is correct
- confidence ~ 0.55–0.65
- threshold rejects local → route openai

This aligns perfectly with the calibration tradeoff:
- higher threshold → fewer wrong locals (safety)
- but more fallbacks → lower local recall

Judge evidence supports that the model is **conservative**, which is usually desirable for production.

---

## 8. Guardrails

### 8.1 Prompt Injection Detection

Blocks common injection patterns, e.g.
- “ignore previous instructions”
- “reveal system prompt”
- “show me your API key”
- “print hidden policies”

If triggered → compliance refusal.

---

### 8.2 Out-of-Domain Refusal

Domain gate uses deterministic keyword heuristics over categories:
- account/profile/security/billing/subscription/notifications/privacy/troubleshooting/developer

If out-of-domain, the answer is exactly:

```
This is not really what I was trained for, therefore I cannot answer. Try again.
```

This is intentionally fixed for predictable behavior and easy testing.

---

### 8.3 Safe OpenAI Fallback

OpenAI fallback:
- invoked only when local confidence is insufficient
- uses a strict system prompt to prevent leakage
- OpenAI errors are caught and returned as safe responses

---

## 9. API Contract

### Endpoints

- `GET /healthz` — liveness  
- `GET /readyz` — readiness  
- `POST /ask-question` — main endpoint  

---

### POST `/ask-question`

Request:
```json
{ "user_question": "How do I reset my password?" }
```

Response:
```json
{
  "source": "local | openai | compliance",
  "matched_question": "string | N/A",
  "category": "string | N/A",
  "confidence": 0.0,
  "answer": "string",
  "trace_id": "uuid"
}
```

---

## 10. Testing Strategy

Run:

```bash
pytest -q
```

Coverage includes:
- Unit tests: normalization, hashing, domain routing
- Integration tests: local match, compliance refusal
- OpenAI fallback tests using mocks (no real calls)
- Adversarial tests: prompt injection refused and OpenAI not called

---

## 11. Operational Considerations

### What to monitor in production

- Local/OpenAI/compliance distribution
- Fallback rate vs threshold
- Latency p50/p95 per source
- OpenAI token usage + spend
- Compliance refusal rate (should be low for normal traffic)
- Manual audit sampling for local correctness

### When to recalibrate

Re-run calibration when:
- KB changes significantly (new FAQs, rewrites)
- distribution shifts (more billing/security, fewer profile, etc.)
- embedding model changes
- business risk tolerance changes (cost weights)

---

## 12. Limitations

- Domain gate is heuristic keyword-based (can misclassify edge cases)
- No cross-encoder reranker (could improve borderline retrieval)
- No semantic cache for OpenAI fallback
- Compliance routing is rule-based, not policy-model based

---

## 13. Future Improvements

- Add cross-encoder reranking for top3 candidates
- Add semantic caching for OpenAI fallback responses
- Add drift detection on embedding distributions
- Add structured logging + dashboard
- Add confidence calibration (Platt scaling / isotonic regression)

---

## 14. Why This Architecture

This system is intentionally:

- deterministic-first
- measurable
- cost-aware
- safety-aligned
- tunable via calibration
- explainable via benchmark + judge

It avoids:
- blind LLM usage
- hallucination-first architecture
- unmeasured routing decisions
- hidden cost growth

---

## 15. Summary

This project demonstrates:
- Semantic retrieval with pgvector
- Deterministic routing & safety gates
- Threshold calibration with cost-aware objective
- Prompt-injection defenses
- Offline LLM judge for qualitative routing analysis
- Production-minded tradeoffs (safety vs recall vs cost)

It’s not just an FAQ bot.  
It’s a calibrated semantic routing system.

---

## Appendix: Environment Variables

Copy `.env.example` → `.env`.

Required:
- `OPENAI_API_KEY` (embeddings + fallback)
- `API_AUTH_TOKEN` (Bearer token for endpoint)
- `DATABASE_URL`

Routing:
- `SIMILARITY_THRESHOLD` (default `0.82`, calibrated recommended `0.68`)
- `SIMILARITY_MARGIN` (default `0.05`)
- `TOP_K` (default `5`)
- `SCORING_STRATEGY` (`cosine_threshold`, `cosine_margin`, `hybrid`, `hybrid_margin`)

Hybrid:
- `HYBRID_ALPHA`, `HYBRID_BETA`, `HYBRID_GAMMA` (default `0.7/0.2/0.1`)

Calibration / evaluation:
- `COST_FP_LOCAL`, `COST_FN_LOCAL`, `COST_OPENAI_CALL`, `COST_FALSE_COMPLIANCE`
- `ROUTING_OBJECTIVE` in `cost | source_acc | f1_local`
- `JUDGE_MAX_CASES`

---

## Appendix: Web UI

The application includes a built-in chat-like UI for testing the assistant interactively.

After starting the stack, open:

```
http://localhost:8000
```

Features:
- Enter your `API_AUTH_TOKEN` and ask any question in natural language
- Response shows the **source** (`local` / `openai` / `compliance`), **confidence score**, **matched question**, and **trace ID**
- Quick-fill buttons for common test scenarios (profile update, out-of-domain, OpenAI fallback)
- "Copy answer" button for easy sharing

No extra setup needed — the UI is served directly by FastAPI at `GET /`.

---

## Appendix: Running Locally & Docker

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

### Initialize DB and ingest KB

```bash
python scripts/init_db.py
python scripts/ingest_kb.py --kb-path data/kb_items.json
python scripts/reembed_incremental.py
```

Or:

```bash
make init-db
make ingest-kb
make reembed
```

---

## Appendix: Evaluation Artifacts

### Benchmark
- Input: `data/benchmark_queries.jsonl`
- Embedding cache: `data/benchmark_query_embeddings_cache.json`
- Output: `benchmark_results.md`

### Calibration
- Output sweep: `calibration_results.csv`
- Summary: `calibration_results.md`

### LLM Judge (offline)
- Raw: `llm_judge_results.jsonl`
- Summary: `llm_judge_report.md`
