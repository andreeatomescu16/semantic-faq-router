# Semantic FAQ Assistant (FastAPI + pgvector + LangChain)

Production-grade semantic FAQ service with:
- Local FAQ knowledge base in PostgreSQL
- Embeddings + pgvector cosine similarity retrieval
- Deterministic semantic router (domain gate + threshold routing)
- Guarded OpenAI fallback via LangChain
- Prompt injection defenses and compliance refusal

## Architecture

```
Client -> FastAPI /ask-question
          -> Auth check (Bearer token)
          -> Trace middleware (trace_id)
          -> Guardrails (prompt injection patterns)
          -> Domain Router (in-scope gate)
              -> out-of-domain => compliance refusal
              -> in-domain:
                  -> normalize query
                  -> embed query
                  -> pgvector topK nearest search
                  -> if score >= threshold => local KB answer
                  -> else => guarded OpenAI fallback
```

## Project Structure

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

## Environment

Copy `.env.example` to `.env` and set real values:

- `OPENAI_API_KEY` (required for embeddings and fallback LLM)
- `API_AUTH_TOKEN` (required for `/ask-question`)
- `DATABASE_URL`
- `SIMILARITY_THRESHOLD` (default `0.82`)
- `SIMILARITY_MARGIN` (default `0.05`)
- `TOP_K` (default `5`)
- `SCORING_STRATEGY` (`cosine_threshold`, `cosine_margin`, `hybrid`, `hybrid_margin`)
- `HYBRID_ALPHA`, `HYBRID_BETA`, `HYBRID_GAMMA` (default `0.7/0.2/0.1`)

No secrets are hardcoded.

## Run with Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

## Initialize DB and Ingest KB

```bash
python scripts/init_db.py
python scripts/ingest_kb.py --kb-path data/kb_items.json
python scripts/reembed_incremental.py
```

Or with Make:

```bash
make init-db
make ingest-kb
make reembed
```

## API

- `GET /healthz` - liveness
- `GET /readyz` - readiness
- `POST /ask-question`

Request:

```json
{ "user_question": "How do I reset my password?" }
```

Response shape:

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

If out-of-domain, response answer is exactly:

`This is not really what I was trained for, therefore I cannot answer. Try again.`

## Routing Logic

1. Prompt injection detector first; if matched => compliance refusal.
2. Domain gate using deterministic keyword heuristics:
   - account/profile/security/billing/subscription/notifications/privacy/troubleshooting/developer
3. If in-domain:
   - Retrieve best local semantic match from pgvector.
   - If `best_score >= SIMILARITY_THRESHOLD`: return local answer.
   - Otherwise fallback to guarded OpenAI via LangChain.

### Scoring Strategies

- `cosine_threshold`: baseline using top cosine score as confidence.
- `cosine_margin`: baseline + requires `(best - second) >= SIMILARITY_MARGIN`.
- `hybrid`: reranks topK with `alpha*cosine + beta*lexical_jaccard + gamma*category_boost`.
- `hybrid_margin`: hybrid + margin requirement.

## Guardrails

- Prompt injection phrase detector (e.g., "ignore previous instructions", "reveal system prompt", "api key").
- Out-of-domain compliance refusal.
- Safe fallback prompt instructs no secret disclosure.
- OpenAI errors are caught and converted to safe response.

## Tests

Run:

```bash
pytest -q
```

Coverage includes:
- Unit: normalization, hashing, domain routing
- Integration: local match, compliance fallback, OpenAI fallback (mocked)
- Adversarial: prompt injection refused and OpenAI not called

## Benchmark

Use the benchmark to compare strategies with objective metrics:

```bash
python scripts/benchmark_retrieval.py
```

Inputs and outputs:
- Dataset: `data/benchmark_queries.jsonl` (positives, hard negatives, and OOD examples)
- Cache: `data/benchmark_query_embeddings_cache.json` (query embeddings reused across runs)
- Report: `benchmark_results.md`

Metrics:
- **Top1 Accuracy** and **MRR** on positive samples
- **Source Accuracy** on all samples
- **FPR(local)** lower is better (fewer wrong local answers)
- **FNR(local)** lower is better (fewer missed local answers)

OpenAI fallback is disabled in benchmark mode: routing records `"openai"` as a decision only; no chat completion calls are made.

Example output:

```text
| Strategy | Source Acc | Top1 Acc | MRR | FPR(local) | FNR(local) |
|---|---:|---:|---:|---:|---:|
| cosine_threshold | 0.879 | 0.804 | 0.862 | 0.143 | 0.087 |
| cosine_margin | 0.909 | 0.804 | 0.862 | 0.071 | 0.152 |
| hybrid | 0.924 | 0.848 | 0.891 | 0.071 | 0.109 |
| hybrid_margin | 0.939 | 0.848 | 0.891 | 0.036 | 0.152 |
```

Tuning guidance:
- Raise `SIMILARITY_THRESHOLD` or `SIMILARITY_MARGIN` to reduce false-positive local answers.
- Increase `HYBRID_BETA` to favor lexical overlap; increase `HYBRID_GAMMA` to favor domain-category alignment.
- Re-run benchmark after every tuning change and compare `benchmark_results.md`.

## Threshold Calibration

Use calibration to derive routing thresholds from data (instead of hand-picking).

Run:

```bash
python -m scripts.calibrate_routing
```

What it does:
- loads `data/benchmark_queries.jsonl`
- precomputes domain routing + topK retrieval once per query
- sweeps thresholds/margins for:
  - `cosine_threshold`
  - `cosine_margin`
  - `hybrid`
  - `hybrid_margin`
- optimizes three objectives:
  - maximize Source Accuracy
  - maximize F1(local)
  - minimize routing cost: `COST_FP_LOCAL * FP_local + COST_FN_LOCAL * FN_local`

Env knobs:
- `COST_FP_LOCAL` (default `5`)
- `COST_FN_LOCAL` (default `1`)

Artifacts:
- `calibration_results.csv` (all tried configs + metrics)
- `calibration_results.md` (best configs + recommended `.env` blocks)

Recalibrate whenever KB content changes significantly (new items, rewrites, or changed category distribution).

## Minimal Evaluation Plan

Track these metrics on a curated eval set:
- **Retrieval quality**: Top-1 Accuracy, MRR for in-domain questions
- **Router quality**: precision/recall for in-domain vs out-of-domain gate
- **Safety**: prompt injection refusal pass rate
- **Latency**: p50/p95 end-to-end response time by source (`local`, `openai`, `compliance`)