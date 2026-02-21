# LLM Judge Report

- Timestamp: `2026-02-21T15:19:59.424274+00:00`
- Judge model: `gpt-5-mini`
- JUDGE_MAX_CASES: `50`
- Attempted cases: `50`
- Judged cases: `50`
- Skipped invalid judge outputs: `0`
- Estimated judge cost (calls * COST_OPENAI_CALL): `15.00`

## Coverage Metrics
- Attempted: `50`
- Valid first pass: `50`
- Repaired successfully: `0`
- Still invalid: `0`
- Coverage: `1.000`

## Source Alignment Metrics
- Source alignment (valid only): `0.720`
- Source alignment (overall): `0.720`
- Gray-zone source alignment: `0.615`
- Judge verdict='agree' rate (valid only): `0.720`

## Source Alignment by predicted_source
- `local`: `0.800` (4/5)
- `openai`: `0.500` (12/24)
- `compliance`: `0.952` (20/21)

## Top Disagreement Reasons
- `expected_source mandates local and top_k[0] directly matches the password reset query.`: 1
- `expected_source is local and top_k[0] exactly matches the query about resetting a password.`: 1
- `Record specifies expected_source 'local' and the top candidate 'Change login email' (index 0) directly matches the query.`: 1
- `expected_source mandates local and top_k[1] ('Are there any guidelines on setting a strong password?') most closely matches the query about what makes a strong pass.`: 1
- `Expected source is local and the top candidate directly matches adding a passkey, so route to local index 0.`: 1
- `expected_source requires local and top_k index 2 directly matches the password reset email not received query.`: 1
- `expected_source requires local and the top candidate 'Edit avatar?' directly matches uploading a profile photo.`: 1
- `expected_source requires local and the top candidate 'Verify my email address' clearly matches the query about re-sending a verification email.`: 1
- `expected_source mandates local and the top candidate 'Where can I download invoices?' clearly matches the query about billing invoices.`: 1
- `Record specifies expected_source local and the top candidate (index 0) directly matches the refund query.`: 1

## Representative Examples (up to 10)

| id | predicted | preferred | severity | confidence | rationale |
|---|---|---|---|---:|---|
| q005 | openai | local | major | 0.587 | expected_source mandates local and top_k[0] directly matches the password reset query. |
| q006 | openai | local | major | 0.563 | expected_source is local and top_k[0] exactly matches the query about resetting a password. |
| q008 | local | local | minor | 0.683 | expected_source forces local routing and the top candidate (index 0) exactly matches the query about changing the registered email address. |
| q009 | openai | local | minor | 0.648 | Record specifies expected_source 'local' and the top candidate 'Change login email' (index 0) directly matches the query. |
| q012 | openai | local | minor | 0.412 | expected_source mandates local and top_k[1] ('Are there any guidelines on setting a strong password?') most closely matches the query about what makes a strong pass. |
| q014 | local | local | minor | 0.689 | expected_source is local and the top candidate (index 0) clearly matches the query to enable two-factor authentication. |
| q018 | openai | local | major | 0.594 | Expected source is local and the top candidate directly matches adding a passkey, so route to local index 0. |
| q020 | openai | local | minor | 0.572 | expected_source requires local and top_k index 2 directly matches the password reset email not received query. |
| q022 | local | local | minor | 0.664 | expected_source is local and the top candidate at index 0 clearly matches the query. |
| q024 | openai | local | minor | 0.501 | expected_source requires local and the top candidate 'Edit avatar?' directly matches uploading a profile photo. |

Raw judged outputs are in `llm_judge_results.jsonl`.
This judge is offline-only and not used in production serving.