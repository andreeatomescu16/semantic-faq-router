import pytest

from app.services.llm_judge import (
    EvalRecord,
    JudgeExecutionResult,
    JudgeOutputParseError,
    LLMJudgeClient,
    JudgeVerdict,
    extract_first_json_object,
    parse_and_validate_judge_output,
    parse_judge_output,
)
from scripts.llm_judge_eval import JudgeRunMetrics, judge_records, should_judge_case


def _record(
    identifier: str,
    *,
    predicted_source: str = "local",
    expected_source: str | None = "local",
    confidence: float | None = 0.6,
) -> EvalRecord:
    return EvalRecord(
        id=identifier,
        query="How do I update profile?",
        expected_source=expected_source,
        predicted_source=predicted_source,  # type: ignore[arg-type]
        confidence=confidence,
        selected_index=0 if predicted_source == "local" else None,
        selected_question="How do I change my profile information?"
        if predicted_source == "local"
        else None,
        top_k=[],
    )


def test_parse_judge_output_valid_json() -> None:
    parsed = parse_judge_output(
        """
        {
          "preferred_source": "local",
          "preferred_index": 0,
          "severity": "minor",
          "rationale": "The top candidate clearly matches."
        }
        """
    )
    assert isinstance(parsed, JudgeVerdict)
    assert parsed.preferred_source == "local"


def test_parse_judge_output_invalid_json_raises() -> None:
    with pytest.raises(JudgeOutputParseError):
        parse_judge_output('{"preferred_source":"local"}')


def test_parse_judge_output_leading_prose_is_extracted() -> None:
    parsed = parse_and_validate_judge_output(
        'Here is the result: {"preferred_source":"local","preferred_index":0,'
        '"severity":"minor","rationale":"Looks correct."}'
    )
    assert parsed.preferred_source == "local"


def test_parse_judge_output_trailing_prose_is_extracted() -> None:
    parsed = parse_and_validate_judge_output(
        '{"preferred_source":"openai","preferred_index":null,'
        '"severity":"major","rationale":"Local candidate is weak."} Extra text after JSON.'
    )
    assert parsed.preferred_source == "openai"


def test_extract_first_json_object_returns_none_without_object() -> None:
    assert extract_first_json_object("no json here") is None


class _FakeJudgeClient:
    def __init__(self) -> None:
        self.calls = 0

    def judge_with_repair(self, record: EvalRecord) -> JudgeExecutionResult:
        self.calls += 1
        if record.id == "bad":
            return JudgeExecutionResult(
                verdict=None,
                computed_verdict="disagree",
                raw_text="bad output",
                repaired=False,
                valid_first_pass=False,
            )
        parsed = JudgeVerdict(
            preferred_source=record.predicted_source,  # type: ignore[arg-type]
            preferred_index=record.selected_index,
            severity="minor",
            rationale="Looks fine.",
        )
        return JudgeExecutionResult(
            verdict=parsed,
            computed_verdict="agree",
            raw_text=parsed.model_dump_json(),
            repaired=False,
            valid_first_pass=True,
        )


def test_invalid_judge_output_is_skipped() -> None:
    judge = _FakeJudgeClient()
    records = [_record("bad"), _record("good")]
    judged, metrics = judge_records(
        records,
        judge_client=judge,
        max_cases=50,
        gray_low=0.55,
        gray_high=0.70,
        similarity_threshold=0.65,
    )
    assert metrics.still_invalid == 1
    assert metrics.attempted_cases == 2
    assert len(judged) == 1
    assert judged[0].eval_record.id == "good"
    assert judged[0].computed_verdict == "agree"


def test_should_judge_case_criteria() -> None:
    assert not should_judge_case(
        _record("local", predicted_source="local", expected_source="local", confidence=0.9),
        gray_low=0.55,
        gray_high=0.70,
        similarity_threshold=0.65,
    )
    assert should_judge_case(
        _record("gray", predicted_source="openai", confidence=0.60),
        gray_low=0.55,
        gray_high=0.70,
        similarity_threshold=0.65,
    )


def test_repair_does_not_fabricate_missing_fields() -> None:
    client = object.__new__(LLMJudgeClient)

    def _fake_invoke(system_prompt: str, user_prompt: str, *, repair: bool = False) -> str:
        del system_prompt, user_prompt
        assert repair is True
        return '{"preferred_source":"local"}'

    client._invoke = _fake_invoke  # type: ignore[attr-defined]
    repaired = client.repair_json_if_needed("garbage output", _record("repair-bad"))
    assert repaired is None


def test_judge_run_metrics_coverage() -> None:
    metrics = JudgeRunMetrics(
        attempted_cases=10,
        valid_first_pass=6,
        repaired_successfully=2,
        still_invalid=2,
    )
    assert metrics.total_valid == 8
    assert metrics.coverage == 0.8
