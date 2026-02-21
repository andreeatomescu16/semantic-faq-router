import pytest
from pydantic import ValidationError

from app.services.llm_judge import EvalCandidate, EvalRecord, JudgeVerdict, LLMJudgeClient


def _record() -> EvalRecord:
    return EvalRecord(
        id="r1",
        query="reset password",
        expected_source="local",
        predicted_source="local",
        confidence=0.7,
        selected_index=0,
        selected_question="What steps do I take to reset my password?",
        top_k=[
            EvalCandidate(
                index=0,
                question="What steps do I take to reset my password?",
                category="security",
                score=0.9,
            )
        ],
    )


def test_preferred_index_none_for_non_local() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(
            preferred_source="openai",
            preferred_index=0,
            severity="minor",
            rationale="invalid",
        )


def test_preferred_index_negative_fails() -> None:
    with pytest.raises(ValidationError):
        JudgeVerdict(
            preferred_source="local",
            preferred_index=-1,
            severity="minor",
            rationale="invalid",
        )


def test_preferred_index_out_of_range_fails_in_execution() -> None:
    client = object.__new__(LLMJudgeClient)
    parsed = JudgeVerdict(
        preferred_source="local",
        preferred_index=2,
        severity="minor",
        rationale="out of range",
    )
    record = _record()
    result = client._apply_ground_truth_and_validate(record=record, parsed=parsed)  # type: ignore[attr-defined]
    assert result is None
