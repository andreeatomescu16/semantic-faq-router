import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.core.config import Settings


class EvalCandidate(BaseModel):
    index: int
    question: str
    category: str
    score: float


class EvalRecord(BaseModel):
    id: str
    query: str
    expected_source: str | None
    predicted_source: Literal["local", "openai", "compliance"]
    confidence: float | None
    selected_index: int | None
    selected_question: str | None
    top_k: list[EvalCandidate]


class JudgeVerdict(BaseModel):
    preferred_source: Literal["local", "openai", "compliance"]
    preferred_index: int | None
    severity: Literal["minor", "major"]
    rationale: str

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_index_consistency(self) -> "JudgeVerdict":
        if self.preferred_source != "local" and self.preferred_index is not None:
            raise ValueError("preferred_index must be null when preferred_source is not local.")
        if self.preferred_source == "local" and self.preferred_index is not None:
            if self.preferred_index < 0:
                raise ValueError("preferred_index must be >= 0 for local preferred_source.")
        return self


class JudgeOutputParseError(ValueError):
    """Raised when judge output cannot be parsed into JudgeVerdict."""


@dataclass(frozen=True)
class JudgePromptConfig:
    system_prompt: str
    user_template: str


@dataclass(frozen=True)
class JudgeExecutionResult:
    verdict: JudgeVerdict | None
    computed_verdict: Literal["agree", "disagree"]
    raw_text: str
    repaired: bool
    valid_first_pass: bool


def _judge_schema_definition() -> dict[str, str]:
    return {
        "preferred_source": "local|openai|compliance",
        "preferred_index": "int|null",
        "severity": "minor|major",
        "rationale": "one short sentence",
    }


def _judge_schema_json() -> str:
    return json.dumps(_judge_schema_definition())


def _load_judge_prompt_config(path: Path) -> JudgePromptConfig:
    if not path.exists():
        raise FileNotFoundError(f"Missing judge prompt config: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks: dict[str, list[str]] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_key, current_lines
        if current_key is not None:
            blocks[current_key] = current_lines.copy()
        current_key = None
        current_lines = []

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.strip() == "---":
            idx += 1
            continue
        if line.startswith("system_prompt: |"):
            flush()
            current_key = "system_prompt"
            idx += 1
            continue
        if line.startswith("user_template: |"):
            flush()
            current_key = "user_template"
            idx += 1
            continue
        if current_key is not None:
            if line.startswith("  "):
                current_lines.append(line[2:])
                idx += 1
                continue
            if line.strip() == "":
                current_lines.append("")
                idx += 1
                continue
            flush()
            continue
        idx += 1
    flush()

    system_prompt = "\n".join(blocks.get("system_prompt", []))
    user_template = "\n".join(blocks.get("user_template", []))
    if not system_prompt or not user_template:
        raise ValueError("judge_prompt.yaml must define system_prompt and user_template.")
    return JudgePromptConfig(system_prompt=system_prompt, user_template=user_template)


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_and_validate_judge_output(raw_text: str) -> JudgeVerdict:
    text = raw_text.strip()
    if not text:
        raise JudgeOutputParseError("Empty judge output.")

    try:
        parsed = json.loads(text)
        return JudgeVerdict.model_validate(parsed)
    except (json.JSONDecodeError, ValidationError):
        extracted = extract_first_json_object(text)
        if extracted is None:
            raise JudgeOutputParseError("No JSON object found in judge output.") from None
        try:
            parsed = json.loads(extracted)
            return JudgeVerdict.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise JudgeOutputParseError("Invalid judge JSON payload.") from exc


def try_parse_and_validate_judge_output(raw_text: str) -> JudgeVerdict | None:
    try:
        return parse_and_validate_judge_output(raw_text)
    except JudgeOutputParseError:
        return None


def parse_judge_output(raw_text: str) -> JudgeVerdict:
    return parse_and_validate_judge_output(raw_text)


class LLMJudgeClient:
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for judge evaluation.")
        self._settings = settings
        self._prompt_config = _load_judge_prompt_config(Path("config/judge_prompt.yaml"))
        self._json_mode_enabled = True
        self._llm = self._build_llm(
            model_name=settings.judge_model,
            max_tokens=800,
            temperature=1 if settings.judge_model.lower().startswith("gpt-5") else 0,
            json_mode=self._json_mode_enabled,
        )
        self._repair_llm = self._build_llm(
            model_name="gpt-4o-mini",
            max_tokens=800,
            temperature=0,
            json_mode=False,
        )

    def _build_llm(
        self,
        *,
        model_name: str,
        max_tokens: int,
        temperature: int,
        json_mode: bool,
    ) -> ChatOpenAI:
        kwargs: dict[str, object] = {
            "model": model_name,
            "api_key": self._settings.openai_api_key,
            "timeout": self._settings.openai_timeout_seconds,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    def _invoke(self, system_prompt: str, user_prompt: str, *, repair: bool = False) -> str:
        llm = self._repair_llm if repair else self._llm
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            )
            return str(response.content).strip()
        except Exception as exc:
            if (not repair) and self._json_mode_enabled and "response_format" in str(exc).lower():
                self._json_mode_enabled = False
                self._llm = self._build_llm(
                    model_name=self._settings.judge_model,
                    max_tokens=800,
                    temperature=1 if self._settings.judge_model.lower().startswith("gpt-5") else 0,
                    json_mode=False,
                )
                response = self._llm.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                return str(response.content).strip()
            raise

    def judge_once(self, record: EvalRecord) -> tuple[str, JudgeVerdict | None]:
        user_prompt = self._prompt_config.user_template.format(
            record_json=record.model_dump_json(indent=2),
            top_k_len=len(record.top_k),
        )
        raw_text = self._invoke(self._prompt_config.system_prompt, user_prompt, repair=False)
        parsed = try_parse_and_validate_judge_output(raw_text)
        return raw_text, parsed

    def repair_json_if_needed(self, raw_text: str, record: EvalRecord) -> JudgeVerdict | None:
        maybe_direct = try_parse_and_validate_judge_output(raw_text)
        if maybe_direct is not None:
            return maybe_direct

        extracted = extract_first_json_object(raw_text)
        if extracted:
            maybe_extracted = try_parse_and_validate_judge_output(extracted)
            if maybe_extracted is not None:
                return maybe_extracted

        repair_system = (
            "You are a JSON formatter. "
            "Do NOT change meaning. "
            "Do NOT reinterpret. "
            "Do NOT infer new values. "
            "Only fix formatting so it matches schema."
        )
        repair_input = extracted if extracted else raw_text
        repair_user = (
            "Input text:\n"
            f"{repair_input}\n\n"
            "Schema:\n"
            f"{_judge_schema_json()}\n\n"
            "Rules:\n"
            f"- expected_source in record is authoritative when not null: {record.expected_source}\n"
            f"- predicted_source in record: {record.predicted_source}\n"
            f"- top_k_len: {len(record.top_k)}\n"
            "Return only valid JSON. No explanation."
        )
        repaired_text = self._invoke(repair_system, repair_user, repair=True)
        return try_parse_and_validate_judge_output(repaired_text)

    def _apply_ground_truth_and_validate(
        self,
        *,
        record: EvalRecord,
        parsed: JudgeVerdict,
    ) -> JudgeVerdict | None:
        preferred_source = record.expected_source or parsed.preferred_source
        preferred_index = parsed.preferred_index
        if preferred_source != "local":
            preferred_index = None
        candidate = JudgeVerdict(
            preferred_source=preferred_source,  # type: ignore[arg-type]
            preferred_index=preferred_index,
            severity=parsed.severity,
            rationale=parsed.rationale,
        )

        top_k_len = len(record.top_k)
        if candidate.preferred_source == "local":
            if top_k_len > 0 and candidate.preferred_index is None:
                return None
            if candidate.preferred_index is not None and candidate.preferred_index >= top_k_len:
                return None
        return candidate

    def judge_with_repair(self, record: EvalRecord) -> JudgeExecutionResult:
        raw_text, parsed = self.judge_once(record)
        valid_first_pass = False
        repaired = False
        if parsed is not None:
            parsed = self._apply_ground_truth_and_validate(record=record, parsed=parsed)
            valid_first_pass = parsed is not None
        if parsed is None:
            repaired_candidate = self.repair_json_if_needed(raw_text, record)
            if repaired_candidate is not None:
                parsed = self._apply_ground_truth_and_validate(
                    record=record,
                    parsed=repaired_candidate,
                )
            repaired = parsed is not None

        computed_verdict: Literal["agree", "disagree"] = "disagree"
        if parsed is not None and parsed.preferred_source == record.predicted_source:
            computed_verdict = "agree"
        return JudgeExecutionResult(
            verdict=parsed,
            computed_verdict=computed_verdict,
            raw_text=raw_text,
            repaired=repaired,
            valid_first_pass=valid_first_pass,
        )

    def judge(self, record: EvalRecord) -> JudgeVerdict:
        execution = self.judge_with_repair(record)
        if execution.verdict is None:
            raise JudgeOutputParseError("Judge output invalid after repair pass.")
        return execution.verdict
