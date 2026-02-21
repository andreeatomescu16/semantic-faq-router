from app.services.llm_judge import extract_first_json_object


def test_extract_with_text_prefix() -> None:
    raw = 'prefix text {"preferred_source":"local","preferred_index":0,"severity":"minor","rationale":"ok"}'
    extracted = extract_first_json_object(raw)
    assert extracted is not None
    assert extracted.startswith("{")


def test_extract_handles_braces_inside_string() -> None:
    raw = '{"preferred_source":"local","preferred_index":0,"severity":"minor","rationale":"brace } in text"} trailing'
    extracted = extract_first_json_object(raw)
    assert extracted is not None
    assert extracted.endswith("}")


def test_extract_returns_first_when_multiple_json_objects() -> None:
    raw = (
        '{"preferred_source":"local","preferred_index":0,"severity":"minor","rationale":"first"}'
        '{"preferred_source":"openai","preferred_index":null,"severity":"major","rationale":"second"}'
    )
    extracted = extract_first_json_object(raw)
    assert extracted is not None
    assert '"rationale":"first"' in extracted


def test_extract_returns_none_when_no_json() -> None:
    assert extract_first_json_object("no json payload") is None
