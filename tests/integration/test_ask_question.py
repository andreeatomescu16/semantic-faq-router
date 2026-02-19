from fastapi.testclient import TestClient


def test_ask_question_local_match(client: TestClient) -> None:
    response = client.post("/ask-question", json={"user_question": "How to edit my profile?"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "local"
    assert body["matched_question"] != "N/A"
    assert body["category"] == "profile"
    assert body["confidence"] is not None


def test_ask_question_out_of_domain_compliance(client: TestClient) -> None:
    response = client.post("/ask-question", json={"user_question": "Who won the world cup?"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "compliance"
    assert body["matched_question"] == "N/A"
    assert body["answer"] == "This is not really what I was trained for, therefore I cannot answer. Try again."


def test_ask_question_openai_fallback(client: TestClient) -> None:
    response = client.post("/ask-question", json={"user_question": "I need help with password policy details"})
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "openai"
    assert body["answer"].startswith("mocked-openai-answer:")
