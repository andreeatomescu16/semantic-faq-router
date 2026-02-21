from fastapi.testclient import TestClient


def test_web_ui_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Semantic FAQ Assistant" in response.text
