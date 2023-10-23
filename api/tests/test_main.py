import pytest
from fastapi.testclient import TestClient

from mle_agent.main import app

url = "http://fastapi.localhost"


@pytest.fixture(scope="function")
def test_client() -> TestClient:
    return TestClient(app)


@pytest.mark.parametrize("question", ["What is the meaning of life?"])
def test_chat_ask(
    test_client: TestClient,
    question: str,
):
    with test_client as client:
        response = client.post("/chat/ask", json={"question": question})
        assert response is not None
        assert response.status_code == 200
