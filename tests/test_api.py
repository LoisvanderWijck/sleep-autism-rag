from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(api, "ask", lambda q: ("An answer [1]", ["a.pdf"]))
    response = client.post("/ask", json={"question": "What is PPI?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "An answer [1]", "sources": ["a.pdf"]}


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422