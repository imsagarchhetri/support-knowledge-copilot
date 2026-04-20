from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_endpoint():
    r = client.post("/ask", json={"question": "How do I reset my password?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert body["citations"]
    assert "confidence" in body


def test_ask_refusal_endpoint():
    r = client.post("/ask", json={"question": "Who won the world cup in 1998?"})
    assert r.status_code == 200
    assert r.json()["refused"] is True
