import pytest
from fastapi.testclient import TestClient
from backend.app import app


@pytest.fixture(scope="module")
def client():
    # Trigger startup event
    with TestClient(app) as test_client:
        yield test_client


def test_api_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_store" in data


def test_api_config(client):
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "available_strategies" in data
    assert len(data["available_strategies"]) == 4


def test_api_text_query(client):
    payload = {
        "query": "what is the capital of France?",
        "top_k": 3
    }
    response = client.post("/api/text/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "answer" in data
    assert "latency_ms" in data
    assert data["latency_ms"]["total_ms"] > 0


def test_api_voice_query_mock(client):
    dummy_wav = b"RIFF" + b"\x00" * 200 # Fake 200 bytes WAV header payload
    files = {
        "audio": ("test.wav", dummy_wav, "audio/wav")
    }
    response = client.post("/api/voice/query", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "transcript" in data
    assert "answer" in data
    assert "latency_ms" in data
