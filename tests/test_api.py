"""tests/test_api.py"""
import io
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock


@pytest.fixture
def mock_pipeline():
    mock = MagicMock()
    mock.model_status.return_value = {
        "ner": "pkl", "asr": "whisper-small",
        "device": "cpu", "pkl_size_mb": 267,
    }
    mock.run_from_text.return_value = {
        "care_documentation": {
            "patient_name": "Müller",
            "symptoms": ["fever"],
            "medications": [{"name": "ibuprofen", "dosage": "400mg", "administered": False}],
            "vitals": {"temperature_celsius": 38.5},
            "raw_transcript": "Patient Müller had fever",
        },
        "entities": [
            {"text": "Müller",    "label": "PATIENT",    "confidence": 0.97},
            {"text": "fever",     "label": "SYMPTOM",    "confidence": 0.93},
            {"text": "ibuprofen", "label": "MEDICATION", "confidence": 0.95},
        ],
        "timings_ms": {"ner": 12.1, "total": 12.1},
    }
    mock.run_from_audio.return_value = {
        "transcript": "Patient had fever",
        "care_documentation": {"patient_name": "Test"},
        "entities": [],
        "timings_ms": {"asr": 180.0, "ner": 12.0, "total": 192.0},
    }
    return mock


@pytest.fixture
def client(mock_pipeline):
    import app.main as m
    m.pipeline = mock_pipeline
    from app.main import app
    return TestClient(app)


def test_health_200(client):
    assert client.get("/health").status_code == 200

def test_health_shows_pkl(client):
    assert client.get("/health").json()["ner"] == "pkl"

def test_predict_text_200(client):
    assert client.post("/predict/text", json={"text": "Patient had fever"}).status_code == 200

def test_predict_text_entities(client):
    r = client.post("/predict/text", json={"text": "Patient had fever"})
    assert isinstance(r.json()["entities"], list)

def test_predict_text_empty(client):
    assert client.post("/predict/text", json={"text": "  "}).status_code == 422

def test_predict_text_care_doc(client):
    r   = client.post("/predict/text", json={"text": "Patient Müller had fever"})
    doc = r.json()["care_documentation"]
    assert doc.get("patient_name") == "Müller"

def test_predict_audio_200(client):
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
    buf.seek(0)
    r = client.post("/predict/audio", files={"file": ("t.wav", buf, "audio/wav")})
    assert r.status_code == 200

def test_predict_audio_bad_format(client):
    r = client.post("/predict/audio", files={"file": ("t.txt", b"hi", "text/plain")})
    assert r.status_code == 422

def test_temperature_regex():
    import re
    m = re.search(r"(\d+\.?\d*)\s*(?:degrees?|°c)", "fever 38.5 degrees", re.I)
    assert m and float(m.group(1)) == 38.5

def test_negation():
    from app.pipeline import NEGATION_WORDS
    assert "refused" in NEGATION_WORDS
