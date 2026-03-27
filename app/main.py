"""
app/main.py — FastAPI inference server
"""

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline import HealthcarePipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

pipeline: HealthcarePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    log.info("Loading pipeline...")
    pipeline = HealthcarePipeline()
    yield


app = FastAPI(
    title="voize Healthcare API",
    description="Caregiver speech/text → structured care documentation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextRequest(BaseModel):
    text: str
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Patient Müller had fever of 38.5 degrees, refused ibuprofen 400mg."
            }
        }
    }


class PredictResponse(BaseModel):
    transcript:         str | None = None
    care_documentation: dict
    entities:           list[dict]
    timings_ms:         dict


@app.get("/health")
def health():
    if not pipeline:
        return {"status": "loading"}
    s = pipeline.model_status()
    return {
        "status":      "healthy",
        "ner":         s["ner"],
        "asr":         s["asr"],
        "pkl_size_mb": s["pkl_size_mb"],
        "device":      s["device"],
    }


@app.get("/metrics")
def get_metrics():
    path = Path("evaluation/metrics.json")
    if not path.exists():
        return {"message": "No metrics. Copy metrics.json from Colab."}
    return json.loads(path.read_text())


@app.post("/predict/text", response_model=PredictResponse)
def predict_text(req: TextRequest):
    if not pipeline:
        raise HTTPException(503, "Pipeline loading")
    if not req.text.strip():
        raise HTTPException(422, "Text is empty")
    return PredictResponse(**pipeline.run_from_text(req.text))


@app.post("/predict/audio", response_model=PredictResponse)
async def predict_audio(file: UploadFile = File(...)):
    if not pipeline:
        raise HTTPException(503, "Pipeline loading")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac"}:
        raise HTTPException(422, f"Unsupported format: {suffix}")

    raw = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(raw))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16_000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        audio = audio.astype(np.float32)
    except Exception as e:
        raise HTTPException(422, f"Cannot read audio: {e}")

    return PredictResponse(**pipeline.run_from_audio(audio))
