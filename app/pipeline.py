from __future__ import annotations
 
import logging
import pickle
import re
import time
from pathlib import Path
 
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    DistilBertConfig,
    DistilBertForTokenClassification,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
 
log = logging.getLogger(__name__)
 
NER_PKL = Path("models/ner_model.pkl")
 
NEGATION_WORDS = {
    "refused", "denied", "no", "not", "without",
    "verweigert", "keine", "nicht",
}
 
 
class HealthcarePipeline:
 
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info(f"Device: {self.device}")
        self._load_ner()
        self._load_whisper()
        log.info("Pipeline ready")
 
    def _load_ner(self):
        if not NER_PKL.exists():
            raise FileNotFoundError(
                f"Model not found at {NER_PKL}\n"
                "1. Train on Colab\n"
                "2. Download ner_model.pkl\n"
                "3. Copy to models/ner_model.pkl"
            )
 
        log.info(f"Loading NER from pkl: {NER_PKL}")
        with open(NER_PKL, "rb") as f:
            data = pickle.load(f)
 
        tokenizer_name = data.get("tokenizer_name", "distilbert-base-cased")
 
        base_config = DistilBertConfig.from_pretrained(tokenizer_name)
        base_config.num_labels = data["num_labels"]
        base_config.id2label   = {int(k): v for k, v in data["id2label"].items()}
        base_config.label2id   = {v: int(k) for k, v in data["id2label"].items()}
 
        self.ner_model = DistilBertForTokenClassification(base_config)
        self.ner_model.load_state_dict(data["model_state_dict"])
        self.ner_model.to(self.device)
        self.ner_model.eval()
 
        self.ner_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.id2label = base_config.id2label
 
        size_mb = round(NER_PKL.stat().st_size / 1e6, 1)
        log.info(f"NER loaded ({size_mb} MB) labels: {self.id2label}")
 
    def _load_whisper(self):
        log.info("Loading Whisper-small...")
        self.asr_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.asr_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
        self.asr_model.to(self.device)
        self.asr_model.eval()
        log.info("Whisper loaded")
 
    def model_status(self):
        return {
            "ner":         "pkl",
            "asr":         "whisper-small",
            "device":      self.device,
            "pkl_size_mb": round(NER_PKL.stat().st_size / 1e6, 1),
        }
 
    def transcribe(self, audio):
        inputs = self.asr_processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(self.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = self.asr_model.generate(**inputs, language="german")
        ms = (time.perf_counter() - t0) * 1000
        text = self.asr_processor.batch_decode(ids, skip_special_tokens=True)[0]
        return text, round(ms, 1)
 
    def extract_entities(self, text):
        words = text.split()
        if not words:
            return [], 0.0
 
        enc = self.ner_tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(self.device)
 
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.ner_model(**enc).logits
        ms = (time.perf_counter() - t0) * 1000
 
        probs    = torch.softmax(logits, dim=-1)[0]
        pred_ids = torch.argmax(logits, dim=-1)[0]
        word_ids = enc.word_ids()
 
        entities = []
        current  = None
        prev     = None
 
        for i, wid in enumerate(word_ids):
            if wid is None or wid == prev:
                continue
            label = self.id2label.get(int(pred_ids[i]), "O")
            conf  = float(probs[i, int(pred_ids[i])])
 
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                current = {"text": words[wid], "label": label[2:], "confidence": conf}
            elif label.startswith("I-") and current:
                current["text"] += " " + words[wid]
                current["confidence"] = min(current["confidence"], conf)
            else:
                if current:
                    entities.append(current)
                current = None
            prev = wid
 
        if current:
            entities.append(current)
 
        return [e for e in entities if e["confidence"] >= 0.75], round(ms, 1)
 
    def build_care_doc(self, entities, transcript):
        doc = {
            "patient_name":     None,
            "observation_time": None,
            "vitals":           {},
            "symptoms":         [],
            "medications":      [],
            "procedures":       [],
            "raw_transcript":   transcript,
        }
 
        for ent in entities:
            text  = ent["text"].strip()
            label = ent["label"]
 
            if label == "PATIENT" and not doc["patient_name"]:
                doc["patient_name"] = text
            elif label == "SYMPTOM":
                if text not in doc["symptoms"]:
                    doc["symptoms"].append(text)
            elif label == "MEDICATION":
                idx   = transcript.lower().find(text.lower())
                prev  = transcript[:idx].split()[-5:] if idx != -1 else []
                given = not any(w.lower() in NEGATION_WORDS for w in prev)
                dos   = re.search(r"(\d+\s*mg)", text)
                doc["medications"].append({
                    "name":         text,
                    "dosage":       dos.group(1) if dos else None,
                    "administered": given,
                })
            elif label == "DATE" and not doc["observation_time"]:
                doc["observation_time"] = text
            elif label == "PROCEDURE":
                if text not in doc["procedures"]:
                    doc["procedures"].append(text)
 
        t = transcript
        m = re.search(r"(\d+\.?\d*)\s*(?:degrees?|°c)", t, re.IGNORECASE)
        if m:
            doc["vitals"]["temperature_celsius"] = float(m.group(1))
        m = re.search(r"(\d{2,3})\s+over\s+(\d{2,3})", t, re.IGNORECASE)
        if m:
            doc["vitals"]["blood_pressure"] = m.group(1) + "/" + m.group(2)
        m = re.search(r"(\d+)\s*bpm", t, re.IGNORECASE)
        if m:
            doc["vitals"]["pulse_bpm"] = int(m.group(1))
        m = re.search(r"(\d+\.?\d*)\s*mmol", t, re.IGNORECASE)
        if m:
            doc["vitals"]["blood_glucose_mmol"] = float(m.group(1))
        m = re.search(r"(\d+)\s*percent", t, re.IGNORECASE)
        if m:
            doc["vitals"]["oxygen_saturation_pct"] = int(m.group(1))
 
        return {k: v for k, v in doc.items() if v not in (None, [], {})}
 
    def run_from_text(self, text):
        entities, ner_ms = self.extract_entities(text)
        care_doc         = self.build_care_doc(entities, text)
        return {
            "care_documentation": care_doc,
            "entities":           entities,
            "timings_ms":         {"ner": ner_ms, "total": ner_ms},
        }
 
    def run_from_audio(self, audio):
        transcript, asr_ms = self.transcribe(audio)
        entities, ner_ms   = self.extract_entities(transcript)
        care_doc           = self.build_care_doc(entities, transcript)
        return {
            "transcript":         transcript,
            "care_documentation": care_doc,
            "entities":           entities,
            "timings_ms": {
                "asr":   asr_ms,
                "ner":   ner_ms,
                "total": round(asr_ms + ner_ms, 1),
            },
        }