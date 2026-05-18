from __future__ import annotations

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from train_models import BEST_MODEL_PATH, FEATURE_COLUMNS, main as train_models_main


app = FastAPI(
    title="Heart Failure Mortality Risk API",
    version="1.0.0",
    description="Teaching API that serves a mortality risk model for UCI dataset 519.",
)


class PatientRecord(BaseModel):
    age: float = Field(..., ge=0, le=120)
    anaemia: int = Field(..., ge=0, le=1)
    creatinine_phosphokinase: int = Field(..., ge=0)
    diabetes: int = Field(..., ge=0, le=1)
    ejection_fraction: int = Field(..., ge=0, le=100)
    high_blood_pressure: int = Field(..., ge=0, le=1)
    platelets: float = Field(..., gt=0)
    serum_creatinine: float = Field(..., gt=0)
    serum_sodium: int = Field(..., gt=0)
    sex: int = Field(..., ge=0, le=1)
    smoking: int = Field(..., ge=0, le=1)
    time: int = Field(..., ge=0)


def load_model():
    if not BEST_MODEL_PATH.exists():
        train_models_main()
    return joblib.load(BEST_MODEL_PATH)


@app.get("/")
def root() -> dict:
    return {
        "name": "Heart Failure Mortality Risk API",
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict",
        "interactive_docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_available": BEST_MODEL_PATH.exists()}


@app.post("/predict")
def predict(record: PatientRecord) -> dict:
    model = load_model()
    row = pd.DataFrame([record.model_dump()], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "death_event_prediction": prediction,
        "death_event_probability": probability,
        "interpretation": "1 means predicted death during follow-up; 0 means predicted survival during follow-up.",
    }
