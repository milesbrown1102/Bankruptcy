"""
FastAPI backend for the Bankruptcy Prediction app.

Serves the artifacts produced by ml/train.py — it never retrains.

Run (from backend/):
    uvicorn api.main:app --reload --port 8000

Endpoints:
    GET  /api/health
    GET  /api/summary            -> metrics for every model, both variants
    GET  /api/features           -> feature names, stats, example inputs
    POST /api/predict            -> per-model probabilities + consensus
"""

import json
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

app = FastAPI(title="Bankruptcy Prediction API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------- load once

MODELS: Dict[str, dict] = {}
METRICS: dict = {}
FEATURES: dict = {}


@app.on_event("startup")
def load_artifacts():
    global METRICS, FEATURES
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    features_path = ARTIFACTS_DIR / "features.json"
    if not metrics_path.exists():
        raise RuntimeError(
            f"No artifacts found in {ARTIFACTS_DIR}. Run backend/ml/train.py first."
        )
    METRICS = json.loads(metrics_path.read_text())
    FEATURES = json.loads(features_path.read_text())
    for variant in METRICS.keys():
        MODELS[variant] = joblib.load(ARTIFACTS_DIR / f"models_{variant}.joblib")
    print(f"Loaded {sum(len(v) for v in MODELS.values())} models "
          f"across {len(MODELS)} dataset variants.")


# ------------------------------------------------------------- schemas


class PredictRequest(BaseModel):
    variant: str = Field(default="original", description="'original' or 'knn30'")
    features: Dict[str, float]
    threshold_mode: str = Field(
        default="tuned",
        description="'tuned' uses each model's F1-optimized threshold; "
                    "'default' uses 0.5",
    )


# ------------------------------------------------------------- routes


@app.get("/api/health")
def health():
    return {"status": "ok", "variants": list(MODELS.keys())}


@app.get("/api/summary")
def summary():
    return METRICS


@app.get("/api/features")
def features(variant: str = "original"):
    if variant not in FEATURES:
        raise HTTPException(404, f"Unknown variant '{variant}'")
    return FEATURES[variant]


@app.post("/api/predict")
def predict(req: PredictRequest):
    if req.variant not in MODELS:
        raise HTTPException(404, f"Unknown variant '{req.variant}'")

    expected = FEATURES[req.variant]["names"]
    missing = [f for f in expected if f not in req.features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")

    sample = pd.DataFrame([{f: req.features[f] for f in expected}])

    results = []
    votes_bankrupt = 0
    for key, model in MODELS[req.variant].items():
        proba = float(model.predict_proba(sample)[0][1])
        meta = METRICS[req.variant][key]
        threshold = (
            meta["holdout"]["threshold"] if req.threshold_mode == "tuned" else 0.5
        )
        is_bankrupt = proba >= threshold
        votes_bankrupt += int(is_bankrupt)
        results.append({
            "model_key": key,
            "model_name": meta["name"],
            "probability": round(proba, 4),
            "threshold": threshold,
            "prediction": "bankrupt" if is_bankrupt else "not_bankrupt",
        })

    n_models = len(results)
    avg_proba = sum(r["probability"] for r in results) / n_models
    if votes_bankrupt > n_models / 2:
        consensus = "bankrupt"
    elif votes_bankrupt < n_models / 2:
        consensus = "not_bankrupt"
    else:
        consensus = "split"

    # Risk tier from average probability (simple, transparent banding)
    if avg_proba >= 0.5:
        tier = "high"
    elif avg_proba >= 0.2:
        tier = "elevated"
    elif avg_proba >= 0.05:
        tier = "moderate"
    else:
        tier = "low"

    return {
        "consensus": consensus,
        "votes_bankrupt": votes_bankrupt,
        "n_models": n_models,
        "average_probability": round(avg_proba, 4),
        "risk_tier": tier,
        "models": results,
    }
