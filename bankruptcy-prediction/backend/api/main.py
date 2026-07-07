"""
FastAPI backend for the Bankruptcy Prediction app — v3.

Serves artifacts from ml/train.py. Never retrains.

Endpoints:
    GET  /api/health
    GET  /api/summary
    GET  /api/features?variant=
    POST /api/predict          -> per-model probs, consensus, conformal band
    POST /api/explain          -> SHAP contributions + counterfactual (one model)
    POST /api/batch            -> score many companies from parsed CSV rows
"""

import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .explain import shap_values_for, counterfactual_for, conformal_pvalue
from .narrative import generate_narrative

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

app = FastAPI(title="Bankruptcy Prediction API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

MODELS: Dict[str, dict] = {}
BACKGROUNDS: Dict[str, pd.DataFrame] = {}
CONFORMAL: Dict[str, dict] = {}
METRICS: dict = {}
FEATURES: dict = {}


@app.on_event("startup")
def load_artifacts():
    global METRICS, FEATURES
    if not (ARTIFACTS_DIR / "metrics.json").exists():
        raise RuntimeError(f"No artifacts in {ARTIFACTS_DIR}. Run backend/ml/train.py first.")
    METRICS = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    FEATURES = json.loads((ARTIFACTS_DIR / "features.json").read_text())
    for variant in METRICS.keys():
        MODELS[variant] = joblib.load(ARTIFACTS_DIR / f"models_{variant}.joblib")
        BACKGROUNDS[variant] = joblib.load(ARTIFACTS_DIR / f"background_{variant}.joblib")
        CONFORMAL[variant] = joblib.load(ARTIFACTS_DIR / f"conformal_{variant}.joblib")
    print(f"Loaded {sum(len(v) for v in MODELS.values())} models across {len(MODELS)} variants.")


# ------------------------------------------------------------- schemas

class PredictRequest(BaseModel):
    variant: str = "mean"
    features: Dict[str, float]
    threshold_mode: str = "tuned"


class ExplainRequest(BaseModel):
    variant: str = "mean"
    model_key: str = "rf"
    features: Dict[str, float]
    threshold_mode: str = "tuned"
    want_counterfactual: bool = True
    want_narrative: bool = True


class BatchRequest(BaseModel):
    variant: str = "mean"
    model_key: str = "rf"
    rows: List[Dict[str, float]]
    threshold_mode: str = "tuned"


# ------------------------------------------------------------- helpers

def _check_variant(variant):
    if variant not in MODELS:
        raise HTTPException(404, f"Unknown variant '{variant}'")

def _order_features(variant, features):
    expected = FEATURES[variant]["names"]
    missing = [f for f in expected if f not in features]
    if missing:
        raise HTTPException(422, f"Missing features: {missing}")
    return pd.DataFrame([{f: features[f] for f in expected}]), expected

def _threshold(variant, key, mode):
    return METRICS[variant][key]["holdout"]["threshold"] if mode == "tuned" else 0.5

def _risk_tier(avg):
    if avg >= 0.5: return "high"
    if avg >= 0.2: return "elevated"
    if avg >= 0.05: return "moderate"
    return "low"


# ------------------------------------------------------------- routes

@app.get("/api/health")
def health():
    return {"status": "ok", "variants": list(MODELS.keys())}

@app.get("/api/summary")
def summary():
    return METRICS

@app.get("/api/features")
def features(variant: str = "mean"):
    if variant not in FEATURES:
        raise HTTPException(404, f"Unknown variant '{variant}'")
    return FEATURES[variant]


@app.post("/api/predict")
def predict(req: PredictRequest):
    _check_variant(req.variant)
    sample, expected = _order_features(req.variant, req.features)

    results, votes = [], 0
    ensemble_probs = []
    for key, model in MODELS[req.variant].items():
        proba = float(model.predict_proba(sample)[0][1])
        ensemble_probs.append(proba)
        thr = _threshold(req.variant, key, req.threshold_mode)
        flagged = proba >= thr
        votes += int(flagged)
        results.append({
            "model_key": key, "model_name": METRICS[req.variant][key]["name"],
            "probability": round(proba, 4), "threshold": thr,
            "prediction": "bankrupt" if flagged else "not_bankrupt",
        })

    n = len(results)
    avg = sum(ensemble_probs) / n
    consensus = "bankrupt" if votes > n / 2 else "not_bankrupt" if votes < n / 2 else "split"

    # Conformal band from the best single model (RF by default if present)
    lead_key = "rf" if "rf" in MODELS[req.variant] else list(MODELS[req.variant])[0]
    lead_proba = next(r["probability"] for r in results if r["model_key"] == lead_key)
    conf = conformal_pvalue(CONFORMAL[req.variant][lead_key], lead_proba)

    return {
        "consensus": consensus, "votes_bankrupt": votes, "n_models": n,
        "average_probability": round(avg, 4), "risk_tier": _risk_tier(avg),
        "conformal": {**conf, "model": METRICS[req.variant][lead_key]["name"]},
        "models": results,
    }


@app.post("/api/explain")
def explain(req: ExplainRequest):
    _check_variant(req.variant)
    if req.model_key not in MODELS[req.variant]:
        raise HTTPException(404, f"Unknown model '{req.model_key}'")
    sample, expected = _order_features(req.variant, req.features)

    pipeline = MODELS[req.variant][req.model_key]
    background = BACKGROUNDS[req.variant]
    thr = _threshold(req.variant, req.model_key, req.threshold_mode)
    proba = float(pipeline.predict_proba(sample)[0][1])
    prediction = "bankrupt" if proba >= thr else "not_bankrupt"

    shap_out = shap_values_for(pipeline, background, sample, expected)

    cf = None
    if req.want_counterfactual:
        cf = counterfactual_for(pipeline, sample, expected,
                                 FEATURES[req.variant]["stats"], thr)

    narrative = None
    if req.want_narrative:
        tier = _risk_tier(proba)
        narrative = generate_narrative(
            proba, tier, prediction,
            shap_out.get("contributions", []),
            METRICS[req.variant][req.model_key]["name"],
        )

    return {
        "model_name": METRICS[req.variant][req.model_key]["name"],
        "probability": round(proba, 4),
        "threshold": thr,
        "prediction": prediction,
        "shap": shap_out,
        "counterfactual": cf,
        "narrative": narrative,
    }


@app.post("/api/batch")
def batch(req: BatchRequest):
    _check_variant(req.variant)
    if req.model_key not in MODELS[req.variant]:
        raise HTTPException(404, f"Unknown model '{req.model_key}'")
    expected = FEATURES[req.variant]["names"]
    pipeline = MODELS[req.variant][req.model_key]
    thr = _threshold(req.variant, req.model_key, req.threshold_mode)

    # Build a frame, coercing/validating columns
    clean_rows, skipped = [], 0
    for row in req.rows:
        if all(f in row for f in expected):
            clean_rows.append({f: row[f] for f in expected})
        else:
            skipped += 1
    if not clean_rows:
        raise HTTPException(422, f"No rows had all required columns: {expected}")

    frame = pd.DataFrame(clean_rows)
    probs = pipeline.predict_proba(frame)[:, 1]

    out = []
    for i, p in enumerate(probs):
        out.append({
            "row": i, "probability": round(float(p), 4),
            "prediction": "bankrupt" if p >= thr else "not_bankrupt",
            "risk_tier": _risk_tier(float(p)),
        })
    out.sort(key=lambda r: r["probability"], reverse=True)

    n_flagged = sum(1 for r in out if r["prediction"] == "bankrupt")
    return {
        "model_name": METRICS[req.variant][req.model_key]["name"],
        "threshold": thr, "n_scored": len(out), "n_skipped": skipped,
        "n_flagged": n_flagged, "required_columns": expected, "results": out,
    }
