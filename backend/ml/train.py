"""
Model training pipeline — v2 (methodology fixes applied).

Fixes vs. the original prototype:
  1. StratifiedKFold instead of KFold (critical with 3.2% minority class)
  2. Imbalance handling per model:
       - class_weight='balanced' for LR / RF / SVM / DT
       - scale_pos_weight for XGBoost
       - SMOTE (inside an imblearn Pipeline, so it only ever runs on
         training folds — no resampling leakage) for GB / MLP, which
         have no class_weight support
  3. Full metric suite on the BANKRUPT class: precision, recall, F1,
     ROC-AUC, PR-AUC, MCC, balanced accuracy, confusion matrix
  4. Per-model decision threshold tuned to maximize F1 using
     cross-validated predictions on the TRAINING split only, then
     evaluated once on the untouched holdout set
  5. Models trained once, persisted to backend/artifacts/ with a
     metrics JSON — the API serves these, it never retrains
  6. Scaling handled inside each model's Pipeline (fit on train only)

Run:
    python train.py
Outputs (in backend/artifacts/):
    models_<variant>.joblib   - dict of fitted pipelines
    metrics.json              - all metrics, curves, thresholds
    features.json             - feature names, stats, example rows
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    balanced_accuracy_score, confusion_matrix, roc_curve,
    precision_recall_curve,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- paths

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_FEATURES_PATH = ROOT_DIR / "data" / "processed" / "Filtered_Top_Features_with_Bankruptcy_Indicator.xlsx"

RANDOM_STATE = 42
DATASET_VARIANTS = {
    "original": "Original Top Features",
    "knn30": "KNN-30 Top Features",
}

# ---------------------------------------------------------------- models


def build_model_specs(pos_weight: float):
    """
    Each entry: name -> (pipeline, short_key).
    Scaling lives INSIDE the pipeline so it's always fit on training
    data only. SMOTE (where used) also lives inside an imblearn
    pipeline, so it resamples training folds only.
    """
    return {
        "Logistic Regression": SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                        random_state=RANDOM_STATE)),
        ]),
        "Random Forest": SkPipeline([
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                random_state=RANDOM_STATE)),
        ]),
        "SVM": SkPipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced",
                         random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": SkPipeline([
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_split=20,
                                            class_weight="balanced",
                                            random_state=RANDOM_STATE)),
        ]),
        "Neural Network": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1500,
                                   activation="relu", solver="adam",
                                   early_stopping=True, n_iter_no_change=15,
                                   random_state=RANDOM_STATE)),
        ]),
        "XGBoost": SkPipeline([
            ("clf", XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                                   scale_pos_weight=pos_weight,
                                   eval_metric="logloss",
                                   random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


MODEL_KEYS = {
    "Logistic Regression": "lr",
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
    "SVM": "svm",
    "Decision Tree": "dt",
    "Neural Network": "mlp",
    "XGBoost": "xgb",
}

# ---------------------------------------------------------------- helpers


def tune_threshold(y_true, y_proba):
    """Threshold that maximizes F1 for the bankrupt (positive) class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    # precision_recall_curve returns len(thresholds) = len(precisions) - 1
    best_idx = int(np.nanargmax(f1s[:-1]))
    return float(thresholds[best_idx])


def evaluate(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision_bankrupt": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_bankrupt": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_bankrupt": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
    }


def downsample_curve(x, y, max_points=120):
    """Thin curve points so the JSON stays light for the frontend."""
    if len(x) <= max_points:
        idx = np.arange(len(x))
    else:
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return [{"x": round(float(x[i]), 4), "y": round(float(y[i]), 4)} for i in idx]


# ---------------------------------------------------------------- training


def train_variant(variant_key: str, sheet_name: str):
    print(f"\n=== Variant: {variant_key} ({sheet_name}) ===")
    df = pd.read_excel(FILTERED_FEATURES_PATH, sheet_name=sheet_name)
    target_col = df.columns[0]
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)
    print(f"Rows: {len(y)} | bankrupt: {n_pos} | non-bankrupt: {n_neg} "
          f"| scale_pos_weight: {pos_weight:.1f}")

    # Single stratified holdout, untouched until final evaluation
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fitted_models = {}
    variant_metrics = {}

    for name, pipeline in build_model_specs(pos_weight).items():
        key = MODEL_KEYS[name]
        print(f"  training {name}...")

        # Cross-validated probabilities on the TRAINING split only —
        # used both for CV metrics and for threshold tuning.
        cv_proba = cross_val_predict(pipeline, X_train, y_train, cv=skf,
                                      method="predict_proba", n_jobs=-1)[:, 1]
        threshold = tune_threshold(y_train.values, cv_proba)
        cv_metrics = evaluate(y_train.values, cv_proba, threshold)

        # Fit on the full training split, evaluate ONCE on holdout
        pipeline.fit(X_train, y_train)
        hold_proba = pipeline.predict_proba(X_hold)[:, 1]
        hold_metrics = evaluate(y_hold.values, hold_proba, threshold)

        # Default-threshold (0.5) holdout metrics, for the "why tuned
        # thresholds matter" comparison in the dashboard
        hold_default = evaluate(y_hold.values, hold_proba, 0.5)

        fpr, tpr, _ = roc_curve(y_hold.values, hold_proba)
        prec, rec, _ = precision_recall_curve(y_hold.values, hold_proba)

        fitted_models[key] = pipeline
        variant_metrics[key] = {
            "name": name,
            "cv": cv_metrics,
            "holdout": hold_metrics,
            "holdout_default_threshold": hold_default,
            "roc_curve": downsample_curve(fpr, tpr),
            "pr_curve": downsample_curve(rec, prec),
        }
        print(f"    holdout: recall={hold_metrics['recall_bankrupt']:.3f} "
              f"f1={hold_metrics['f1_bankrupt']:.3f} "
              f"pr_auc={hold_metrics['pr_auc']:.3f} "
              f"(threshold {threshold:.3f})")

    joblib.dump(fitted_models, ARTIFACTS_DIR / f"models_{variant_key}.joblib")

    # Feature metadata for the frontend form
    feature_stats = {
        col: {
            "min": round(float(X[col].min()), 6),
            "max": round(float(X[col].max()), 6),
            "mean": round(float(X[col].mean()), 6),
            "median": round(float(X[col].median()), 6),
        }
        for col in X.columns
    }
    # Real example rows for "load example" buttons (medians per class)
    examples = {
        "distressed": {c: round(float(v), 6) for c, v in X[y == 1].median().items()},
        "healthy": {c: round(float(v), 6) for c, v in X[y == 0].median().items()},
    }

    return {
        "metrics": variant_metrics,
        "features": {
            "names": list(X.columns),
            "stats": feature_stats,
            "examples": examples,
            "class_counts": {"bankrupt": n_pos, "non_bankrupt": n_neg},
        },
    }


def main():
    if not FILTERED_FEATURES_PATH.exists():
        sys.exit(f"Missing {FILTERED_FEATURES_PATH} — run src/run_pipeline.py first.")

    all_metrics = {}
    all_features = {}
    for variant_key, sheet in DATASET_VARIANTS.items():
        result = train_variant(variant_key, sheet)
        all_metrics[variant_key] = result["metrics"]
        all_features[variant_key] = result["features"]

    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (ARTIFACTS_DIR / "features.json").write_text(json.dumps(all_features, indent=2))
    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
