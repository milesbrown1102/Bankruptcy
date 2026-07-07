"""
Model training pipeline — v3 (leak-free + explainability + conformal).

Adds on top of v2:
  - Imputation is now a step INSIDE each model pipeline (SimpleImputer /
    KNNImputer), so it is refit per CV fold and never sees the label. No
    more bankrupt/non-bankrupt-conditioned imputation.
  - Feature selection runs on the training split only (see data_prep.py).
  - SHAP: a background sample + fitted models are saved so the API can
    compute per-prediction SHAP values on demand.
  - Conformal prediction: split-conformal calibration scores are saved so
    the API can return an uncertainty band with a coverage guarantee.

Variants (kept so the app keeps its dataset switcher, but cleaner now):
    mean  -> median imputation (simple baseline)
    knn   -> KNN imputation (k=15), leak-free this time

Run:  python train.py
"""

import json
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
from sklearn.impute import SimpleImputer, KNNImputer
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

from data_prep import load_raw, select_features_on_training, TARGET_COLUMN, RANDOM_STATE

import warnings
warnings.filterwarnings("ignore")

BACKEND_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 20
CALIB_FRACTION = 0.2

VARIANTS = {"mean": "median", "knn": "knn"}

MODEL_KEYS = {
    "Logistic Regression": "lr", "Random Forest": "rf",
    "Gradient Boosting": "gb", "SVM": "svm", "Decision Tree": "dt",
    "Neural Network": "mlp", "XGBoost": "xgb",
}

FEATURE_DESCRIPTIONS = {
    "Net Income to Total Assets": "Return on assets: profit generated per dollar of assets. Low or negative signals distress.",
    "ROA(A) before interest and % after tax": "Return on assets before interest, after tax. Core operating profitability.",
    "ROA(B) before interest and depreciation after tax": "Return on assets before interest and depreciation. A cash-flow-flavored profitability measure.",
    "ROA(C) before interest and depreciation before interest": "Return on assets before interest and depreciation. Broadest profitability ratio here.",
    "Debt ratio %": "Total liabilities over total assets. Higher = more leverage = more bankruptcy risk.",
    "Net worth/Assets": "Equity as a share of assets (inverse of leverage). Lower = thinner equity cushion.",
    "Persistent EPS in the Last Four Seasons": "Sustained earnings per share over the past year. Weak or negative signals trouble.",
    "Retained Earnings to Total Assets": "Accumulated profits kept in the business relative to assets. Low = little internal buffer.",
    "Net profit before tax/Paid-in capital": "Pre-tax profit relative to capital contributed by shareholders.",
    "Per Share Net profit before tax (Yuan ??)": "Pre-tax profit per share, in NT dollars.",
    "Borrowing dependency": "Reliance on borrowed funds. Higher dependency raises default risk.",
    "Net Value Per Share (B)": "Book equity value per share.",
    "Net Value Per Share (A)": "Book equity value per share (alternate calculation).",
    "Net Value Per Share (C)": "Book equity value per share (alternate calculation).",
    "Working Capital to Total Assets": "Short-term liquidity buffer relative to assets. Low = liquidity strain.",
    "Current Ratio": "Current assets over current liabilities. Below 1 signals short-term solvency risk.",
    "Total income/Total expense": "Income relative to expenses. Near or below 1 means costs are barely covered.",
    "Cash/Total Assets": "Cash reserves as a share of assets.",
    "Interest Coverage Ratio (Interest expense to EBIT)": "How comfortably earnings cover interest payments.",
    "Liability to Equity": "Total liabilities relative to equity — a direct leverage gauge.",
    "Net profit before tax/Paid-in capital": "Pre-tax profit relative to shareholder capital.",
}


def make_imputer(kind):
    if kind == "knn":
        return ("imputer", KNNImputer(n_neighbors=15))
    return ("imputer", SimpleImputer(strategy="median"))


def build_model_specs(pos_weight, impute_kind):
    imp = make_imputer(impute_kind)
    return {
        "Logistic Regression": SkPipeline([
            imp, ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": SkPipeline([
            imp,
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": ImbPipeline([
            imp, ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=RANDOM_STATE)),
        ]),
        "SVM": SkPipeline([
            imp, ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": SkPipeline([
            imp,
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_split=20, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Neural Network": ImbPipeline([
            imp, ("scaler", StandardScaler()), ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1500, activation="relu",
                                   solver="adam", early_stopping=True, n_iter_no_change=15, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": SkPipeline([
            imp,
            ("clf", XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5,
                                   scale_pos_weight=pos_weight, eval_metric="logloss",
                                   random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
    }


def tune_threshold(y_true, y_proba):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    return float(thresholds[int(np.nanargmax(f1s[:-1]))])


def evaluate(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
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
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def downsample_curve(x, y, max_points=120):
    idx = np.arange(len(x)) if len(x) <= max_points else np.linspace(0, len(x) - 1, max_points).astype(int)
    return [{"x": round(float(x[i]), 4), "y": round(float(y[i]), 4)} for i in idx]


def train_variant(variant_key, impute_kind):
    print(f"\n=== Variant: {variant_key} (impute={impute_kind}) ===")
    df = load_raw()
    X_all = df.drop(columns=[TARGET_COLUMN])
    y_all = df[TARGET_COLUMN].astype(int)

    X_tr_full, X_hold_full, y_tr, y_hold = train_test_split(
        X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all)

    selected = select_features_on_training(X_tr_full, y_tr, top_n=TOP_N)
    print(f"Selected {len(selected)} features on training split")

    X_tr = X_tr_full[selected]
    X_hold = X_hold_full[selected]

    n_pos, n_neg = int(y_tr.sum()), int(len(y_tr) - y_tr.sum())
    pos_weight = n_neg / max(n_pos, 1)

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_tr, y_tr, test_size=CALIB_FRACTION, random_state=RANDOM_STATE, stratify=y_tr)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fitted_models, variant_metrics, conformal = {}, {}, {}

    for name, pipeline in build_model_specs(pos_weight, impute_kind).items():
        key = MODEL_KEYS[name]
        print(f"  {name}...")
        cv_proba = cross_val_predict(pipeline, X_fit, y_fit, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
        threshold = tune_threshold(y_fit.values, cv_proba)
        cv_metrics = evaluate(y_fit.values, cv_proba, threshold)

        pipeline.fit(X_fit, y_fit)
        cal_proba = pipeline.predict_proba(X_cal)[:, 1]
        conformal[key] = {
            "cal_scores_pos": sorted((1 - cal_proba[y_cal.values == 1]).round(5).tolist()),
            "cal_scores_neg": sorted((cal_proba[y_cal.values == 0]).round(5).tolist()),
        }

        pipeline.fit(X_tr, y_tr)
        hold_proba = pipeline.predict_proba(X_hold)[:, 1]
        hold_metrics = evaluate(y_hold.values, hold_proba, threshold)
        hold_default = evaluate(y_hold.values, hold_proba, 0.5)

        fpr, tpr, _ = roc_curve(y_hold.values, hold_proba)
        prec, rec, _ = precision_recall_curve(y_hold.values, hold_proba)

        fitted_models[key] = pipeline
        variant_metrics[key] = {
            "name": name, "cv": cv_metrics, "holdout": hold_metrics,
            "holdout_default_threshold": hold_default,
            "roc_curve": downsample_curve(fpr, tpr),
            "pr_curve": downsample_curve(rec, prec),
        }
        print(f"    holdout recall={hold_metrics['recall_bankrupt']:.3f} "
              f"f1={hold_metrics['f1_bankrupt']:.3f} pr_auc={hold_metrics['pr_auc']:.3f}")

    joblib.dump(fitted_models, ARTIFACTS_DIR / f"models_{variant_key}.joblib")
    background = X_tr.sample(n=min(100, len(X_tr)), random_state=RANDOM_STATE)
    joblib.dump(background, ARTIFACTS_DIR / f"background_{variant_key}.joblib")
    joblib.dump(conformal, ARTIFACTS_DIR / f"conformal_{variant_key}.joblib")

    feature_stats = {
        col: {"min": round(float(X_tr[col].min()), 6), "max": round(float(X_tr[col].max()), 6),
              "mean": round(float(X_tr[col].mean()), 6), "median": round(float(X_tr[col].median()), 6)}
        for col in selected
    }
    examples = {
        "distressed": {c: round(float(v), 6) for c, v in X_tr[y_tr == 1].median().items()},
        "healthy": {c: round(float(v), 6) for c, v in X_tr[y_tr == 0].median().items()},
    }
    return {
        "metrics": variant_metrics,
        "features": {
            "names": selected, "stats": feature_stats, "examples": examples,
            "class_counts": {"bankrupt": int(y_all.sum()), "non_bankrupt": int(len(y_all) - y_all.sum())},
            "descriptions": {c: FEATURE_DESCRIPTIONS.get(c.strip(), "Financial ratio from the Taiwan dataset.") for c in selected},
        },
    }


def main():
    all_metrics, all_features = {}, {}
    for variant_key, impute_kind in VARIANTS.items():
        result = train_variant(variant_key, impute_kind)
        all_metrics[variant_key] = result["metrics"]
        all_features[variant_key] = result["features"]
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    (ARTIFACTS_DIR / "features.json").write_text(json.dumps(all_features, indent=2))
    print(f"\nArtifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
