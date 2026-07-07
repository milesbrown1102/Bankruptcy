"""
Leak-free data preparation for the modelling pipeline.

This REPLACES the label-conditioned imputation + full-dataset feature
selection used in the original research script (src/), for the purposes
of training deployable models. The research script is kept as-is for the
imputation *study*; this module is what the served models are built on.

Key differences (the methodology fixes):
  1. Imputation is fit on the TRAINING data only and applied to test data,
     and it NEVER sees the label (no bankrupt/non-bankrupt split before
     imputing). Imputation is a step inside each model Pipeline, so under
     cross-validation it is refit per fold automatically.
  2. Feature selection (point-biserial top-k) is computed on the TRAINING
     split only. The same selected columns are then applied to the holdout.
     No test-set information leaks into which features are chosen.
  3. KNN imputation is preceded by scaling inside the pipeline, so distance
     calculations aren't dominated by large-magnitude ratios.

Because feature selection is now fold-dependent in principle, but we need a
single stable feature set to expose in the app's input form, we select
features once on the training split of a fixed stratified split and reuse
that set. This is the standard, defensible compromise: selection uses only
training data, and the holdout is never consulted.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATASET_PATH = ROOT_DIR / "data" / "raw" / "Taiwan_Bankruptcy_Dataset.xlsx"
RAW_SHEET = "data"
TARGET_COLUMN = "Bankrupt?"
RANDOM_STATE = 42


def load_raw():
    """Load the full raw dataset (all 95 features + target)."""
    df = pd.read_excel(RAW_DATASET_PATH, sheet_name=RAW_SHEET, engine="openpyxl")
    # Drop the single stray missing cell in the raw file by column-median
    # (this is real missingness in the source file, not synthetic).
    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    return df


def select_features_on_training(X_train, y_train, top_n=20,
                                 p_value_threshold=0.05,
                                 correlation_threshold=0.2):
    """
    Point-biserial feature selection using TRAINING data only.
    Returns the ordered list of selected feature names.
    """
    results = {}
    for col in X_train.columns:
        # Guard against constant columns (undefined correlation)
        if X_train[col].std() == 0:
            continue
        corr, p_value = pointbiserialr(y_train, X_train[col])
        if abs(corr) > correlation_threshold and p_value < p_value_threshold:
            results[col] = abs(corr)

    ordered = sorted(results.items(), key=lambda kv: kv[1], reverse=True)
    selected = [name for name, _ in ordered[:top_n]]

    # Fallback: if the thresholds are too strict and we get very few,
    # relax to just the top_n by absolute correlation.
    if len(selected) < min(top_n, 5):
        all_corrs = {}
        for col in X_train.columns:
            if X_train[col].std() == 0:
                continue
            corr, _ = pointbiserialr(y_train, X_train[col])
            all_corrs[col] = abs(corr)
        ordered = sorted(all_corrs.items(), key=lambda kv: kv[1], reverse=True)
        selected = [name for name, _ in ordered[:top_n]]

    return selected
