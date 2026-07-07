"""
Data loading, missingness simulation, imputation, and feature selection.

NOTE: This is a direct, lightly-cleaned port of the original data_pre.py.
Logic is intentionally UNCHANGED for now (paths + structure only) —
methodology fixes (label-leakage in imputation, StratifiedKFold, scaled
KNN imputation, etc.) are planned for the next pass.
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from scipy.stats import pointbiserialr

from config import RAW_DATASET_PATH, RAW_DATASET_SHEET


def load_data(file_path=None, sheet_name=None):
    """Load the dataset and return the DataFrame."""
    file_path = file_path or RAW_DATASET_PATH
    sheet_name = sheet_name or RAW_DATASET_SHEET
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")
    return df


def create_missing_values(df, missing_percentage, seed=45):
    """
    Introduce missing values into all columns except the first one.
    Missing values are added uniformly across the dataset.
    """
    df_missing = df.copy()
    np.random.seed(seed)  # For reproducibility
    for col in df_missing.columns[1:]:
        mask = np.random.rand(len(df_missing)) < (missing_percentage / 100)
        df_missing.loc[mask, col] = np.nan
    return df_missing


def split_data(df):
    """
    Split the data based on the first column (Bankruptcy indicator).
    Returns three DataFrames:
      - bankrupt_df: rows where the indicator equals 1,
      - not_bankrupt_df: rows where the indicator equals 0,
      - missing_bankrupt_df: rows where the indicator is missing.
    """
    bankrupt_df = df[df.iloc[:, 0] == 1]
    not_bankrupt_df = df[df.iloc[:, 0] == 0]
    missing_bankrupt_df = df[df.iloc[:, 0].isna()]
    return bankrupt_df, not_bankrupt_df, missing_bankrupt_df


def knn_impute(df, n_neighbors):
    """
    Apply KNN imputation using sklearn's KNNImputer.
    This method fills missing values based on the closest feature distances.
    """
    df_imputed = df.copy()
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed.iloc[:, 1:] = imputer.fit_transform(df_imputed.iloc[:, 1:])
    return df_imputed


def impute_data(df):
    """
    Generate multiple imputation versions for the given DataFrame subset.
    Returns a dict of method_name -> imputed DataFrame.
    """
    df_mean = df.copy()
    for col in df_mean.columns[1:]:
        df_mean[col] = df_mean[col].fillna(df_mean[col].mean())

    df_median = df.copy()
    for col in df_median.columns[1:]:
        df_median[col] = df_median[col].fillna(df_median[col].median())

    df_knn_5 = knn_impute(df, 5)
    df_knn_15 = knn_impute(df, 15)
    df_knn_30 = knn_impute(df, 30)

    return {
        "Mean": df_mean,
        "Median": df_median,
        "KNN-5": df_knn_5,
        "KNN-15": df_knn_15,
        "KNN-30": df_knn_30,
    }


def calculate_closeness(original_df, imputed_data, missing_indices):
    """
    For each artificially-missing cell, count how often each imputation
    method's value is closest to the true original value.
    """
    closest_method_counts = {method: 0 for method in imputed_data}
    for col in original_df.columns[1:]:
        missing_mask = missing_indices[col]
        for i in original_df[missing_mask].index:
            original_value = original_df.loc[i, col]
            imputed_values = {method: imputed_data[method].loc[i, col] for method in imputed_data}
            differences = {method: abs(original_value - v) for method, v in imputed_values.items()}
            closest_method = min(differences, key=differences.get)
            closest_method_counts[closest_method] += 1
    return closest_method_counts


def calculate_correlations(original_df, imputed_data, missing_indices):
    """
    Average Pearson correlation (per method) between true and imputed
    values, over columns with >= 2 artificially-missing cells.
    """
    correlation_results = {}
    for method, df_imputed in imputed_data.items():
        correlations = []
        for col in original_df.columns[1:]:
            mask = missing_indices[col]
            if mask.sum() >= 2:
                orig_vals = original_df.loc[mask, col]
                imp_vals = df_imputed.loc[mask, col]
                if np.std(orig_vals) == 0 or np.std(imp_vals) == 0:
                    corr = 1.0
                else:
                    try:
                        corr = np.corrcoef(orig_vals, imp_vals)[0, 1]
                    except Exception as e:
                        print(f"Error calculating correlation for {method} / {col}: {e}")
                        corr = np.nan
                if not np.isnan(corr):
                    correlations.append(corr)
        correlation_results[method] = np.mean(correlations) if correlations else np.nan
    return correlation_results


def filter_correlations_by_threshold(original_df, imputed_data, missing_indices, threshold):
    """Per-column, per-method correlations that meet or exceed a threshold."""
    filtered_results = []
    for col in original_df.columns[1:]:
        mask = missing_indices[col]
        if mask.sum() >= 2:
            orig_vals = original_df.loc[mask, col]
            for method, df_imputed in imputed_data.items():
                imputed_vals = df_imputed.loc[mask, col]
                if np.std(orig_vals) == 0 or np.std(imputed_vals) == 0:
                    corr = 1.0
                else:
                    corr = np.corrcoef(orig_vals, imputed_vals)[0, 1]
                if corr >= threshold:
                    filtered_results.append((col, method, corr))
    return filtered_results


def point_biserial_correlation(df, binary_column, p_value_threshold=0.05,
                                correlation_threshold=0.2, top_n=20):
    """
    Point-biserial correlation between a binary target and each feature.
    Returns top_n features (sorted by |correlation|) that pass both
    the correlation and p-value thresholds.
    """
    correlation_results = {}
    for col in df.columns:
        if col != binary_column:
            corr, p_value = pointbiserialr(df[binary_column], df[col])
            if abs(corr) > correlation_threshold and p_value < p_value_threshold:
                correlation_results[col] = {"Correlation": corr, "p-value": p_value}

    sorted_features = sorted(
        correlation_results.items(), key=lambda x: abs(x[1]["Correlation"]), reverse=True
    )
    return dict(sorted_features[:top_n])


def filter_selected_features(original_df, selected_features, output_file_path, target_column="Bankrupt?"):
    """
    Filter the original DataFrame down to the target column + selected
    features and save to Excel.

    (Fixed vs. the original: was checking for a column named "Bankruptcy"
    that never existed, so the target column was silently dropped. Now
    it explicitly takes the real target column name.)
    """
    if isinstance(selected_features, dict):
        selected_columns = list(selected_features.keys())
    else:
        selected_columns = list(selected_features)

    if target_column in original_df.columns and target_column not in selected_columns:
        selected_columns = [target_column] + selected_columns

    filtered_df = original_df[selected_columns]
    with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
        filtered_df.to_excel(writer, index=False)

    print(f"Filtered data saved to {output_file_path}")
    return filtered_df
