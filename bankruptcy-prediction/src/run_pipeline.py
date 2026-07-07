"""
End-to-end data prep pipeline: load -> inject missingness -> impute ->
compare imputation quality -> select top features -> save processed files
+ plots.

Run from the src/ folder:
    python run_pipeline.py
    python run_pipeline.py --missing-pct 15 --top-n 20

Logic is unchanged from your original script — this pass is about making
it runnable cleanly from VS Code (config-driven paths, CLI args instead
of blocking input() calls, plots saved to disk as well as shown).
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    OUTPUTS_DIR,
    IMPUTED_DATA_PATH,
    FILTERED_FEATURES_PATH,
    TARGET_COLUMN,
    DEFAULT_MISSING_PERCENTAGE,
    DEFAULT_TOP_N_FEATURES,
)
from data_pipeline import (
    load_data,
    create_missing_values,
    split_data,
    impute_data,
    calculate_closeness,
    calculate_correlations,
    point_biserial_correlation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Bankruptcy data prep pipeline")
    parser.add_argument("--missing-pct", type=float, default=DEFAULT_MISSING_PERCENTAGE,
                         help="Percent of missing values to synthetically introduce")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_FEATURES,
                         help="Number of top features to select via point-biserial correlation")
    parser.add_argument("--no-show", action="store_true",
                         help="Save plots to outputs/ without opening interactive windows")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load
    df = load_data()
    num_rows, num_cols = df.shape
    print(f"Loaded dataset: {num_rows} rows, {num_cols} columns")

    # 2. Introduce missingness
    total_cells = num_rows * (num_cols - 1)
    missing_values_count = int((args.missing_pct / 100) * total_cells)
    print(f"Introducing {args.missing_pct}% missing -> {missing_values_count} cells")
    df_with_missing = create_missing_values(df, args.missing_pct)

    # 3. Split by target
    bankrupt_df, not_bankrupt_df, missing_bankrupt_df = split_data(df_with_missing)
    print(f"Bankrupt: {bankrupt_df.shape[0]} | Non-bankrupt: {not_bankrupt_df.shape[0]} "
          f"| Missing indicator: {missing_bankrupt_df.shape[0]}")

    # 4. Impute each subset (NOTE: known label-leakage issue, flagged for next pass)
    print("Imputing bankrupt subset...")
    imputed_bankrupt = impute_data(bankrupt_df)
    print("Imputing non-bankrupt subset...")
    imputed_nonbankrupt = impute_data(not_bankrupt_df)

    methods = list(imputed_bankrupt.keys())
    imputed_data = {
        method: pd.concat([imputed_bankrupt[method], imputed_nonbankrupt[method]])
        for method in methods
    }

    with pd.ExcelWriter(IMPUTED_DATA_PATH) as writer:
        for method in methods:
            imputed_data[method].iloc[:len(bankrupt_df)].to_excel(
                writer, sheet_name=f"{method}_Bankrupt", index=False)
            imputed_data[method].iloc[len(bankrupt_df):].to_excel(
                writer, sheet_name=f"{method}_NonBankrupt", index=False)
    print(f"Imputed data saved to {IMPUTED_DATA_PATH}")

    # 5. Imputation quality checks
    missing_indices = df_with_missing.isna()
    closeness = calculate_closeness(df, imputed_data, missing_indices)
    print("\nCloseness (times each method was nearest the true value):")
    for method, count in closeness.items():
        print(f"  {method}: {count} / {missing_values_count}")

    correlations = calculate_correlations(df, imputed_data, missing_indices)
    print("\nAverage correlation to true values, by method:")
    for method, corr in correlations.items():
        print(f"  {method}: {corr:.4f}")
    best_method = max(correlations, key=correlations.get)
    print(f"Best imputation method by correlation: {best_method}")

    # 6. Point-biserial feature selection
    pb_results = point_biserial_correlation(df, TARGET_COLUMN, top_n=args.top_n)
    print(f"\nTop {args.top_n} features by point-biserial correlation:")
    for feature, stats in pb_results.items():
        print(f"  {feature}: r={stats['Correlation']:.4f}, p={stats['p-value']:.4f}")

    top_features = [TARGET_COLUMN] + list(pb_results.keys())
    filtered_original = df[top_features]
    filtered_knn30 = imputed_data["KNN-30"][top_features]

    with pd.ExcelWriter(FILTERED_FEATURES_PATH) as writer:
        filtered_original.to_excel(writer, sheet_name="Original Top Features", index=False)
        filtered_knn30.to_excel(writer, sheet_name="KNN-30 Top Features", index=False)
    print(f"Filtered top-feature datasets saved to {FILTERED_FEATURES_PATH}")

    # 7. Plots
    _plot_closeness_pie(closeness, missing_values_count, args.no_show)
    _plot_correlation_bar(correlations, args.no_show)
    _plot_feature_heatmap(df, pb_results, TARGET_COLUMN, args.no_show, suffix="original")
    _plot_feature_heatmap(imputed_data["KNN-30"], pb_results, TARGET_COLUMN, args.no_show, suffix="knn30")

    print("\nPipeline complete.")


def _plot_closeness_pie(closeness, total, no_show):
    plt.figure(figsize=(8, 8))
    methods = list(closeness.keys())
    counts = list(closeness.values())
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33']
    plt.pie(counts, labels=methods, autopct='%1.1f%%', startangle=90,
            colors=colors[:len(methods)], wedgeprops={'edgecolor': 'black'},
            textprops={'fontsize': 14})
    plt.title('Closeness Distribution by Imputation Method', fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "closeness_pie.png", dpi=150)
    if not no_show:
        plt.show()
    plt.close()


def _plot_correlation_bar(correlations, no_show):
    methods = [m for m, c in correlations.items() if isinstance(c, (int, float)) and not np.isnan(c)]
    values = [c for c in correlations.values() if isinstance(c, (int, float)) and not np.isnan(c)]
    if not methods:
        return
    plt.figure(figsize=(10, 6))
    colors = ['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FF8C33']
    plt.bar(methods, values, color=colors[:len(methods)])
    plt.xlabel('Imputation Method', fontsize=14)
    plt.ylabel('Correlation to True Value', fontsize=14)
    plt.title('Imputation Accuracy by Method', fontsize=16)
    plt.xticks(rotation=45, fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "correlation_bar.png", dpi=150)
    if not no_show:
        plt.show()
    plt.close()


def _plot_feature_heatmap(df, pb_results, target_column, no_show, suffix, top_n=10):
    top_features = sorted(pb_results.items(), key=lambda x: abs(x[1]['Correlation']), reverse=True)[:top_n]
    feature_names = [f[0] for f in top_features]
    feature_names = [f for f in feature_names if f in df.columns]
    if len(feature_names) < 2:
        return

    corr_matrix = df[feature_names].corr()
    abbrev = {f: chr(65 + i) for i, f in enumerate(corr_matrix.columns)}
    labels = [abbrev[f] for f in corr_matrix.columns]

    plt.figure(figsize=(12, 9))
    ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                      cbar_kws={'label': 'Correlation'}, fmt='.2f',
                      linewidths=0.5, linecolor='gray', annot_kws={"size": 10},
                      xticklabels=labels, yticklabels=labels, square=True, robust=True)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.title(f'Top Feature Correlation Heatmap ({suffix})', fontsize=15)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / f"feature_heatmap_{suffix}.png", dpi=150)
    if not no_show:
        plt.show()
    plt.close()

    print(f"\nAbbreviation key ({suffix}):")
    for full_name, ab in abbrev.items():
        print(f"  {ab} : {full_name}")


if __name__ == "__main__":
    main()
