"""
Bankruptcy Prediction Desktop App — v2 UI

What changed vs. your original bankruptcy_gui.py:
  - ttkbootstrap theme (modern flat dark UI instead of raw tkinter)
  - Tabs (Predict / Model Comparison) instead of a popup Toplevel window
  - Models are trained ONCE at startup and cached, not retrained on every
    click of "Predict" (same models/params/data — just not refit each time)
  - Cleaner results display (per-model cards + consensus bar)

What did NOT change (on purpose — methodology pass comes next):
  - Still plain KFold, still accuracy-only, still the same 6 models
  - Still reads from Filtered_Top_Features_with_Bankruptcy_Indicator.xlsx
    produced by run_pipeline.py

Run:
    python run_pipeline.py      # once, to generate the processed files
    python gui_app.py           # then launch the app
"""

import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import messagebox
import ttkbootstrap as ttb
from ttkbootstrap.constants import *

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

from config import FILTERED_FEATURES_PATH, RANDOM_STATE

MODEL_COLORS = {
    "Logistic Regression": "#5DADE2",
    "Random Forest": "#48C9B0",
    "Gradient Boosting": "#F5B041",
    "SVM": "#EC7063",
    "Decision Tree": "#AF7AC5",
    "Neural Network": "#52BE80",
}

DATASET_LABELS = {"Original": "Original Top Features", "KNN-30": "KNN-30 Top Features"}


def build_model_specs():
    """Fresh, unfitted model instances + whether they need scaled input."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE), False),
        "Gradient Boosting": (GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE), False),
        "SVM": (SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE), False),
        "Neural Network": (MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000,
                                          activation="relu", solver="adam", random_state=RANDOM_STATE), True),
    }


class ModelBundle:
    """Holds fitted models + scaler + CV/holdout accuracy for one dataset variant."""

    def __init__(self, df):
        self.df = df
        self.X = df.drop(columns=[df.columns[0]])
        self.y = df[df.columns[0]]
        self.feature_names = list(self.X.columns)
        self.scaler = StandardScaler().fit(self.X)
        self.fitted = {}          # name -> fitted model
        self.needs_scaling = {}   # name -> bool
        self.cv_acc = {}          # name -> mean CV accuracy
        self.holdout_acc = {}     # name -> holdout accuracy

    def train(self):
        X, y = self.X, self.y
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns, index=X.index)

        X_main, X_holdout, y_main, y_holdout = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        X_main_scaled = pd.DataFrame(self.scaler.transform(X_main), columns=X.columns, index=X_main.index)
        X_holdout_scaled = pd.DataFrame(self.scaler.transform(X_holdout), columns=X.columns, index=X_holdout.index)

        kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for name, (model, needs_scaling) in build_model_specs().items():
            fold_accs = []
            for train_idx, test_idx in kfold.split(X_main):
                Xtr = X_main_scaled.iloc[train_idx] if needs_scaling else X_main.iloc[train_idx]
                Xte = X_main_scaled.iloc[test_idx] if needs_scaling else X_main.iloc[test_idx]
                ytr, yte = y_main.iloc[train_idx], y_main.iloc[test_idx]
                m = build_model_specs()[name][0]
                m.fit(Xtr, ytr)
                fold_accs.append(accuracy_score(yte, m.predict(Xte)))
            self.cv_acc[name] = float(np.mean(fold_accs))

            # Final model: trained on ALL data (not just the 80% "main" split)
            # so the deployed/predicting model uses every available row.
            Xtr_full = X_scaled if needs_scaling else X
            final_model = build_model_specs()[name][0]
            final_model.fit(Xtr_full, y)
            self.fitted[name] = final_model
            self.needs_scaling[name] = needs_scaling

            # Holdout accuracy still reported for reference (trained on the
            # 80% split only, matching your original evaluation protocol).
            eval_model = build_model_specs()[name][0]
            Xtr_eval = X_main_scaled if needs_scaling else X_main
            Xho_eval = X_holdout_scaled if needs_scaling else X_holdout
            eval_model.fit(Xtr_eval, y_main)
            self.holdout_acc[name] = accuracy_score(y_holdout, eval_model.predict(Xho_eval))

    def predict_sample(self, sample_dict):
        sample_df = pd.DataFrame([sample_dict])[self.feature_names]
        sample_scaled = pd.DataFrame(self.scaler.transform(sample_df), columns=self.feature_names)
        results = []
        for name, model in self.fitted.items():
            X_in = sample_scaled if self.needs_scaling[name] else sample_df
            proba = model.predict_proba(X_in)[0][1]
            results.append((name, proba))
        return results


class BankruptcyApp:
    def __init__(self):
        self.root = ttb.Window(themename="darkly")
        self.root.title("Bankruptcy Prediction System")
        self.root.geometry("1200x850")

        self.bundles = {}  # "Original" / "KNN-30" -> ModelBundle
        self.dataset_var = tk.StringVar(value="Original")
        self.entries = {}

        self._build_layout()
        self._load_data_and_train_async()

    # ---------- layout ----------

    def _build_layout(self):
        header = ttb.Frame(self.root, padding=15)
        header.pack(fill=X)
        ttb.Label(header, text="Bankruptcy Prediction System", font=("Helvetica", 20, "bold")).pack(side=LEFT)
        self.status_label = ttb.Label(header, text="Loading data & training models...", bootstyle="warning")
        self.status_label.pack(side=RIGHT)

        self.notebook = ttb.Notebook(self.root, bootstyle="dark")
        self.notebook.pack(fill=BOTH, expand=True, padx=15, pady=10)

        self.predict_tab = ttb.Frame(self.notebook, padding=10)
        self.compare_tab = ttb.Frame(self.notebook, padding=10)
        self.notebook.add(self.predict_tab, text="  Predict  ")
        self.notebook.add(self.compare_tab, text="  Model Comparison  ")

        self._build_predict_tab()
        self._build_compare_tab_placeholder()

    def _build_predict_tab(self):
        container = ttb.Frame(self.predict_tab)
        container.pack(fill=BOTH, expand=True)

        left = ttb.Labelframe(container, text="Inputs", padding=10, bootstyle="info")
        left.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 10))

        ds_frame = ttb.Frame(left)
        ds_frame.pack(fill=X, pady=(0, 10))
        ttb.Label(ds_frame, text="Dataset variant:").pack(side=LEFT, padx=(0, 10))
        for key, label in DATASET_LABELS.items():
            ttb.Radiobutton(ds_frame, text=label, variable=self.dataset_var, value=key,
                             bootstyle="info-toolbutton").pack(side=LEFT, padx=5)

        self.feature_canvas = tk.Canvas(left, highlightthickness=0)
        scrollbar = ttb.Scrollbar(left, orient=VERTICAL, command=self.feature_canvas.yview, bootstyle="round")
        self.scroll_frame = ttb.Frame(self.feature_canvas)
        self.scroll_frame.bind("<Configure>", lambda e: self.feature_canvas.configure(
            scrollregion=self.feature_canvas.bbox("all")))
        self.feature_canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.feature_canvas.configure(yscrollcommand=scrollbar.set)
        self.feature_canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)

        btn_frame = ttb.Frame(left)
        btn_frame.pack(fill=X, pady=10)
        self.predict_btn = ttb.Button(btn_frame, text="Predict with All Models",
                                       command=self._on_predict, bootstyle="success", state=DISABLED)
        self.predict_btn.pack(fill=X)

        right = ttb.Labelframe(container, text="Results", padding=10, bootstyle="info")
        right.pack(side=LEFT, fill=BOTH, expand=True)
        self.results_frame = ttb.Frame(right)
        self.results_frame.pack(fill=BOTH, expand=True)
        self.placeholder_label = ttb.Label(
            self.results_frame,
            text="Enter feature values and click Predict.",
            bootstyle="secondary"
        )
        self.placeholder_label.pack(pady=20)

    def _build_compare_tab_placeholder(self):
        self.compare_placeholder = ttb.Label(
            self.compare_tab, text="Training models — charts will appear here shortly...",
            bootstyle="warning"
        )
        self.compare_placeholder.pack(pady=30)

    # ---------- data + training ----------

    def _load_data_and_train_async(self):
        thread = threading.Thread(target=self._load_data_and_train, daemon=True)
        thread.start()

    def _load_data_and_train(self):
        df_original = pd.read_excel(FILTERED_FEATURES_PATH, sheet_name="Original Top Features")
        df_knn30 = pd.read_excel(FILTERED_FEATURES_PATH, sheet_name="KNN-30 Top Features")

        for key, df in {"Original": df_original, "KNN-30": df_knn30}.items():
            bundle = ModelBundle(df)
            bundle.train()
            self.bundles[key] = bundle

        self.root.after(0, self._on_training_complete)

    def _on_training_complete(self):
        self.status_label.configure(text="Models ready", bootstyle="success")
        self.predict_btn.configure(state=NORMAL)
        self._populate_feature_inputs()
        self._render_comparison_charts()

    def _populate_feature_inputs(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.entries = {}

        bundle = self.bundles[self.dataset_var.get()]
        for feature in bundle.feature_names:
            ttb.Label(self.scroll_frame, text=feature, wraplength=350).pack(anchor="w", pady=(6, 0))
            entry = ttb.Entry(self.scroll_frame)
            entry.pack(fill=X, pady=(0, 4))
            self.entries[feature] = entry

        # Re-populate whenever the dataset radio button changes
        self.dataset_var.trace_add("write", lambda *a: self._populate_feature_inputs())

    # ---------- prediction ----------

    def _on_predict(self):
        bundle = self.bundles[self.dataset_var.get()]
        sample = {}
        try:
            for feature in bundle.feature_names:
                sample[feature] = float(self.entries[feature].get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values for every feature.")
            return

        results = bundle.predict_sample(sample)
        self._render_results(results)

    def _render_results(self, results):
        for widget in self.results_frame.winfo_children():
            widget.destroy()

        for name, proba in results:
            row = ttb.Frame(self.results_frame, padding=6)
            row.pack(fill=X, pady=3)
            label = "Bankrupt" if proba > 0.5 else "Not Bankrupt"
            style = "danger" if proba > 0.5 else "success"
            ttb.Label(row, text=name, width=20, font=("Helvetica", 11, "bold")).pack(side=LEFT)
            ttb.Label(row, text=f"{label}  ({proba:.1%})", bootstyle=style,
                      font=("Helvetica", 11)).pack(side=LEFT, padx=10)
            bar = ttb.Progressbar(row, value=proba * 100, bootstyle=f"{style}-striped", length=200)
            bar.pack(side=RIGHT)

        votes = sum(1 for _, p in results if p > 0.5)
        consensus_text = (
            f"Likely Bankrupt ({votes}/{len(results)} models agree)" if votes > len(results) / 2
            else f"Likely Not Bankrupt ({len(results) - votes}/{len(results)} models agree)" if votes < len(results) / 2
            else "Split decision"
        )
        avg_proba = np.mean([p for _, p in results])

        sep = ttb.Separator(self.results_frame)
        sep.pack(fill=X, pady=10)
        ttb.Label(self.results_frame, text=f"Consensus: {consensus_text}",
                  font=("Helvetica", 13, "bold"), bootstyle="info").pack(anchor="w")
        ttb.Label(self.results_frame, text=f"Average bankruptcy probability: {avg_proba:.1%}",
                  font=("Helvetica", 11)).pack(anchor="w", pady=(4, 0))

    # ---------- comparison charts ----------

    def _render_comparison_charts(self):
        self.compare_placeholder.destroy()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
        fig.patch.set_alpha(0)
        fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold")
        axes = axes.flatten()

        panels = [
            ("Original", "cv_acc", "Original — Cross-Validation"),
            ("Original", "holdout_acc", "Original — Holdout"),
            ("KNN-30", "cv_acc", "KNN-30 — Cross-Validation"),
            ("KNN-30", "holdout_acc", "KNN-30 — Holdout"),
        ]
        for ax, (dataset_key, metric_attr, title) in zip(axes, panels):
            bundle = self.bundles[dataset_key]
            scores = getattr(bundle, metric_attr)
            names = list(scores.keys())
            values = [scores[n] for n in names]
            colors = [MODEL_COLORS.get(n, "#888888") for n in names]
            bars = ax.bar(names, values, color=colors)
            ax.set_ylim(0, 1)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Accuracy")
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h + 0.01),
                            ha="center", fontsize=8)

        plt.tight_layout()

        chart_frame = ttb.Frame(self.compare_tab)
        chart_frame.pack(fill=BOTH, expand=True)
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = BankruptcyApp()
    app.run()
