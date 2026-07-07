# RiskDesk — Bankruptcy Prediction System

Corporate bankruptcy risk prediction on the Taiwan Bankruptcy Dataset
(6,819 firms, 220 bankrupt ≈ 3.2%). Full-stack: FastAPI model-serving
backend + React analytics frontend.

## What's in v3 (this version)

**Backend methodology fixes (the important part):**
- `StratifiedKFold` cross-validation (critical at 3.2% minority prevalence)
- Class imbalance handled per model: `class_weight='balanced'` (LR/RF/SVM/DT),
  `scale_pos_weight` (XGBoost), SMOTE inside an imblearn pipeline so it only
  ever resamples training folds (GB/MLP)
- Full metric suite on the bankrupt class: precision, recall, F1, ROC-AUC,
  PR-AUC, MCC, balanced accuracy, confusion matrices
- Per-model decision thresholds tuned to maximize bankrupt-class F1 on
  cross-validated training predictions, evaluated once on an untouched
  stratified 20% holdout
- Scaling lives inside each model pipeline (fit on training data only)
- 7 models (adds XGBoost), trained once, persisted to `backend/artifacts/`
- FastAPI serves saved artifacts — nothing retrains at request time

**Frontend (React + Vite + Recharts):**
- Predict tab: financial-ratio form with one-click healthy / distressed /
  median example loaders, per-model probability bars with threshold ticks,
  consensus gauge + vote strip, tuned-vs-0.5 threshold toggle
- Model analytics tab: metric selector (F1/recall/precision/PR-AUC/ROC-AUC/
  MCC/balanced acc/accuracy) with hoverable CV-vs-holdout bars, interactive
  ROC and PR curves, confusion matrix grid
- Dataset variant switch (original vs KNN-30 imputed) in the top bar

## Requirements

- Python 3.10+ (tested on 3.13)
- Node.js 18+ and npm (in WSL: `sudo apt install nodejs npm` or use nvm)

## Setup

```bash
# From the project root
python3 -m venv venv           # skip if you already have the venv
source venv/bin/activate
pip install -r backend/requirements.txt
pip install -r requirements.txt          # research-pipeline deps (only if you'll rerun src/)

cd frontend && npm install && cd ..
```

## Running the app

Pre-trained artifacts are included in `backend/artifacts/`, so you can go
straight to serving. Two terminals:

**Terminal 1 — API:**
```bash
source venv/bin/activate
cd backend
uvicorn api.main:app --reload --port 8000
```

**Terminal 2 — frontend:**
```bash
cd frontend
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173). The dev server
proxies `/api` to port 8000 automatically.

## Retraining the models

```bash
source venv/bin/activate
cd backend/ml
python train.py
```
Takes a few minutes (7 models × 5-fold stratified CV × 2 dataset variants).
Reads `data/processed/Filtered_Top_Features_with_Bankruptcy_Indicator.xlsx`;
if that's missing, generate it first with the research pipeline:
```bash
cd src && python run_pipeline.py --missing-pct 10 --top-n 20 --no-show
```

## Results snapshot (holdout, tuned thresholds)

Accuracy is ~96–97% for everything — including a model that predicts
"never bankrupt" — which is exactly why we don't lead with it. The honest
numbers on the bankrupt class (original-features variant):

| Model | Recall | F1 | PR-AUC |
|---|---|---|---|
| Random Forest | 0.61 | 0.46 | 0.46 |
| Neural Network | 0.57 | 0.46 | 0.37 |
| XGBoost | 0.50 | 0.39 | 0.41 |
| Logistic Regression | 0.48 | 0.43 | 0.41 |

(KNN-30 variant scores higher across the board — see the dashboard — but
note the caveat below before quoting those numbers.)

## Known caveats / next milestones

- The KNN-30 variant inherits the **label-conditioned imputation** from the
  research pipeline (bankrupt / non-bankrupt subsets imputed separately),
  so its metrics are optimistic. The original-features variant has no such
  issue. Fixing the imputation study design is a planned milestone.
- Feature selection (point-biserial) is computed on the full dataset rather
  than inside each CV fold — a mild selection-leakage that's standard to
  fix in the paper-grade version.
- Planned: SHAP explainability endpoint + waterfall in the verdict panel,
  counterfactual explanations, conformal prediction intervals,
  cost-sensitive decision layer.

## Project structure

```
├── backend/
│   ├── ml/train.py          # training pipeline (run to regenerate artifacts)
│   ├── api/main.py          # FastAPI app
│   ├── artifacts/           # saved models + metrics.json + features.json
│   └── requirements.txt
├── frontend/                # React + Vite + Recharts app
│   └── src/
│       ├── App.jsx
│       └── components/      # Predict, Dashboard, Gauge
├── src/                     # research pipeline (imputation study) + legacy desktop GUI
└── data/
```
