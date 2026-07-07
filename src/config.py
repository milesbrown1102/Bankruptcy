"""
Central configuration for file paths and shared constants.
Keeping these in one place means you can move the project anywhere
(different machine, different folder name) and only ever edit this file.
"""

from pathlib import Path

# Project root = one level up from src/
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

# Make sure these exist even on a fresh clone
for d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RAW_DATASET_PATH = DATA_RAW_DIR / "Taiwan_Bankruptcy_Dataset.xlsx"
RAW_DATASET_SHEET = "data"

IMPUTED_DATA_PATH = DATA_PROCESSED_DIR / "Imputed_Data_By_Method.xlsx"
FILTERED_FEATURES_PATH = DATA_PROCESSED_DIR / "Filtered_Top_Features_with_Bankruptcy_Indicator.xlsx"

TARGET_COLUMN = "Bankrupt?"

# Default settings (unchanged from your original pipeline for now —
# imbalance/CV methodology fixes come in the next pass)
DEFAULT_MISSING_PERCENTAGE = 10
DEFAULT_TOP_N_FEATURES = 20
RANDOM_STATE = 42
