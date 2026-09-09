from pathlib import Path
import json
import warnings

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

MODEL_FILE = PROCESSED / "ner_final_modeling_table_8factor.csv"
OUTPUT_MODEL = PROCESSED / "step69_production_8factor_rf.joblib"
OUTPUT_CONFIG = PROCESSED / "step69_production_8factor_rf_config.json"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

CATEGORICAL_FEATURES = [
    "geomorph_origin",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 69 — PRODUCTION 8-FACTOR RANDOM FOREST")
print("=" * 70)


# ============================================================
# LOAD MODELING TABLE
# ============================================================

print("\nLoading modeling table...")

df = pd.read_csv(MODEL_FILE)

print(f"Rows loaded: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = FEATURES + [TARGET]

missing = [
    col for col in required
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("All required columns found.")


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nChecking missing values...")

missing_counts = df[required].isna().sum()

print(missing_counts.to_string())

if missing_counts.sum() > 0:
    raise ValueError(
        "Missing values detected. "
        "Production model cannot continue."
    )


# ============================================================
# CHECK DUPLICATES
# ============================================================

print("\nChecking duplicate coordinates...")

duplicates = df.duplicated(
    subset=["lat", "lon"]
).sum()

print(f"Duplicate coordinates: {duplicates}")

if duplicates > 0:
    raise ValueError(
        "Duplicate coordinates detected."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution:")

print(
    df[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# PREPARE X / y
# ============================================================

X = df[FEATURES].copy()
y = df[TARGET].copy()


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            NUMERIC_FEATURES,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)


# ============================================================
# RANDOM FOREST
# ============================================================

rf_params = {
    "n_estimators": 700,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": None,
}


print("\n" + "=" * 70)
print("PRODUCTION MODEL CONFIGURATION")
print("=" * 70)

print(json.dumps(rf_params, indent=2))


# ============================================================
# PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            RandomForestClassifier(
                **rf_params
            ),
        ),
    ]
)


# ============================================================
# FIT ON ALL COMPLETE SAMPLES
# ============================================================

print("\n" + "=" * 70)
print("FITTING PRODUCTION MODEL")
print("=" * 70)

print(f"Training samples: {len(X)}")

model.fit(X, y)

print("Production model fitted successfully.")


# ============================================================
# TRAINING PROBABILITIES
# ============================================================

train_prob = model.predict_proba(X)[:, 1]

print("\nTraining probability statistics:")

print(f"Minimum : {train_prob.min():.6f}")
print(f"Maximum : {train_prob.max():.6f}")
print(f"Mean    : {train_prob.mean():.6f}")
print(f"Median  : {pd.Series(train_prob).median():.6f}")


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

joblib.dump(
    model,
    OUTPUT_MODEL
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "model": "RandomForestClassifier",
    "purpose": "production susceptibility mapping",
    "n_estimators": 700,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "class_weight": None,
    "training_samples": int(len(df)),
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "removed_factor": "Level_I LUCC",
    "unavailable_factor": "Lithology",
    "evaluation_reference":
        "Step 65 untouched spatial test results",
}


with open(
    OUTPUT_CONFIG,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED MODEL")
print("=" * 70)

loaded_model = joblib.load(
    OUTPUT_MODEL
)

print(
    "Loaded model type:",
    type(loaded_model)
)

print(
    "Classifier:",
    loaded_model.named_steps["classifier"]
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 69 COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_MODEL)
print(OUTPUT_CONFIG)

print("\nThe production model is ready for")
print("8-factor susceptibility raster prediction.")

print("=" * 70)