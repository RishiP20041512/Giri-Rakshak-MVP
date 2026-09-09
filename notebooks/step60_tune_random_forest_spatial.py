"""
STEP 60 — TUNE RANDOM FOREST USING SPATIAL CROSS-VALIDATION

Purpose:
- Tune Random Forest hyperparameters using ONLY the 788 training
  observations.
- Use 5-fold StratifiedGroupKFold with the existing 0.50-degree
  spatial groups.
- Keep the 203-observation final test set completely untouched.
- Compare a small, scientifically reasonable hyperparameter grid.
- Select the configuration primarily using mean spatial ROC-AUC,
  with PR-AUC and variability also reported.

Important:
- The final test set is NOT used for tuning.
- Preprocessing is fitted independently within every fold.
- Categorical variables are one-hot encoded inside each pipeline.
- Lithology is not included because it is currently unavailable.

Input:
    processed/ner_train_modeling.csv
    processed/ner_test_modeling.csv

Outputs:
    processed/step60_rf_tuning_results.csv
    processed/step60_rf_tuning_summary.csv
    processed/step60_selected_rf_config.csv
"""


import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedGroupKFold

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = r"processed\ner_train_modeling.csv"
TEST_FILE = r"processed\ner_test_modeling.csv"

RESULTS_FILE = r"processed\step60_rf_tuning_results.csv"
SUMMARY_FILE = r"processed\step60_rf_tuning_summary.csv"
SELECTED_FILE = r"processed\step60_selected_rf_config.csv"


# ============================================================
# PREDICTORS
# ============================================================

NUMERICAL_FEATURES = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density"
]

CATEGORICAL_FEATURES = [
    "geomorph_origin",
    "Level_I"
]

FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET = "label"

N_SPLITS = 5

RANDOM_STATE = 42


# ============================================================
# RANDOM FOREST SEARCH GRID
# ============================================================
#
# We deliberately keep this grid small.
#
# n_estimators:
#   Controls number of trees.
#
# max_features:
#   Controls number of predictors considered at each split.
#
# min_samples_leaf:
#   Controls minimum observations in a terminal node.
#   Increasing this value generally reduces overfitting.
#
# max_depth:
#   Limits tree depth.
#
# Total configurations:
#
#   2 * 2 * 3 * 2 = 24
#
# With 5 spatial folds:
#
#   24 * 5 = 120 model fits
#
# ============================================================

PARAM_GRID = [

    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "max_depth": None
    },

    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": None
    },

    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_depth": None
    },

    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": 15
    },

    {
        "n_estimators": 300,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_depth": 15
    },

    {
        "n_estimators": 300,
        "max_features": 0.5,
        "min_samples_leaf": 1,
        "max_depth": None
    },

    {
        "n_estimators": 300,
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "max_depth": None
    },

    {
        "n_estimators": 300,
        "max_features": 0.5,
        "min_samples_leaf": 5,
        "max_depth": None
    },

    {
        "n_estimators": 300,
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "max_depth": 15
    },

    {
        "n_estimators": 300,
        "max_features": 0.5,
        "min_samples_leaf": 5,
        "max_depth": 15
    },

    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "max_depth": None
    },

    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": None
    },

    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_depth": None
    },

    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": 15
    },

    {
        "n_estimators": 500,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_depth": 15
    },

    {
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 1,
        "max_depth": None
    },

    {
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "max_depth": None
    },

    {
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 5,
        "max_depth": None
    },

    {
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 2,
        "max_depth": 15
    },

    {
        "n_estimators": 500,
        "max_features": 0.5,
        "min_samples_leaf": 5,
        "max_depth": 15
    },

    {
        "n_estimators": 700,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": None
    },

    {
        "n_estimators": 700,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_depth": None
    },

    {
        "n_estimators": 700,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "max_depth": 15
    },

    {
        "n_estimators": 700,
        "max_features": "sqrt",
        "min_samples_leaf": 5,
        "max_depth": 15
    }
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 60 — TUNE RANDOM FOREST USING SPATIAL CROSS-VALIDATION")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

print("\n[1] Checking input files...")
print("-" * 70)

for file_path in [TRAIN_FILE, TEST_FILE]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(f"  Found: {file_path}")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[2] Loading datasets...")
print("-" * 70)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"  Training observations  : {len(train)}")
print(f"  Final test observations: {len(test)}")


# ============================================================
# 3. PROTECT FINAL TEST SET
# ============================================================

print("\n[3] Protecting final test set...")
print("-" * 70)

print(
    "  The final test set will NOT be used for hyperparameter tuning."
)

print(
    f"  Reserved test observations: {len(test)}"
)


# ============================================================
# 4. VERIFY COLUMNS
# ============================================================

print("\n[4] Checking required columns...")
print("-" * 70)

required_columns = (
    FEATURES
    + [
        TARGET,
        "lat",
        "lon",
        "spatial_group"
    ]
)

for dataset_name, dataset in [
    ("training", train),
    ("test", test)
]:

    missing = [
        col
        for col in required_columns
        if col not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {dataset_name}:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

print("  All required columns present.")


# ============================================================
# 5. PREPARE TRAINING DATA
# ============================================================

print("\n[5] Preparing spatial CV data...")
print("-" * 70)

X = train[FEATURES].copy()
y = train[TARGET].copy()
groups = train["spatial_group"].copy()

print(f"  X shape          : {X.shape}")
print(f"  Target shape     : {y.shape}")
print(f"  Spatial groups   : {groups.nunique()}")


# ============================================================
# 6. VERIFY TRAINING DATA
# ============================================================

print("\n[6] Validating training data...")
print("-" * 70)

if X.isna().sum().sum() != 0:
    raise ValueError(
        "Missing predictor values found."
    )

if y.isna().sum() != 0:
    raise ValueError(
        "Missing target values found."
    )

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Target must contain classes 0 and 1."
    )

print(
    f"  Background : {int((y == 0).sum())}"
)

print(
    f"  Landslide  : {int((y == 1).sum())}"
)

print(
    f"  LS ratio   : {y.mean():.4f}"
)

print("  Training data validation PASSED.")


# ============================================================
# 7. CREATE SPATIAL CV
# ============================================================

print("\n[7] Creating spatial cross-validation...")
print("-" * 70)

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

print(
    f"  Method : StratifiedGroupKFold"
)

print(
    f"  Folds  : {N_SPLITS}"
)

print(
    "  Group  : spatial_group"
)


# ============================================================
# 8. CHECK FOLD INTEGRITY
# ============================================================

print("\n[8] Checking fold integrity...")
print("-" * 70)

fold_splits = list(
    cv.split(
        X,
        y,
        groups
    )
)

for fold_number, (train_idx, val_idx) in enumerate(
    fold_splits,
    start=1
):

    train_groups = set(
        groups.iloc[train_idx]
    )

    val_groups = set(
        groups.iloc[val_idx]
    )

    overlap = train_groups.intersection(
        val_groups
    )

    if overlap:
        raise RuntimeError(
            f"Spatial overlap detected in fold {fold_number}."
        )

    train_classes = set(
        y.iloc[train_idx].unique()
    )

    val_classes = set(
        y.iloc[val_idx].unique()
    )

    if train_classes != {0, 1}:
        raise RuntimeError(
            f"Fold {fold_number} training data "
            "does not contain both classes."
        )

    if val_classes != {0, 1}:
        raise RuntimeError(
            f"Fold {fold_number} validation data "
            "does not contain both classes."
        )

print("  All spatial folds are valid.")
print("  Spatial overlap = 0.")
print("  Both classes present in every fold.")


# ============================================================
# 9. ENCODER FACTORY
# ============================================================

def make_encoder():

    try:

        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False
        )

    except TypeError:

        return OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse=False
        )


# ============================================================
# 10. PIPELINE FACTORY
# ============================================================

def build_rf_pipeline(params):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        )
                    ]
                ),
                NUMERICAL_FEATURES
            ),

            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),
                        (
                            "onehot",
                            make_encoder()
                        )
                    ]
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        min_samples_leaf=params["min_samples_leaf"],
        max_depth=params["max_depth"],
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )


# ============================================================
# 11. DISPLAY SEARCH SIZE
# ============================================================

print("\n[9] Hyperparameter search configuration...")
print("-" * 70)

print(
    f"  Configurations : {len(PARAM_GRID)}"
)

print(
    f"  Spatial folds  : {N_SPLITS}"
)

print(
    f"  Total model fits: "
    f"{len(PARAM_GRID) * N_SPLITS}"
)

print("\n  Parameters varied:")
print("    n_estimators")
print("    max_features")
print("    min_samples_leaf")
print("    max_depth")


# ============================================================
# 12. RUN TUNING
# ============================================================

print("\n[10] Running spatial hyperparameter tuning...")
print("-" * 70)

all_results = []

total_fits = (
    len(PARAM_GRID)
    * N_SPLITS
)

completed_fits = 0


for config_number, params in enumerate(
    PARAM_GRID,
    start=1
):

    print(
        f"\n  Configuration "
        f"{config_number}/{len(PARAM_GRID)}"
    )

    print(
        f"    n_estimators   = "
        f"{params['n_estimators']}"
    )

    print(
        f"    max_features   = "
        f"{params['max_features']}"
    )

    print(
        f"    min_samples_leaf= "
        f"{params['min_samples_leaf']}"
    )

    print(
        f"    max_depth      = "
        f"{params['max_depth']}"
    )

    for fold_number, (train_idx, val_idx) in enumerate(
        fold_splits,
        start=1
    ):

        model = build_rf_pipeline(
            params
        )

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model.fit(
            X_train_fold,
            y_train_fold
        )

        probabilities = model.predict_proba(
            X_val_fold
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        roc_auc = roc_auc_score(
            y_val_fold,
            probabilities
        )

        pr_auc = average_precision_score(
            y_val_fold,
            probabilities
        )

        accuracy = accuracy_score(
            y_val_fold,
            predictions
        )

        balanced_accuracy = balanced_accuracy_score(
            y_val_fold,
            predictions
        )

        precision = precision_score(
            y_val_fold,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_val_fold,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val_fold,
            predictions,
            zero_division=0
        )

        all_results.append({

            "config_number": config_number,

            "n_estimators":
                params["n_estimators"],

            "max_features":
                str(params["max_features"]),

            "min_samples_leaf":
                params["min_samples_leaf"],

            "max_depth":
                (
                    "None"
                    if params["max_depth"] is None
                    else params["max_depth"]
                ),

            "fold": fold_number,

            "n_train": len(train_idx),

            "n_validation": len(val_idx),

            "roc_auc": roc_auc,

            "pr_auc": pr_auc,

            "accuracy": accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "precision": precision,

            "recall_sensitivity":
                recall,

            "f1": f1
        })

        completed_fits += 1

        print(
            f"    Fold {fold_number}: "
            f"ROC-AUC={roc_auc:.4f}, "
            f"PR-AUC={pr_auc:.4f}"
        )

    # --------------------------------------------------------
    # Configuration-level quick summary
    # --------------------------------------------------------

    config_results = pd.DataFrame(
        [
            row
            for row in all_results
            if row["config_number"] == config_number
        ]
    )

    print(
        f"    Mean ROC-AUC: "
        f"{config_results['roc_auc'].mean():.4f}"
    )

    print(
        f"    Mean PR-AUC : "
        f"{config_results['pr_auc'].mean():.4f}"
    )

    print(
        f"    Progress: "
        f"{completed_fits}/{total_fits} fits"
    )


# ============================================================
# 13. CREATE RESULTS DATAFRAME
# ============================================================

print("\n[11] Creating tuning results...")
print("-" * 70)

results_df = pd.DataFrame(
    all_results
)

print(
    f"  Rows generated : {len(results_df)}"
)

print(
    f"  Expected rows  : "
    f"{len(PARAM_GRID) * N_SPLITS}"
)


# ============================================================
# 14. SUMMARIZE CONFIGURATIONS
# ============================================================

print("\n[12] Summarizing hyperparameter configurations...")
print("-" * 70)

summary_rows = []

for config_number in range(
    1,
    len(PARAM_GRID) + 1
):

    config_results = results_df[
        results_df["config_number"]
        == config_number
    ]

    params = PARAM_GRID[
        config_number - 1
    ]

    row = {

        "config_number":
            config_number,

        "n_estimators":
            params["n_estimators"],

        "max_features":
            str(params["max_features"]),

        "min_samples_leaf":
            params["min_samples_leaf"],

        "max_depth":
            (
                "None"
                if params["max_depth"] is None
                else params["max_depth"]
            )
    }

    metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall_sensitivity",
        "f1"
    ]

    for metric in metrics:

        row[f"{metric}_mean"] = (
            config_results[metric].mean()
        )

        row[f"{metric}_std"] = (
            config_results[metric].std(
                ddof=1
            )
        )

        row[f"{metric}_min"] = (
            config_results[metric].min()
        )

        row[f"{metric}_max"] = (
            config_results[metric].max()
        )

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 15. RANK CONFIGURATIONS
# ============================================================

print("\n[13] Ranking configurations...")
print("-" * 70)

# Primary criterion:
#   highest mean spatial ROC-AUC
#
# Secondary:
#   highest mean PR-AUC
#
# Tertiary:
#   lower ROC-AUC standard deviation
#
# This avoids selecting a model purely because it performs
# well on one fold.

summary_df = summary_df.sort_values(
    by=[
        "roc_auc_mean",
        "pr_auc_mean",
        "roc_auc_std"
    ],
    ascending=[
        False,
        False,
        True
    ]
).reset_index(drop=True)

summary_df["rank"] = (
    np.arange(len(summary_df))
    + 1
)


# ============================================================
# 16. DISPLAY TOP CONFIGURATIONS
# ============================================================

print("\n[14] Top Random Forest configurations...")
print("-" * 70)

display_columns = [
    "rank",
    "config_number",
    "n_estimators",
    "max_features",
    "min_samples_leaf",
    "max_depth",
    "roc_auc_mean",
    "roc_auc_std",
    "pr_auc_mean",
    "pr_auc_std",
    "balanced_accuracy_mean",
    "f1_mean"
]

top_n = min(
    10,
    len(summary_df)
)

print(
    summary_df[
        display_columns
    ].head(top_n).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 17. SELECT BEST CONFIGURATION
# ============================================================

print("\n[15] Selecting best configuration...")
print("-" * 70)

best = summary_df.iloc[0]

best_config_number = int(
    best["config_number"]
)

best_params = PARAM_GRID[
    best_config_number - 1
]

print(
    f"  Selected configuration : "
    f"{best_config_number}"
)

print(
    f"  n_estimators           : "
    f"{best_params['n_estimators']}"
)

print(
    f"  max_features           : "
    f"{best_params['max_features']}"
)

print(
    f"  min_samples_leaf       : "
    f"{best_params['min_samples_leaf']}"
)

print(
    f"  max_depth              : "
    f"{best_params['max_depth']}"
)

print(
    f"\n  Mean spatial ROC-AUC    : "
    f"{best['roc_auc_mean']:.4f}"
)

print(
    f"  ROC-AUC std             : "
    f"{best['roc_auc_std']:.4f}"
)

print(
    f"  Mean spatial PR-AUC     : "
    f"{best['pr_auc_mean']:.4f}"
)

print(
    f"  PR-AUC std              : "
    f"{best['pr_auc_std']:.4f}"
)


# ============================================================
# 18. SAVE RESULTS
# ============================================================

print("\n[16] Saving tuning results...")
print("-" * 70)

# Full fold-level results
results_df.to_csv(
    RESULTS_FILE,
    index=False
)

# Configuration summary
summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

# Selected configuration
selected_df = pd.DataFrame([
    {
        "config_number":
            best_config_number,

        "n_estimators":
            best_params["n_estimators"],

        "max_features":
            str(best_params["max_features"]),

        "min_samples_leaf":
            best_params["min_samples_leaf"],

        "max_depth":
            (
                "None"
                if best_params["max_depth"] is None
                else best_params["max_depth"]
            ),

        "cv_roc_auc_mean":
            best["roc_auc_mean"],

        "cv_roc_auc_std":
            best["roc_auc_std"],

        "cv_pr_auc_mean":
            best["pr_auc_mean"],

        "cv_pr_auc_std":
            best["pr_auc_std"],

        "cv_balanced_accuracy_mean":
            best["balanced_accuracy_mean"],

        "cv_f1_mean":
            best["f1_mean"]
    }
])

selected_df.to_csv(
    SELECTED_FILE,
    index=False
)

print(
    f"  Saved: {RESULTS_FILE}"
)

print(
    f"  Saved: {SUMMARY_FILE}"
)

print(
    f"  Saved: {SELECTED_FILE}"
)


# ============================================================
# 19. FINAL VALIDATION
# ============================================================

print("\n[17] Final validation...")
print("-" * 70)

expected_rows = (
    len(PARAM_GRID)
    * N_SPLITS
)

if len(results_df) != expected_rows:
    raise RuntimeError(
        "Unexpected number of tuning results."
    )

# Every configuration should have five folds
for config_number in range(
    1,
    len(PARAM_GRID) + 1
):

    config_folds = results_df[
        results_df["config_number"]
        == config_number
    ]["fold"].tolist()

    if sorted(config_folds) != list(
        range(1, N_SPLITS + 1)
    ):
        raise RuntimeError(
            f"Missing folds for configuration "
            f"{config_number}."
        )


# Check metrics
metric_columns = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "f1"
]

for metric in metric_columns:

    if results_df[metric].isna().any():
        raise RuntimeError(
            f"Missing values in {metric}."
        )

    if (
        (results_df[metric] < 0).any()
        or
        (results_df[metric] > 1).any()
    ):
        raise RuntimeError(
            f"Invalid values in {metric}."
        )


# Confirm final test size
if len(test) != 203:

    print(
        "  WARNING: Final test set size is not 203."
    )

else:

    print(
        "  Final test set remains 203 observations."
    )


print(
    "  Configuration count : PASSED"
)

print(
    "  Fold completeness   : PASSED"
)

print(
    "  Metric validity     : PASSED"
)

print(
    "  Final test protected: PASSED"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 60 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSpatial tuning design:")
print(
    f"  Training observations : {len(train)}"
)

print(
    f"  Spatial groups        : "
    f"{groups.nunique()}"
)

print(
    f"  Spatial CV folds      : "
    f"{N_SPLITS}"
)

print(
    f"  RF configurations     : "
    f"{len(PARAM_GRID)}"
)

print(
    f"  Total model fits      : "
    f"{len(results_df)}"
)

print("\nSelected configuration:")

print(
    f"  n_estimators    : "
    f"{best_params['n_estimators']}"
)

print(
    f"  max_features    : "
    f"{best_params['max_features']}"
)

print(
    f"  min_samples_leaf: "
    f"{best_params['min_samples_leaf']}"
)

print(
    f"  max_depth       : "
    f"{best_params['max_depth']}"
)

print("\nSelected configuration CV performance:")

print(
    f"  ROC-AUC : "
    f"{best['roc_auc_mean']:.4f} "
    f"+/- {best['roc_auc_std']:.4f}"
)

print(
    f"  PR-AUC  : "
    f"{best['pr_auc_mean']:.4f} "
    f"+/- {best['pr_auc_std']:.4f}"
)

print(
    f"  Balanced Accuracy : "
    f"{best['balanced_accuracy_mean']:.4f}"
)

print(
    f"  F1                : "
    f"{best['f1_mean']:.4f}"
)

print("\nImportant:")
print(
    "  Hyperparameter tuning used only the 788 training observations."
)

print(
    "  The 203-observation final test set was not used."
)

print(
    "  Spatial groups were kept separate in every fold."
)

print(
    "  The selected configuration is NOT yet evaluated "
    "on the final test set."
)

print("\nOutputs:")
print(
    f"  {RESULTS_FILE}"
)

print(
    f"  {SUMMARY_FILE}"
)

print(
    f"  {SELECTED_FILE}"
)

print("\nNext stage:")
print(
    "  Refit the selected Random Forest on all 788 training "
    "observations and perform ONE final evaluation on the "
    "untouched 203-observation spatial test set."
)

print("=" * 70)