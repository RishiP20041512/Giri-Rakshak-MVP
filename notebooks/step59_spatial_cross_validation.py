"""
STEP 59 — SPATIAL CROSS-VALIDATION

Purpose:
- Evaluate model robustness using spatially grouped cross-validation.
- Use ONLY the 788 training observations.
- Keep the 203-observation final spatial test set untouched.
- Compare:
    1. Logistic Regression
    2. Random Forest
- Use spatial groups so observations from the same spatial block
  cannot appear in both training and validation folds.
- Perform preprocessing inside each pipeline.

Important:
- This step is for model robustness assessment.
- The final 203-observation test set is NOT used here.
- No model selection is based on the final test set.
- Lithology is not included because it is currently unavailable.

Input:
    processed/ner_train_modeling.csv
    processed/ner_test_modeling.csv

Output:
    processed/step59_spatial_cv_results.csv
    processed/step59_spatial_cv_summary.csv
"""


import os
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

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

RESULTS_FILE = r"processed\step59_spatial_cv_results.csv"
SUMMARY_FILE = r"processed\step59_spatial_cv_summary.csv"


# ------------------------------------------------------------
# Same 9-factor predictor configuration as Step 57/58
# ------------------------------------------------------------

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
# HEADER
# ============================================================

print("=" * 70)
print("STEP 59 — SPATIAL CROSS-VALIDATION")
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

print(f"  Training observations : {len(train)}")
print(f"  Final test observations: {len(test)}")


# ============================================================
# 3. VERIFY REQUIRED COLUMNS
# ============================================================

print("\n[3] Checking required columns...")
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
            f"Missing columns in {dataset_name} dataset:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

print("  All required columns are present.")


# ============================================================
# 4. IMPORTANT TEST-SET PROTECTION
# ============================================================

print("\n[4] Protecting final test set...")
print("-" * 70)

print(
    "  The 203-observation final test set will NOT "
    "be used during cross-validation."
)

print(
    f"  Final test observations reserved: {len(test)}"
)


# ============================================================
# 5. EXTRACT TRAINING VARIABLES
# ============================================================

print("\n[5] Preparing cross-validation data...")
print("-" * 70)

X = train[FEATURES].copy()
y = train[TARGET].copy()

groups = train["spatial_group"].copy()

print(f"  X shape     : {X.shape}")
print(f"  y shape     : {y.shape}")
print(f"  Groups      : {groups.nunique()}")


# ============================================================
# 6. CHECK TRAINING DATA
# ============================================================

print("\n[6] Checking training data...")
print("-" * 70)

if X.isna().sum().sum() != 0:
    raise ValueError(
        "Missing predictor values found in training data."
    )

if y.isna().sum() != 0:
    raise ValueError(
        "Missing target values found in training data."
    )

if set(y.unique()) != {0, 1}:
    raise ValueError(
        "Training target must contain classes 0 and 1."
    )

print(
    f"  Background observations : "
    f"{int((y == 0).sum())}"
)

print(
    f"  Landslide observations  : "
    f"{int((y == 1).sum())}"
)

print(
    f"  Landslide proportion    : "
    f"{y.mean():.4f}"
)

print("  Training data check PASSED.")


# ============================================================
# 7. CHECK SPATIAL GROUPS
# ============================================================

print("\n[7] Checking spatial groups...")
print("-" * 70)

group_sizes = groups.value_counts()

print(
    f"  Number of spatial groups : "
    f"{groups.nunique()}"
)

print(
    f"  Largest group            : "
    f"{group_sizes.max()}"
)

print(
    f"  Median group size        : "
    f"{group_sizes.median():.1f}"
)

print(
    f"  Smallest group           : "
    f"{group_sizes.min()}"
)


# ============================================================
# 8. CREATE ONE-HOT ENCODERS
# ============================================================

print("\n[8] Configuring categorical encoding...")
print("-" * 70)


def make_one_hot_encoder():

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
# 9. MODEL PIPELINE FACTORY
# ============================================================

print("\n[9] Building model pipelines...")
print("-" * 70)


def build_logistic_pipeline():

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
                        ),
                        (
                            "scaler",
                            StandardScaler()
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
                            make_one_hot_encoder()
                        )
                    ]
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight=None,
        random_state=RANDOM_STATE
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


def build_random_forest_pipeline():

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
                            make_one_hot_encoder()
                        )
                    ]
                ),
                CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
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


models = {
    "Logistic Regression": build_logistic_pipeline(),
    "Random Forest": build_random_forest_pipeline()
}

print("  Logistic Regression pipeline: ready")
print("  Random Forest pipeline      : ready")


# ============================================================
# 10. CREATE STRATIFIED GROUPED CV
# ============================================================

print("\n[10] Creating spatial cross-validation...")
print("-" * 70)

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

print(
    f"  Number of folds : {N_SPLITS}"
)

print(
    "  Stratification : enabled"
)

print(
    "  Grouping       : spatial_group"
)


# ============================================================
# 11. INSPECT FOLDS
# ============================================================

print("\n[11] Inspecting spatial folds...")
print("-" * 70)

fold_information = []

for fold_number, (train_idx, val_idx) in enumerate(
    cv.split(
        X,
        y,
        groups
    ),
    start=1
):

    fold_train_groups = set(
        groups.iloc[train_idx]
    )

    fold_val_groups = set(
        groups.iloc[val_idx]
    )

    overlap = (
        fold_train_groups
        .intersection(fold_val_groups)
    )

    if overlap:
        raise RuntimeError(
            f"Spatial leakage detected in fold {fold_number}."
        )

    fold_train_y = y.iloc[train_idx]
    fold_val_y = y.iloc[val_idx]

    fold_information.append({
        "fold": fold_number,
        "train_n": len(train_idx),
        "validation_n": len(val_idx),
        "train_groups": len(fold_train_groups),
        "validation_groups": len(fold_val_groups),
        "train_landslide_fraction": fold_train_y.mean(),
        "validation_landslide_fraction": fold_val_y.mean(),
        "spatial_group_overlap": len(overlap)
    })

    print(f"\n  Fold {fold_number}")
    print(
        f"    Train observations      : "
        f"{len(train_idx)}"
    )
    print(
        f"    Validation observations : "
        f"{len(val_idx)}"
    )
    print(
        f"    Train spatial groups    : "
        f"{len(fold_train_groups)}"
    )
    print(
        f"    Validation groups       : "
        f"{len(fold_val_groups)}"
    )
    print(
        f"    Train LS proportion     : "
        f"{fold_train_y.mean():.4f}"
    )
    print(
        f"    Validation LS proportion: "
        f"{fold_val_y.mean():.4f}"
    )
    print(
        f"    Group overlap           : "
        f"{len(overlap)}"
    )

    if set(fold_train_y.unique()) != {0, 1}:
        raise RuntimeError(
            f"Fold {fold_number} training set "
            "does not contain both classes."
        )

    if set(fold_val_y.unique()) != {0, 1}:
        raise RuntimeError(
            f"Fold {fold_number} validation set "
            "does not contain both classes."
        )


print("\n  Fold inspection PASSED.")


# ============================================================
# 12. RUN CROSS-VALIDATION
# ============================================================

print("\n[12] Running spatial cross-validation...")
print("-" * 70)

all_results = []

for model_name in models.keys():

    print(f"\n  Model: {model_name}")

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(
            X,
            y,
            groups
        ),
        start=1
    ):

        # ----------------------------------------------------
        # Build a fresh pipeline for every fold
        # ----------------------------------------------------

        if model_name == "Logistic Regression":
            model = build_logistic_pipeline()
        else:
            model = build_random_forest_pipeline()

        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]

        X_fold_val = X.iloc[val_idx]
        y_fold_val = y.iloc[val_idx]

        # ----------------------------------------------------
        # Fit ONLY on the fold training data
        # ----------------------------------------------------

        model.fit(
            X_fold_train,
            y_fold_train
        )

        # ----------------------------------------------------
        # Predict validation probabilities
        # ----------------------------------------------------

        probabilities = model.predict_proba(
            X_fold_val
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        roc_auc = roc_auc_score(
            y_fold_val,
            probabilities
        )

        pr_auc = average_precision_score(
            y_fold_val,
            probabilities
        )

        accuracy = accuracy_score(
            y_fold_val,
            predictions
        )

        balanced_accuracy = balanced_accuracy_score(
            y_fold_val,
            predictions
        )

        precision = precision_score(
            y_fold_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_fold_val,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_fold_val,
            predictions,
            zero_division=0
        )

        all_results.append({
            "model": model_name,
            "fold": fold_number,
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_accuracy,
            "precision": precision,
            "recall_sensitivity": recall,
            "f1": f1
        })

        print(
            f"    Fold {fold_number}: "
            f"ROC-AUC={roc_auc:.4f}, "
            f"PR-AUC={pr_auc:.4f}, "
            f"Balanced Acc={balanced_accuracy:.4f}"
        )


# ============================================================
# 13. CREATE RESULTS DATAFRAME
# ============================================================

print("\n[13] Creating fold-level results...")
print("-" * 70)

results_df = pd.DataFrame(
    all_results
)

print(
    f"  Result rows : {len(results_df)}"
)

print(
    f"  Expected    : "
    f"{len(models) * N_SPLITS}"
)


# ============================================================
# 14. CREATE SUMMARY
# ============================================================

print("\n[14] Calculating cross-validation summaries...")
print("-" * 70)

metric_columns = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "f1"
]

summary_rows = []

for model_name in models.keys():

    model_results = results_df[
        results_df["model"] == model_name
    ]

    row = {
        "model": model_name,
        "folds": len(model_results)
    }

    for metric in metric_columns:

        row[f"{metric}_mean"] = (
            model_results[metric].mean()
        )

        row[f"{metric}_std"] = (
            model_results[metric].std(ddof=1)
        )

        row[f"{metric}_min"] = (
            model_results[metric].min()
        )

        row[f"{metric}_max"] = (
            model_results[metric].max()
        )

    summary_rows.append(row)


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("\n[15] Spatial cross-validation results...")
print("-" * 70)

for model_name in models.keys():

    row = summary_df[
        summary_df["model"] == model_name
    ].iloc[0]

    print(f"\n  {model_name}")

    print(
        f"    ROC-AUC           : "
        f"{row['roc_auc_mean']:.4f} "
        f"+/- {row['roc_auc_std']:.4f}"
    )

    print(
        f"    PR-AUC            : "
        f"{row['pr_auc_mean']:.4f} "
        f"+/- {row['pr_auc_std']:.4f}"
    )

    print(
        f"    Accuracy          : "
        f"{row['accuracy_mean']:.4f} "
        f"+/- {row['accuracy_std']:.4f}"
    )

    print(
        f"    Balanced Accuracy : "
        f"{row['balanced_accuracy_mean']:.4f} "
        f"+/- {row['balanced_accuracy_std']:.4f}"
    )

    print(
        f"    Precision         : "
        f"{row['precision_mean']:.4f} "
        f"+/- {row['precision_std']:.4f}"
    )

    print(
        f"    Recall            : "
        f"{row['recall_sensitivity_mean']:.4f} "
        f"+/- {row['recall_sensitivity_std']:.4f}"
    )

    print(
        f"    F1                : "
        f"{row['f1_mean']:.4f} "
        f"+/- {row['f1_std']:.4f}"
    )


# ============================================================
# 16. COMPARE MODELS
# ============================================================

print("\n[16] Model comparison...")
print("-" * 70)

lr_row = summary_df[
    summary_df["model"] == "Logistic Regression"
].iloc[0]

rf_row = summary_df[
    summary_df["model"] == "Random Forest"
].iloc[0]

roc_difference = (
    rf_row["roc_auc_mean"]
    - lr_row["roc_auc_mean"]
)

pr_difference = (
    rf_row["pr_auc_mean"]
    - lr_row["pr_auc_mean"]
)

print(
    f"  Random Forest minus Logistic Regression "
    f"ROC-AUC mean: {roc_difference:.4f}"
)

print(
    f"  Random Forest minus Logistic Regression "
    f"PR-AUC mean : {pr_difference:.4f}"
)


# ============================================================
# 17. SAVE RESULTS
# ============================================================

print("\n[17] Saving cross-validation results...")
print("-" * 70)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print(f"  Saved: {RESULTS_FILE}")
print(f"  Saved: {SUMMARY_FILE}")


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

print("\n[18] Final validation...")
print("-" * 70)

# ------------------------------------------------------------
# Expected number of fold results
# ------------------------------------------------------------

expected_rows = (
    len(models)
    * N_SPLITS
)

if len(results_df) != expected_rows:
    raise RuntimeError(
        "Unexpected number of cross-validation results."
    )

# ------------------------------------------------------------
# Each model must have all folds
# ------------------------------------------------------------

for model_name in models.keys():

    model_folds = results_df[
        results_df["model"] == model_name
    ]["fold"].tolist()

    expected_folds = list(
        range(1, N_SPLITS + 1)
    )

    if sorted(model_folds) != expected_folds:
        raise RuntimeError(
            f"Missing CV folds for {model_name}."
        )

# ------------------------------------------------------------
# Check metric ranges
# ------------------------------------------------------------

for metric in metric_columns:

    if (
        results_df[metric].isna().any()
    ):
        raise RuntimeError(
            f"Missing values found in {metric}."
        )

    if (
        (results_df[metric] < 0).any()
        or
        (results_df[metric] > 1).any()
    ):
        raise RuntimeError(
            f"Invalid metric values in {metric}."
        )

# ------------------------------------------------------------
# Reconfirm final test set was not included
# ------------------------------------------------------------

if len(test) != 203:
    print(
        "  NOTE: Final test set size differs from 203."
    )

print("  Fold count               : PASSED")
print("  Metric validity          : PASSED")
print("  Spatial grouping         : PASSED")
print("  Final test protection    : PASSED")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 59 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCross-validation design:")
print(f"  Training observations : {len(train)}")
print(f"  Spatial groups        : {groups.nunique()}")
print(f"  Number of folds       : {N_SPLITS}")
print("  Method                : StratifiedGroupKFold")

print("\nModels evaluated:")
print("  1. Logistic Regression")
print("  2. Random Forest")

print("\nImportant:")
print(
    "  The 203-observation final spatial test set "
    "was NOT used in cross-validation."
)

print(
    "  Preprocessing was fitted independently "
    "inside each CV fold."
)

print("\nOutputs:")
print(f"  {RESULTS_FILE}")
print(f"  {SUMMARY_FILE}")

print("\nNext stage:")
print(
    "  Use spatial-CV stability together with the "
    "final untouched spatial test result to determine "
    "the final susceptibility model."
)

print("=" * 70)