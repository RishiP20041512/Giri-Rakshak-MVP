"""
STEP 61 — FINAL RANDOM FOREST EVALUATION

Purpose:
- Load the 788 training observations.
- Load the 203 untouched spatial test observations.
- Use the Random Forest configuration selected in Step 60.
- Fit the selected model on ALL 788 training observations.
- Evaluate ONCE on the untouched 203-observation spatial test set.
- Compare final tuned RF performance with the Step 58 baseline RF.
- Calculate complete classification metrics.
- Save final predictions and evaluation reports.

Selected configuration from Step 60:
    n_estimators     = 700
    max_features     = sqrt
    min_samples_leaf = 5
    max_depth        = None

Important:
- The 203-observation test set has NOT been used for tuning.
- No further model tuning is performed after this step.
- This is the final independent spatial test evaluation.

Input:
    processed/ner_train_modeling.csv
    processed/ner_test_modeling.csv
    processed/step60_selected_rf_config.csv
    processed/step58_model_results.csv

Outputs:
    processed/step61_final_rf_metrics.csv
    processed/step61_final_rf_confusion_matrix.csv
    processed/step61_final_rf_predictions.csv
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

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = r"processed\ner_train_modeling.csv"
TEST_FILE = r"processed\ner_test_modeling.csv"

SELECTED_CONFIG_FILE = (
    r"processed\step60_selected_rf_config.csv"
)

BASELINE_RESULTS_FILE = (
    r"processed\step58_model_results.csv"
)

METRICS_FILE = (
    r"processed\step61_final_rf_metrics.csv"
)

CONFUSION_FILE = (
    r"processed\step61_final_rf_confusion_matrix.csv"
)

PREDICTIONS_FILE = (
    r"processed\step61_final_rf_predictions.csv"
)


# ============================================================
# FEATURES
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

RANDOM_STATE = 42


# ============================================================
# EXPECTED SELECTED CONFIGURATION
# ============================================================

EXPECTED_PARAMS = {
    "n_estimators": 700,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "max_depth": None
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 61 — FINAL RANDOM FOREST EVALUATION")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILES
# ============================================================

print("\n[1] Checking input files...")
print("-" * 70)

input_files = [
    TRAIN_FILE,
    TEST_FILE,
    SELECTED_CONFIG_FILE,
    BASELINE_RESULTS_FILE
]

for file_path in input_files:

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(f"  Found: {file_path}")


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[2] Loading train/test datasets...")
print("-" * 70)

train = pd.read_csv(
    TRAIN_FILE
)

test = pd.read_csv(
    TEST_FILE
)

print(
    f"  Training observations : "
    f"{len(train)}"
)

print(
    f"  Final test observations: "
    f"{len(test)}"
)


# ============================================================
# 3. PROTECT FINAL TEST SET
# ============================================================

print("\n[3] Final test-set protection...")
print("-" * 70)

print(
    "  This test set has remained untouched "
    "during spatial CV and hyperparameter tuning."
)

print(
    f"  Final independent test observations: "
    f"{len(test)}"
)

if len(test) != 203:

    print(
        "  WARNING: Expected 203 test observations."
    )

else:

    print(
        "  Expected test size confirmed."
    )


# ============================================================
# 4. LOAD SELECTED CONFIGURATION
# ============================================================

print("\n[4] Loading selected Random Forest configuration...")
print("-" * 70)

selected_config = pd.read_csv(
    SELECTED_CONFIG_FILE
)

if len(selected_config) != 1:
    raise ValueError(
        "Selected configuration file should contain exactly one row."
    )

config_row = selected_config.iloc[0]

selected_n_estimators = int(
    config_row["n_estimators"]
)

selected_max_features = str(
    config_row["max_features"]
)

selected_min_samples_leaf = int(
    config_row["min_samples_leaf"]
)

selected_max_depth_raw = (
    config_row["max_depth"]
)

if (
    pd.isna(selected_max_depth_raw)
    or
    str(selected_max_depth_raw).lower()
    == "none"
):

    selected_max_depth = None

else:

    selected_max_depth = int(
        selected_max_depth_raw
    )


print(
    f"  n_estimators     : "
    f"{selected_n_estimators}"
)

print(
    f"  max_features     : "
    f"{selected_max_features}"
)

print(
    f"  min_samples_leaf : "
    f"{selected_min_samples_leaf}"
)

print(
    f"  max_depth        : "
    f"{selected_max_depth}"
)


# ============================================================
# 5. VERIFY SELECTED CONFIGURATION
# ============================================================

print("\n[5] Verifying selected configuration...")
print("-" * 70)

if selected_n_estimators != EXPECTED_PARAMS["n_estimators"]:
    raise RuntimeError(
        "Unexpected n_estimators."
    )

if selected_max_features != EXPECTED_PARAMS["max_features"]:
    raise RuntimeError(
        "Unexpected max_features."
    )

if selected_min_samples_leaf != EXPECTED_PARAMS["min_samples_leaf"]:
    raise RuntimeError(
        "Unexpected min_samples_leaf."
    )

if selected_max_depth != EXPECTED_PARAMS["max_depth"]:
    raise RuntimeError(
        "Unexpected max_depth."
    )

print(
    "  Configuration matches Step 60 selection."
)


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[6] Checking modeling columns...")
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

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing columns in {dataset_name}:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing_columns
            )
        )

print(
    "  All required columns are present."
)


# ============================================================
# 7. VERIFY NO MISSING VALUES
# ============================================================

print("\n[7] Checking missing values...")
print("-" * 70)

train_missing = (
    train[FEATURES]
    .isna()
    .sum()
    .sum()
)

test_missing = (
    test[FEATURES]
    .isna()
    .sum()
    .sum()
)

print(
    f"  Training missing predictor values: "
    f"{int(train_missing)}"
)

print(
    f"  Testing missing predictor values : "
    f"{int(test_missing)}"
)

if train_missing != 0:
    raise RuntimeError(
        "Training predictors contain missing values."
    )

if test_missing != 0:
    raise RuntimeError(
        "Testing predictors contain missing values."
    )

print(
    "  Missing-value check PASSED."
)


# ============================================================
# 8. VERIFY SPATIAL SEPARATION
# ============================================================

print("\n[8] Verifying spatial separation...")
print("-" * 70)

train_groups = set(
    train["spatial_group"]
)

test_groups = set(
    test["spatial_group"]
)

overlap = train_groups.intersection(
    test_groups
)

print(
    f"  Training spatial groups : "
    f"{len(train_groups)}"
)

print(
    f"  Testing spatial groups  : "
    f"{len(test_groups)}"
)

print(
    f"  Overlapping groups      : "
    f"{len(overlap)}"
)

if overlap:

    raise RuntimeError(
        "Spatial leakage detected."
    )

print(
    "  Spatial separation PASSED."
)


# ============================================================
# 9. PREPARE X AND y
# ============================================================

print("\n[9] Preparing final model data...")
print("-" * 70)

X_train = train[
    FEATURES
].copy()

y_train = train[
    TARGET
].copy()

X_test = test[
    FEATURES
].copy()

y_test = test[
    TARGET
].copy()

print(
    f"  X_train : {X_train.shape}"
)

print(
    f"  X_test  : {X_test.shape}"
)

print(
    f"  y_train : {y_train.shape}"
)

print(
    f"  y_test  : {y_test.shape}"
)


# ============================================================
# 10. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n[10] Final class distribution...")
print("-" * 70)

print("\n  TRAIN")

print(
    y_train
    .value_counts()
    .sort_index()
    .to_string()
)

print("\n  TEST")

print(
    y_test
    .value_counts()
    .sort_index()
    .to_string()
)

if set(y_train.unique()) != {0, 1}:
    raise RuntimeError(
        "Training set does not contain both classes."
    )

if set(y_test.unique()) != {0, 1}:
    raise RuntimeError(
        "Test set does not contain both classes."
    )


# ============================================================
# 11. CREATE ENCODER
# ============================================================

print("\n[11] Creating categorical encoder...")
print("-" * 70)


try:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    )

except TypeError:

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse=False
    )


print(
    "  handle_unknown = ignore"
)

print(
    "  drop = first"
)


# ============================================================
# 12. BUILD PREPROCESSOR
# ============================================================

print("\n[12] Building preprocessing pipeline...")
print("-" * 70)

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

                        encoder
                    )
                ]
            ),

            CATEGORICAL_FEATURES
        )
    ],

    remainder="drop"
)


# ============================================================
# 13. BUILD FINAL RANDOM FOREST
# ============================================================

print("\n[13] Building final Random Forest...")
print("-" * 70)

rf_model = RandomForestClassifier(

    n_estimators=selected_n_estimators,

    max_features=selected_max_features,

    min_samples_leaf=selected_min_samples_leaf,

    max_depth=selected_max_depth,

    class_weight=None,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


final_pipeline = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            rf_model
        )
    ]
)


print(
    "  Final Random Forest configured."
)


# ============================================================
# 14. FIT ON ALL TRAINING DATA
# ============================================================

print("\n[14] Fitting final model...")
print("-" * 70)

print(
    "  Training on all 788 training observations."
)

print(
    "  The 203 test observations are NOT used."
)

final_pipeline.fit(
    X_train,
    y_train
)

print(
    "  Final model fitting completed."
)


# ============================================================
# 15. PREDICT TRAINING DATA
# ============================================================

print("\n[15] Generating training predictions...")
print("-" * 70)

train_probability = (
    final_pipeline
    .predict_proba(X_train)[:, 1]
)

train_prediction = (
    train_probability >= 0.5
).astype(int)

print(
    "  Training predictions generated."
)


# ============================================================
# 16. PREDICT FINAL TEST DATA
# ============================================================

print("\n[16] Generating FINAL test predictions...")
print("-" * 70)

test_probability = (
    final_pipeline
    .predict_proba(X_test)[:, 1]
)

test_prediction = (
    test_probability >= 0.5
).astype(int)

print(
    f"  Final test predictions generated: "
    f"{len(test_prediction)}"
)


# ============================================================
# 17. METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    probability,
    prediction
):

    roc_auc = roc_auc_score(
        y_true,
        probability
    )

    pr_auc = average_precision_score(
        y_true,
        probability
    )

    accuracy = accuracy_score(
        y_true,
        prediction
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            prediction
        )
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {

        "roc_auc":
            roc_auc,

        "pr_auc":
            pr_auc,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "precision":
            precision,

        "recall_sensitivity":
            recall,

        "specificity":
            specificity,

        "f1":
            f1,

        "true_negative":
            int(tn),

        "false_positive":
            int(fp),

        "false_negative":
            int(fn),

        "true_positive":
            int(tp)
    }


# ============================================================
# 18. CALCULATE FINAL METRICS
# ============================================================

print("\n[17] Calculating final performance metrics...")
print("-" * 70)

train_metrics = calculate_metrics(
    y_train,
    train_probability,
    train_prediction
)

test_metrics = calculate_metrics(
    y_test,
    test_probability,
    test_prediction
)


# ============================================================
# 19. PRINT TRAIN PERFORMANCE
# ============================================================

print("\n[18] TRAINING PERFORMANCE")
print("-" * 70)

print(
    f"  ROC-AUC            : "
    f"{train_metrics['roc_auc']:.4f}"
)

print(
    f"  PR-AUC             : "
    f"{train_metrics['pr_auc']:.4f}"
)

print(
    f"  Accuracy           : "
    f"{train_metrics['accuracy']:.4f}"
)

print(
    f"  Balanced Accuracy  : "
    f"{train_metrics['balanced_accuracy']:.4f}"
)

print(
    f"  Precision          : "
    f"{train_metrics['precision']:.4f}"
)

print(
    f"  Recall/Sensitivity : "
    f"{train_metrics['recall_sensitivity']:.4f}"
)

print(
    f"  Specificity        : "
    f"{train_metrics['specificity']:.4f}"
)

print(
    f"  F1                 : "
    f"{train_metrics['f1']:.4f}"
)


# ============================================================
# 20. PRINT FINAL TEST PERFORMANCE
# ============================================================

print("\n[19] FINAL SPATIAL TEST PERFORMANCE")
print("-" * 70)

print(
    f"  ROC-AUC            : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"  PR-AUC             : "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"  Accuracy           : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"  Balanced Accuracy  : "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"  Precision          : "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"  Recall/Sensitivity : "
    f"{test_metrics['recall_sensitivity']:.4f}"
)

print(
    f"  Specificity        : "
    f"{test_metrics['specificity']:.4f}"
)

print(
    f"  F1                 : "
    f"{test_metrics['f1']:.4f}"
)


# ============================================================
# 21. CONFUSION MATRIX
# ============================================================

print("\n[20] FINAL TEST CONFUSION MATRIX")
print("-" * 70)

print(
    "  Format: [[TN, FP], [FN, TP]]"
)

print(
    f"\n       {test_metrics['true_negative']:>4} "
    f"{test_metrics['false_positive']:>4}"
)

print(
    f"       {test_metrics['false_negative']:>4} "
    f"{test_metrics['true_positive']:>4}"
)


# ============================================================
# 22. TRAIN/TEST GENERALIZATION GAP
# ============================================================

print("\n[21] Generalization gap...")
print("-" * 70)

roc_gap = (
    train_metrics["roc_auc"]
    - test_metrics["roc_auc"]
)

pr_gap = (
    train_metrics["pr_auc"]
    - test_metrics["pr_auc"]
)

print(
    f"  Train ROC-AUC : "
    f"{train_metrics['roc_auc']:.4f}"
)

print(
    f"  Test ROC-AUC  : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"  ROC-AUC gap   : "
    f"{roc_gap:.4f}"
)

print(
    f"\n  Train PR-AUC  : "
    f"{train_metrics['pr_auc']:.4f}"
)

print(
    f"  Test PR-AUC   : "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"  PR-AUC gap    : "
    f"{pr_gap:.4f}"
)


# ============================================================
# 23. COMPARE WITH STEP 58 BASELINE
# ============================================================

print("\n[22] Comparing with Step 58 baseline Random Forest...")
print("-" * 70)

baseline_results = pd.read_csv(
    BASELINE_RESULTS_FILE
)

baseline_test = baseline_results[
    (baseline_results["model"] == "Random Forest")
    &
    (baseline_results["dataset"] == "test")
]

if len(baseline_test) == 1:

    baseline_row = baseline_test.iloc[0]

    print(
        f"  Baseline RF ROC-AUC : "
        f"{baseline_row['roc_auc']:.4f}"
    )

    print(
        f"  Tuned RF ROC-AUC    : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"  Difference          : "
        f"{test_metrics['roc_auc'] - baseline_row['roc_auc']:.4f}"
    )

    print(
        f"\n  Baseline RF PR-AUC  : "
        f"{baseline_row['pr_auc']:.4f}"
    )

    print(
        f"  Tuned RF PR-AUC     : "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"  Difference          : "
        f"{test_metrics['pr_auc'] - baseline_row['pr_auc']:.4f}"
    )

else:

    print(
        "  Baseline RF test result could not be found."
    )


# ============================================================
# 24. CREATE METRICS TABLE
# ============================================================

print("\n[23] Creating final metrics table...")
print("-" * 70)

metrics_rows = [

    {
        "model": "Final Tuned Random Forest",
        "dataset": "train",
        "n": len(y_train),
        "roc_auc": train_metrics["roc_auc"],
        "pr_auc": train_metrics["pr_auc"],
        "accuracy": train_metrics["accuracy"],
        "balanced_accuracy":
            train_metrics["balanced_accuracy"],
        "precision":
            train_metrics["precision"],
        "recall_sensitivity":
            train_metrics["recall_sensitivity"],
        "specificity":
            train_metrics["specificity"],
        "f1":
            train_metrics["f1"]
    },

    {
        "model": "Final Tuned Random Forest",
        "dataset": "test",
        "n": len(y_test),
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "accuracy": test_metrics["accuracy"],
        "balanced_accuracy":
            test_metrics["balanced_accuracy"],
        "precision":
            test_metrics["precision"],
        "recall_sensitivity":
            test_metrics["recall_sensitivity"],
        "specificity":
            test_metrics["specificity"],
        "f1":
            test_metrics["f1"]
    }
]

metrics_df = pd.DataFrame(
    metrics_rows
)


# ============================================================
# 25. CREATE CONFUSION MATRIX TABLE
# ============================================================

confusion_df = pd.DataFrame([

    {
        "model":
            "Final Tuned Random Forest",

        "dataset":
            "test",

        "true_negative":
            test_metrics["true_negative"],

        "false_positive":
            test_metrics["false_positive"],

        "false_negative":
            test_metrics["false_negative"],

        "true_positive":
            test_metrics["true_positive"]
    }

])


# ============================================================
# 26. CREATE PREDICTION TABLE
# ============================================================

print("\n[24] Creating final prediction table...")
print("-" * 70)

predictions_df = test[
    [
        "lat",
        "lon",
        "spatial_group",
        TARGET
    ]
].copy()

predictions_df[
    "predicted_probability"
] = test_probability

predictions_df[
    "predicted_label"
] = test_prediction

print(
    f"  Prediction rows: "
    f"{len(predictions_df)}"
)


# ============================================================
# 27. SAVE OUTPUTS
# ============================================================

print("\n[25] Saving final evaluation outputs...")
print("-" * 70)

metrics_df.to_csv(
    METRICS_FILE,
    index=False
)

confusion_df.to_csv(
    CONFUSION_FILE,
    index=False
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"  Saved: {METRICS_FILE}"
)

print(
    f"  Saved: {CONFUSION_FILE}"
)

print(
    f"  Saved: {PREDICTIONS_FILE}"
)


# ============================================================
# 28. FINAL VALIDATION
# ============================================================

print("\n[26] Final validation...")
print("-" * 70)

# Prediction count
if len(predictions_df) != len(test):
    raise RuntimeError(
        "Prediction count does not match test observations."
    )

# Probability range
if (
    predictions_df["predicted_probability"]
    .isna()
    .any()
):

    raise RuntimeError(
        "Missing predicted probabilities."
    )

if (
    (
        predictions_df["predicted_probability"]
        < 0
    )
    |
    (
        predictions_df["predicted_probability"]
        > 1
    )
).any():

    raise RuntimeError(
        "Predicted probabilities outside [0,1]."
    )


# Spatial separation
if overlap:

    raise RuntimeError(
        "Spatial overlap detected during final validation."
    )


# Both classes
if set(y_test.unique()) != {0, 1}:

    raise RuntimeError(
        "Final test set does not contain both classes."
    )


print(
    "  Prediction count       : PASSED"
)

print(
    "  Probability range      : PASSED"
)

print(
    "  Spatial separation     : PASSED"
)

print(
    "  Test class availability: PASSED"
)

print(
    "  Final evaluation       : PASSED"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 61 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal Random Forest configuration:")
print(
    f"  n_estimators     : "
    f"{selected_n_estimators}"
)

print(
    f"  max_features     : "
    f"{selected_max_features}"
)

print(
    f"  min_samples_leaf : "
    f"{selected_min_samples_leaf}"
)

print(
    f"  max_depth        : "
    f"{selected_max_depth}"
)

print("\nFinal independent spatial test:")
print(
    f"  Observations : "
    f"{len(test)}"
)

print(
    f"  ROC-AUC      : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"  PR-AUC       : "
    f"{test_metrics['pr_auc']:.4f}"
)

print(
    f"  Balanced Acc : "
    f"{test_metrics['balanced_accuracy']:.4f}"
)

print(
    f"  F1           : "
    f"{test_metrics['f1']:.4f}"
)

print("\nGeneralization:")
print(
    f"  ROC-AUC gap  : "
    f"{roc_gap:.4f}"
)

print(
    f"  PR-AUC gap   : "
    f"{pr_gap:.4f}"
)

print("\nImportant:")
print(
    "  This is the ONE final evaluation on the "
    "untouched spatial test set."
)

print(
    "  No hyperparameter tuning was performed using "
    "the final test observations."
)

print(
    "  The final model can now be used for interpretation "
    "and susceptibility mapping."
)

print("\nOutputs:")
print(
    f"  {METRICS_FILE}"
)

print(
    f"  {CONFUSION_FILE}"
)

print(
    f"  {PREDICTIONS_FILE}"
)

print("=" * 70)