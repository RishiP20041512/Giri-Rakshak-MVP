"""
STEP 62 — FINAL RANDOM FOREST FEATURE IMPORTANCE
================================================

Purpose
-------
Interpret the finalized Random Forest model using permutation
importance and Random Forest impurity importance.

Final model:
    n_estimators     = 700
    max_features     = sqrt
    min_samples_leaf = 5
    max_depth        = None

Final spatial test:
    203 observations

Important:
    - No hyperparameter tuning is performed here.
    - No model selection is performed here.
    - The train/test split is unchanged.
    - The final model configuration is unchanged.
    - The test observations are used only for model interpretation.

Primary interpretation:
    Permutation importance at the ORIGINAL predictor level.

Metrics:
    1. ROC-AUC
    2. PR-AUC

Predictors:
    1. elevation_m
    2. slope_deg
    3. rainfall_3day
    4. soil_moisture
    5. ndvi
    6. distance_to_road_m
    7. lineament_density
    8. geomorph_origin
    9. Level_I

Outputs:
    processed\step62_permutation_importance.csv
    processed\step62_impurity_importance.csv
    processed\step62_feature_importance_summary.csv
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ================================================================
# 0. SUPPRESS ONLY THE KNOWN HARMLESS ENCODER WARNING
# ================================================================

warnings.filterwarnings(
    "ignore",
    message="Found unknown categories in columns"
)


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "processed"

TRAIN_FILE = (
    PROCESSED_DIR /
    "ner_train_modeling.csv"
)

TEST_FILE = (
    PROCESSED_DIR /
    "ner_test_modeling.csv"
)

CONFIG_FILE = (
    PROCESSED_DIR /
    "step60_selected_rf_config.csv"
)

OUTPUT_PERMUTATION = (
    PROCESSED_DIR /
    "step62_permutation_importance.csv"
)

OUTPUT_IMPURITY = (
    PROCESSED_DIR /
    "step62_impurity_importance.csv"
)

OUTPUT_SUMMARY = (
    PROCESSED_DIR /
    "step62_feature_importance_summary.csv"
)


# ================================================================
# 2. MODEL FEATURES
# ================================================================

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
    "Level_I",
]

ALL_FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)

TARGET = "label"

RANDOM_STATE = 42


# ================================================================
# 3. HELPER FUNCTION
# ================================================================

def check_file(path):
    """
    Check whether a required input file exists.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )


# ================================================================
# 4. HEADER
# ================================================================

print("=" * 70)
print("STEP 62 — FINAL RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)


# ================================================================
# 5. CHECK INPUT FILES
# ================================================================

print("\n[1] Checking input files...")
print("-" * 70)

for file_path in [
    TRAIN_FILE,
    TEST_FILE,
    CONFIG_FILE,
]:

    check_file(file_path)

    print(
        f"  Found: "
        f"{file_path.relative_to(PROJECT_ROOT)}"
    )


# ================================================================
# 6. LOAD TRAINING AND TEST DATA
# ================================================================

print("\n[2] Loading train/test datasets...")
print("-" * 70)

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"  Training observations : "
    f"{len(train_df)}"
)

print(
    f"  Final test observations: "
    f"{len(test_df)}"
)


# ================================================================
# 7. VERIFY EXPECTED DATASET SIZE
# ================================================================

print("\n[3] Verifying expected dataset sizes...")
print("-" * 70)

if len(train_df) != 788:
    raise ValueError(
        f"Expected 788 training observations, "
        f"found {len(train_df)}."
    )

if len(test_df) != 203:
    raise ValueError(
        f"Expected 203 test observations, "
        f"found {len(test_df)}."
    )

print("  Training size : PASSED")
print("  Test size     : PASSED")


# ================================================================
# 8. VERIFY REQUIRED COLUMNS
# ================================================================

print("\n[4] Checking required columns...")
print("-" * 70)

required_columns = (
    ALL_FEATURES +
    [TARGET]
)

missing_train = [
    col
    for col in required_columns
    if col not in train_df.columns
]

missing_test = [
    col
    for col in required_columns
    if col not in test_df.columns
]

if missing_train:
    raise ValueError(
        "Missing columns in training dataset:\n"
        + str(missing_train)
    )

if missing_test:
    raise ValueError(
        "Missing columns in testing dataset:\n"
        + str(missing_test)
    )

print(
    "  All required columns are present."
)


# ================================================================
# 9. PREPARE X AND y
# ================================================================

print("\n[5] Preparing modeling data...")
print("-" * 70)

X_train = train_df[
    ALL_FEATURES
].copy()

y_train = train_df[
    TARGET
].astype(int).copy()

X_test = test_df[
    ALL_FEATURES
].copy()

y_test = test_df[
    TARGET
].astype(int).copy()

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


# ================================================================
# 10. CHECK MISSING VALUES
# ================================================================

print("\n[6] Checking missing predictor values...")
print("-" * 70)

train_missing = int(
    X_train.isna().sum().sum()
)

test_missing = int(
    X_test.isna().sum().sum()
)

print(
    f"  Training missing predictor values: "
    f"{train_missing}"
)

print(
    f"  Testing missing predictor values : "
    f"{test_missing}"
)

if train_missing != 0:
    raise ValueError(
        "Training predictors contain missing values."
    )

if test_missing != 0:
    raise ValueError(
        "Testing predictors contain missing values."
    )

print(
    "  Missing-value check PASSED."
)


# ================================================================
# 11. CHECK SPATIAL SEPARATION
# ================================================================

print("\n[7] Verifying spatial separation...")
print("-" * 70)

if "spatial_group" not in train_df.columns:
    raise ValueError(
        "spatial_group column not found in training dataset."
    )

if "spatial_group" not in test_df.columns:
    raise ValueError(
        "spatial_group column not found in testing dataset."
    )

train_groups = set(
    train_df["spatial_group"]
)

test_groups = set(
    test_df["spatial_group"]
)

overlap = (
    train_groups
    .intersection(test_groups)
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

if len(overlap) != 0:
    raise ValueError(
        "Spatial group overlap detected:\n"
        + str(sorted(overlap))
    )

print(
    "  Spatial separation PASSED."
)


# ================================================================
# 12. LOAD FINAL RF CONFIGURATION
# ================================================================

print("\n[8] Loading selected Random Forest configuration...")
print("-" * 70)

config_df = pd.read_csv(
    CONFIG_FILE
)

if config_df.empty:
    raise ValueError(
        "Selected RF configuration file is empty."
    )

config = config_df.iloc[0]


# ================================================================
# 13. READ CONFIGURATION
# ================================================================

n_estimators = int(
    config["n_estimators"]
)

max_features = str(
    config["max_features"]
)

min_samples_leaf = int(
    config["min_samples_leaf"]
)

max_depth_value = config["max_depth"]

if pd.isna(max_depth_value):

    max_depth = None

else:

    max_depth = int(
        max_depth_value
    )


print(
    f"  n_estimators     : "
    f"{n_estimators}"
)

print(
    f"  max_features     : "
    f"{max_features}"
)

print(
    f"  min_samples_leaf : "
    f"{min_samples_leaf}"
)

print(
    f"  max_depth        : "
    f"{max_depth}"
)


# ================================================================
# 14. VERIFY FINAL CONFIGURATION
# ================================================================

print("\n[9] Verifying final selected configuration...")
print("-" * 70)

if n_estimators != 700:
    raise ValueError(
        f"Unexpected n_estimators: "
        f"{n_estimators}"
    )

if max_features != "sqrt":
    raise ValueError(
        f"Unexpected max_features: "
        f"{max_features}"
    )

if min_samples_leaf != 5:
    raise ValueError(
        f"Unexpected min_samples_leaf: "
        f"{min_samples_leaf}"
    )

if max_depth is not None:
    raise ValueError(
        f"Unexpected max_depth: "
        f"{max_depth}"
    )

print(
    "  Configuration matches Step 60 selection."
)


# ================================================================
# 15. BUILD NUMERIC PREPROCESSOR
# ================================================================

print("\n[10] Building preprocessing pipeline...")
print("-" * 70)

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)


# ================================================================
# 16. BUILD CATEGORICAL PREPROCESSOR
# ================================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False
            )
        )
    ]
)


# ================================================================
# 17. BUILD COLUMN TRANSFORMER
# ================================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            NUMERIC_FEATURES
        ),
        (
            "cat",
            categorical_transformer,
            CATEGORICAL_FEATURES
        ),
    ],
    remainder="drop"
)


# ================================================================
# 18. BUILD FINAL RANDOM FOREST
# ================================================================

print("\n[11] Building final Random Forest...")
print("-" * 70)

rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_features=max_features,
    min_samples_leaf=min_samples_leaf,
    max_depth=max_depth,
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            rf
        ),
    ]
)

print(
    "  Final Random Forest configured."
)


# ================================================================
# 19. FIT MODEL ON TRAINING DATA ONLY
# ================================================================

print("\n[12] Fitting final model...")
print("-" * 70)

print(
    "  Training on all 788 training observations."
)

print(
    "  Final test observations are NOT used for fitting."
)

model.fit(
    X_train,
    y_train
)

print(
    "  Model fitting completed."
)


# ================================================================
# 20. GENERATE TEST PROBABILITIES
# ================================================================

print("\n[13] Generating final test probabilities...")
print("-" * 70)

test_probability = (
    model
    .predict_proba(X_test)[:, 1]
)

print(
    f"  Predictions generated: "
    f"{len(test_probability)}"
)

print(
    f"  Minimum probability: "
    f"{test_probability.min():.6f}"
)

print(
    f"  Maximum probability: "
    f"{test_probability.max():.6f}"
)


# ================================================================
# 21. CALCULATE BASELINE TEST METRICS
# ================================================================

print("\n[14] Calculating baseline final-test performance...")
print("-" * 70)

baseline_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

baseline_pr_auc = (
    average_precision_score(
        y_test,
        test_probability
    )
)

print(
    f"  Baseline ROC-AUC : "
    f"{baseline_roc_auc:.4f}"
)

print(
    f"  Baseline PR-AUC  : "
    f"{baseline_pr_auc:.4f}"
)


# ================================================================
# 22. PERMUTATION IMPORTANCE — ROC-AUC
# ================================================================

print("\n[15] Calculating permutation importance...")
print("-" * 70)

print(
    "  Metric          : ROC-AUC"
)

print(
    "  Repetitions     : 30"
)

print(
    "  Permutation unit: original predictor"
)

roc_perm = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

roc_importance_mean = (
    roc_perm.importances_mean
)

roc_importance_std = (
    roc_perm.importances_std
)


# ================================================================
# 23. PERMUTATION IMPORTANCE — PR-AUC
# ================================================================

print("\n[16] Calculating permutation importance...")
print("-" * 70)

print(
    "  Metric          : PR-AUC"
)

print(
    "  Repetitions     : 30"
)

print(
    "  Permutation unit: original predictor"
)

pr_perm = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="average_precision",
    n_repeats=30,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

pr_importance_mean = (
    pr_perm.importances_mean
)

pr_importance_std = (
    pr_perm.importances_std
)


# ================================================================
# 24. CREATE PERMUTATION TABLE
# ================================================================

print("\n[17] Creating permutation importance table...")
print("-" * 70)

permutation_df = pd.DataFrame(
    {
        "feature": ALL_FEATURES,

        "roc_auc_importance_mean":
            roc_importance_mean,

        "roc_auc_importance_std":
            roc_importance_std,

        "pr_auc_importance_mean":
            pr_importance_mean,

        "pr_auc_importance_std":
            pr_importance_std,
    }
)


# ================================================================
# 25. CALCULATE RANKS
# ================================================================

permutation_df[
    "roc_auc_rank"
] = (
    permutation_df[
        "roc_auc_importance_mean"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

permutation_df[
    "pr_auc_rank"
] = (
    permutation_df[
        "pr_auc_importance_mean"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

permutation_df[
    "mean_rank"
] = (
    permutation_df[
        [
            "roc_auc_rank",
            "pr_auc_rank"
        ]
    ]
    .mean(axis=1)
)


# ================================================================
# 26. SORT FEATURES
# ================================================================

permutation_df = (
    permutation_df
    .sort_values(
        by=[
            "mean_rank",
            "roc_auc_importance_mean",
        ],
        ascending=[
            True,
            False,
        ]
    )
    .reset_index(drop=True)
)

permutation_df[
    "overall_rank"
] = np.arange(
    1,
    len(permutation_df) + 1
)


# ================================================================
# 27. EXTRACT TRANSFORMED FEATURE NAMES
# ================================================================

print("\n[18] Extracting transformed feature names...")
print("-" * 70)

fitted_preprocessor = (
    model.named_steps[
        "preprocessor"
    ]
)

transformed_feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

print(
    f"  Transformed predictors: "
    f"{len(transformed_feature_names)}"
)


# ================================================================
# 28. GET RF CLASSIFIER
# ================================================================

classifier = (
    model.named_steps[
        "classifier"
    ]
)

impurity_values = (
    classifier.feature_importances_
)


# ================================================================
# 29. CHECK IMPURITY LENGTH
# ================================================================

print("\n[19] Checking impurity importance...")
print("-" * 70)

if len(impurity_values) != len(
    transformed_feature_names
):

    raise ValueError(
        "Mismatch between transformed "
        "feature names and RF importance values."
    )

print(
    "  Impurity importance dimensions: PASSED"
)


# ================================================================
# 30. TRANSFORMED IMPURITY IMPORTANCE
# ================================================================

transformed_importance_df = pd.DataFrame(
    {
        "transformed_feature":
            transformed_feature_names,

        "impurity_importance":
            impurity_values,
    }
)

transformed_importance_df = (
    transformed_importance_df
    .sort_values(
        by="impurity_importance",
        ascending=False
    )
    .reset_index(drop=True)
)

transformed_importance_df[
    "rank"
] = np.arange(
    1,
    len(transformed_importance_df) + 1
)


# ================================================================
# 31. AGGREGATE IMPURITY IMPORTANCE
#     BACK TO ORIGINAL FACTORS
# ================================================================

print(
    "\n[20] Aggregating impurity importance "
    "to original factors..."
)

print("-" * 70)

aggregated_rows = []

for feature in ALL_FEATURES:

    total_importance = 0.0

    for name, value in zip(
        transformed_feature_names,
        impurity_values
    ):

        clean_name = name

        if clean_name.startswith(
            "num__"
        ):

            clean_name = (
                clean_name.replace(
                    "num__",
                    "",
                    1
                )
            )

        elif clean_name.startswith(
            "cat__"
        ):

            clean_name = (
                clean_name.replace(
                    "cat__",
                    "",
                    1
                )
            )

        # Exact original numeric feature
        if clean_name == feature:

            total_importance += float(
                value
            )

        # One-hot categorical feature
        elif clean_name.startswith(
            feature + "_"
        ):

            total_importance += float(
                value
            )

    aggregated_rows.append(
        {
            "feature": feature,

            "impurity_importance":
                total_importance,
        }
    )


# ================================================================
# 32. CREATE ORIGINAL-FACTOR IMPURITY TABLE
# ================================================================

impurity_original_df = pd.DataFrame(
    aggregated_rows
)

impurity_original_df = (
    impurity_original_df
    .sort_values(
        by="impurity_importance",
        ascending=False
    )
    .reset_index(drop=True)
)

impurity_original_df[
    "impurity_rank"
] = np.arange(
    1,
    len(impurity_original_df) + 1
)


# ================================================================
# 33. CREATE FINAL SUMMARY
# ================================================================

print("\n[21] Creating final feature-importance summary...")
print("-" * 70)

summary_df = permutation_df.merge(
    impurity_original_df[
        [
            "feature",
            "impurity_importance",
            "impurity_rank",
        ]
    ],
    on="feature",
    how="left"
)


# ================================================================
# 34. ORDER FINAL SUMMARY
# ================================================================

summary_df = summary_df[
    [
        "overall_rank",
        "feature",

        "roc_auc_importance_mean",
        "roc_auc_importance_std",
        "roc_auc_rank",

        "pr_auc_importance_mean",
        "pr_auc_importance_std",
        "pr_auc_rank",

        "mean_rank",

        "impurity_importance",
        "impurity_rank",
    ]
]


# ================================================================
# 35. SAVE PERMUTATION IMPORTANCE
# ================================================================

print("\n[22] Saving output files...")
print("-" * 70)

permutation_df.to_csv(
    OUTPUT_PERMUTATION,
    index=False
)

print(
    "  Saved: "
    f"{OUTPUT_PERMUTATION.relative_to(PROJECT_ROOT)}"
)


# ================================================================
# 36. SAVE IMPURITY IMPORTANCE
# ================================================================

impurity_original_df.to_csv(
    OUTPUT_IMPURITY,
    index=False
)

print(
    "  Saved: "
    f"{OUTPUT_IMPURITY.relative_to(PROJECT_ROOT)}"
)


# ================================================================
# 37. SAVE FINAL SUMMARY
# ================================================================

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False
)

print(
    "  Saved: "
    f"{OUTPUT_SUMMARY.relative_to(PROJECT_ROOT)}"
)


# ================================================================
# 38. PRINT FINAL RANKING
# ================================================================

print("\n[23] FINAL FEATURE IMPORTANCE RANKING")
print("-" * 70)

display_columns = [
    "overall_rank",
    "feature",
    "roc_auc_importance_mean",
    "pr_auc_importance_mean",
    "impurity_importance",
]

print(
    summary_df[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ================================================================
# 39. PRINT ROC-AUC RANKING
# ================================================================

print("\n[24] ROC-AUC PERMUTATION RANKING")
print("-" * 70)

roc_display = (
    summary_df[
        [
            "roc_auc_rank",
            "feature",
            "roc_auc_importance_mean",
            "roc_auc_importance_std",
        ]
    ]
    .sort_values(
        "roc_auc_rank"
    )
)

print(
    roc_display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ================================================================
# 40. PRINT PR-AUC RANKING
# ================================================================

print("\n[25] PR-AUC PERMUTATION RANKING")
print("-" * 70)

pr_display = (
    summary_df[
        [
            "pr_auc_rank",
            "feature",
            "pr_auc_importance_mean",
            "pr_auc_importance_std",
        ]
    ]
    .sort_values(
        "pr_auc_rank"
    )
)

print(
    pr_display.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.6f}"
    )
)


# ================================================================
# 41. VALIDATION CHECKS
# ================================================================

print("\n[26] Final validation...")
print("-" * 70)

checks = {}


checks[
    "Training observation count"
] = (
    len(X_train) == 788
)


checks[
    "Test observation count"
] = (
    len(X_test) == 203
)


checks[
    "Nine predictors"
] = (
    len(ALL_FEATURES) == 9
)


checks[
    "Permutation rows"
] = (
    len(permutation_df) == 9
)


checks[
    "Impurity rows"
] = (
    len(impurity_original_df) == 9
)


checks[
    "Summary rows"
] = (
    len(summary_df) == 9
)


checks[
    "Finite ROC importance"
] = (
    np.isfinite(
        permutation_df[
            "roc_auc_importance_mean"
        ]
    ).all()
)


checks[
    "Finite PR importance"
] = (
    np.isfinite(
        permutation_df[
            "pr_auc_importance_mean"
        ]
    ).all()
)


checks[
    "Spatial separation"
] = (
    len(overlap) == 0
)


checks[
    "Probability range"
] = (
    np.all(
        (test_probability >= 0.0)
        &
        (test_probability <= 1.0)
    )
)


for check_name, result in checks.items():

    status = (
        "PASSED"
        if result
        else
        "FAILED"
    )

    print(
        f"  {check_name:<30} : "
        f"{status}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "One or more Step 62 "
        "validation checks failed."
    )


# ================================================================
# 42. COMPLETION MESSAGE
# ================================================================

print("\n" + "=" * 70)
print("STEP 62 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal model:")
print(
    "  Random Forest"
)

print(
    "  n_estimators     = 700"
)

print(
    "  max_features     = sqrt"
)

print(
    "  min_samples_leaf = 5"
)

print(
    "  max_depth        = None"
)

print("\nFinal test performance used for interpretation:")
print(
    f"  ROC-AUC = {baseline_roc_auc:.4f}"
)

print(
    f"  PR-AUC  = {baseline_pr_auc:.4f}"
)

print("\nPrimary importance method:")
print(
    "  Permutation importance"
)

print("\nSecondary diagnostic:")
print(
    "  Random Forest impurity importance"
)

print("\nFinal outputs:")
print(
    "  processed\\step62_permutation_importance.csv"
)

print(
    "  processed\\step62_impurity_importance.csv"
)

print(
    "  processed\\step62_feature_importance_summary.csv"
)

print("\nImportant:")
print(
    "  The final model configuration was not changed."
)

print(
    "  No hyperparameter tuning was performed."
)

print(
    "  No feature selection was performed."
)

print(
    "  The nine-factor modeling framework remains unchanged."
)

print("=" * 70)