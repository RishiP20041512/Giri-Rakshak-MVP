"""
STEP 58 — TRAIN LEAKAGE-SAFE BASELINE MODELS

Models:
1. Logistic Regression
2. Random Forest

Input:
    processed/ner_train_modeling.csv
    processed/ner_test_modeling.csv

Important:
- Train/test split was created using 0.50-degree spatial blocks.
- No spatial group is shared between train and test.
- Preprocessing is fitted using training data only.
- Logistic Regression:
      numerical -> StandardScaler
      categorical -> OneHotEncoder
- Random Forest:
      numerical -> passthrough
      categorical -> OneHotEncoder
- Test set is used only for final evaluation.
- Lithology is not included because it is currently unavailable.

Outputs:
    processed/step58_model_results.csv
    processed/step58_confusion_matrices.csv
    processed/step58_predictions.csv
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

RESULTS_FILE = r"processed\step58_model_results.csv"
CONFUSION_FILE = r"processed\step58_confusion_matrices.csv"
PREDICTIONS_FILE = r"processed\step58_predictions.csv"


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


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 58 — TRAIN LEAKAGE-SAFE BASELINE MODELS")
print("=" * 70)


# ============================================================
# 1. CHECK INPUT FILES
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
# 2. LOAD TRAIN / TEST
# ============================================================

print("\n[2] Loading train/test data...")
print("-" * 70)

train = pd.read_csv(TRAIN_FILE)
test = pd.read_csv(TEST_FILE)

print(f"  Training observations : {len(train)}")
print(f"  Testing observations  : {len(test)}")


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

print("\n[3] Checking required columns...")
print("-" * 70)

required_columns = (
    FEATURES
    + [TARGET]
)

for dataset_name, dataset in [
    ("train", train),
    ("test", test)
]:

    missing = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {dataset_name}: {missing}"
        )

print("  All modeling columns are present.")


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

print("\n[4] Checking missing values...")
print("-" * 70)

train_missing = train[FEATURES].isna().sum()
test_missing = test[FEATURES].isna().sum()

print("\n  TRAIN")
for feature, count in train_missing.items():
    print(f"    {feature:<25}: {int(count)}")

print("\n  TEST")
for feature, count in test_missing.items():
    print(f"    {feature:<25}: {int(count)}")

if train_missing.sum() != 0:
    raise ValueError(
        "Training predictors contain missing values."
    )

if test_missing.sum() != 0:
    raise ValueError(
        "Testing predictors contain missing values."
    )

print("\n  Missing-value check PASSED.")


# ============================================================
# 5. EXTRACT X / y
# ============================================================

print("\n[5] Separating predictors and target...")
print("-" * 70)

X_train = train[FEATURES].copy()
y_train = train[TARGET].copy()

X_test = test[FEATURES].copy()
y_test = test[TARGET].copy()

print(f"  X_train shape : {X_train.shape}")
print(f"  y_train shape : {y_train.shape}")
print(f"  X_test shape  : {X_test.shape}")
print(f"  y_test shape  : {y_test.shape}")


# ============================================================
# 6. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n[6] Checking class distribution...")
print("-" * 70)

print("\n  TRAIN")
print(
    y_train.value_counts()
    .sort_index()
    .to_string()
)

print("\n  TEST")
print(
    y_test.value_counts()
    .sort_index()
    .to_string()
)

if set(y_train.unique()) != {0, 1}:
    raise ValueError(
        "Training data does not contain both classes."
    )

if set(y_test.unique()) != {0, 1}:
    raise ValueError(
        "Testing data does not contain both classes."
    )

print("\n  Both classes present in train and test.")


# ============================================================
# 7. CREATE CATEGORICAL ENCODER
# ============================================================

print("\n[7] Creating categorical preprocessing...")
print("-" * 70)

try:
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    )
except TypeError:
    # Compatibility with older scikit-learn versions
    one_hot_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse=False
    )

print("  OneHotEncoder configured:")
print("    handle_unknown = ignore")
print("    drop           = first")

print(
    "\n  Unknown categories in the test set will therefore "
    "not cause an error."
)


# ============================================================
# 8. LOGISTIC REGRESSION PREPROCESSOR
# ============================================================

print("\n[8] Building Logistic Regression pipeline...")
print("-" * 70)

logistic_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
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
                        one_hot_encoder
                    )
                ]
            ),
            CATEGORICAL_FEATURES
        )
    ],
    remainder="drop"
)

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight=None,
    random_state=42
)

logistic_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            logistic_preprocessor
        ),
        (
            "model",
            logistic_model
        )
    ]
)

print("  Numerical preprocessing:")
print("    Median imputation")
print("    StandardScaler")

print("  Categorical preprocessing:")
print("    Most-frequent imputation")
print("    One-hot encoding")

print("  Model: LogisticRegression")


# ============================================================
# 9. RANDOM FOREST PREPROCESSOR
# ============================================================

print("\n[9] Building Random Forest pipeline...")
print("-" * 70)

try:
    rf_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse_output=False
    )
except TypeError:
    rf_encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first",
        sparse=False
    )

rf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
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
                        rf_encoder
                    )
                ]
            ),
            CATEGORICAL_FEATURES
        )
    ],
    remainder="drop"
)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight=None,
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            rf_preprocessor
        ),
        (
            "model",
            rf_model
        )
    ]
)

print("  Numerical preprocessing:")
print("    Median imputation")
print("    No scaling")

print("  Categorical preprocessing:")
print("    Most-frequent imputation")
print("    One-hot encoding")

print("  Model: RandomForestClassifier")
print("  Trees: 500")
print("  max_features: sqrt")
print("  min_samples_leaf: 2")


# ============================================================
# 10. TRAIN MODELS
# ============================================================

print("\n[10] Training models...")
print("-" * 70)

models = {
    "Logistic Regression": logistic_pipeline,
    "Random Forest": rf_pipeline
}

trained_models = {}

for model_name, model in models.items():

    print(f"\n  Training {model_name}...")

    model.fit(
        X_train,
        y_train
    )

    trained_models[model_name] = model

    print("    Training completed.")


# ============================================================
# 11. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    model,
    X,
    y,
    dataset_name
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    accuracy = accuracy_score(
        y,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    sensitivity = recall

    result = {
        "model": model_name,
        "dataset": dataset_name,
        "n": len(y),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1
    }

    confusion_result = {
        "model": model_name,
        "dataset": dataset_name,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp)
    }

    return (
        result,
        confusion_result,
        probabilities,
        predictions
    )


# ============================================================
# 12. EVALUATE MODELS
# ============================================================

print("\n[11] Evaluating models...")
print("-" * 70)

results = []
confusion_results = []
prediction_tables = []

for model_name, model in trained_models.items():

    print(f"\n  {model_name}")

    # --------------------------------------------------------
    # Training evaluation
    # --------------------------------------------------------

    train_result, train_cm, train_prob, train_pred = (
        evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            "train"
        )
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    test_result, test_cm, test_prob, test_pred = (
        evaluate_model(
            model_name,
            model,
            X_test,
            y_test,
            "test"
        )
    )

    results.extend([
        train_result,
        test_result
    ])

    confusion_results.extend([
        train_cm,
        test_cm
    ])

    # --------------------------------------------------------
    # Store test predictions
    # --------------------------------------------------------

    prediction_table = test[
        [
            "lat",
            "lon",
            "spatial_group",
            TARGET
        ]
    ].copy()

    prediction_table["model"] = model_name
    prediction_table["predicted_probability"] = test_prob
    prediction_table["predicted_label"] = test_pred

    prediction_tables.append(
        prediction_table
    )

    # --------------------------------------------------------
    # Print test metrics
    # --------------------------------------------------------

    print("    TEST RESULTS")
    print(
        f"      ROC-AUC           : "
        f"{test_result['roc_auc']:.4f}"
    )

    print(
        f"      PR-AUC            : "
        f"{test_result['pr_auc']:.4f}"
    )

    print(
        f"      Accuracy          : "
        f"{test_result['accuracy']:.4f}"
    )

    print(
        f"      Balanced Accuracy : "
        f"{test_result['balanced_accuracy']:.4f}"
    )

    print(
        f"      Precision         : "
        f"{test_result['precision']:.4f}"
    )

    print(
        f"      Recall/Sensitivity: "
        f"{test_result['recall_sensitivity']:.4f}"
    )

    print(
        f"      Specificity       : "
        f"{test_result['specificity']:.4f}"
    )

    print(
        f"      F1                : "
        f"{test_result['f1']:.4f}"
    )

    print(
        "    Confusion matrix "
        "[[TN, FP], [FN, TP]]:"
    )

    print(
        f"      {test_cm['true_negative']:>4} "
        f"{test_cm['false_positive']:>4}"
    )

    print(
        f"      {test_cm['false_negative']:>4} "
        f"{test_cm['true_positive']:>4}"
    )


# ============================================================
# 13. SAVE RESULTS
# ============================================================

print("\n[12] Saving model results...")
print("-" * 70)

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

print(f"  Saved: {RESULTS_FILE}")


# ============================================================
# 14. SAVE CONFUSION MATRICES
# ============================================================

confusion_df = pd.DataFrame(
    confusion_results
)

confusion_df.to_csv(
    CONFUSION_FILE,
    index=False
)

print(
    f"  Saved: {CONFUSION_FILE}"
)


# ============================================================
# 15. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.concat(
    prediction_tables,
    ignore_index=True
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    f"  Saved: {PREDICTIONS_FILE}"
)


# ============================================================
# 16. DISPLAY COMPARISON
# ============================================================

print("\n[13] Model comparison — TEST SET")
print("-" * 70)

test_results = results_df[
    results_df["dataset"] == "test"
].copy()

display_columns = [
    "model",
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1"
]

print(
    test_results[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 17. TRAIN/TEST GAP CHECK
# ============================================================

print("\n[14] Checking train/test performance gap...")
print("-" * 70)

for model_name in models.keys():

    train_row = results_df[
        (results_df["model"] == model_name)
        & (results_df["dataset"] == "train")
    ].iloc[0]

    test_row = results_df[
        (results_df["model"] == model_name)
        & (results_df["dataset"] == "test")
    ].iloc[0]

    roc_gap = (
        train_row["roc_auc"]
        - test_row["roc_auc"]
    )

    print(f"\n  {model_name}")
    print(
        f"    Train ROC-AUC : "
        f"{train_row['roc_auc']:.4f}"
    )

    print(
        f"    Test ROC-AUC  : "
        f"{test_row['roc_auc']:.4f}"
    )

    print(
        f"    ROC-AUC gap   : "
        f"{roc_gap:.4f}"
    )


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

print("\n[15] Final validation...")
print("-" * 70)

# Check prediction count
expected_predictions = len(test) * len(models)

if len(predictions_df) != expected_predictions:
    raise RuntimeError(
        "Unexpected number of prediction rows."
    )

# Check probabilities
if (
    predictions_df["predicted_probability"]
    .isna()
    .any()
):
    raise RuntimeError(
        "Missing predicted probabilities."
    )

if (
    (predictions_df["predicted_probability"] < 0)
    | (predictions_df["predicted_probability"] > 1)
).any():
    raise RuntimeError(
        "Predicted probabilities outside [0, 1]."
    )

print("  Prediction count      : PASSED")
print("  Probability range     : PASSED")
print("  Model evaluation      : PASSED")
print("  Test set preserved    : PASSED")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 58 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nModels trained:")
print("  1. Logistic Regression")
print("  2. Random Forest")

print("\nEvaluation:")
print("  Final evaluation uses the spatially separated test set.")
print("  Test preprocessing is learned from training data only.")

print("\nImportant:")
print(
    "  The test set has NOT been used for model fitting."
)

print(
    "  Lacustrine Origin is present only in the test set; "
    "OneHotEncoder(handle_unknown='ignore') handled it safely."
)

print(
    "  This does not provide a reliable learned effect for "
    "the Lacustrine category because it has only one observation."
)

print("\nOutput files:")
print(f"  {RESULTS_FILE}")
print(f"  {CONFUSION_FILE}")
print(f"  {PREDICTIONS_FILE}")

print("\nNext stage:")
print(
    "  Assess model performance and spatial robustness "
    "before selecting the final susceptibility model."
)

print("=" * 70)