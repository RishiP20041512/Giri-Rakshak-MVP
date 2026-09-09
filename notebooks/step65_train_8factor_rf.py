from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

MODEL_FILE = PROCESSED / "ner_final_modeling_table_8factor.csv"
SPLIT_FILE = PROCESSED / "ner_spatial_split_05deg.csv"

CV_RESULTS = PROCESSED / "step65_8factor_spatial_cv_results.csv"
CV_SUMMARY = PROCESSED / "step65_8factor_spatial_cv_summary.csv"

TEST_RESULTS = PROCESSED / "step65_8factor_test_predictions.csv"
TEST_METRICS = PROCESSED / "step65_8factor_test_metrics.csv"
TEST_CM = PROCESSED / "step65_8factor_test_confusion_matrix.csv"

CONFIG_FILE = PROCESSED / "step65_8factor_rf_config.json"


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

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
GROUP = "spatial_group"
SPLIT = "split"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("8-FACTOR RANDOM FOREST MODEL")
print("=" * 70)

print("\nLoading modeling table...")
model_df = pd.read_csv(MODEL_FILE)

print("Loading spatial split...")
split_df = pd.read_csv(SPLIT_FILE)

print(f"\nModel rows : {len(model_df)}")
print(f"Split rows : {len(split_df)}")


# ============================================================
# MERGE / VERIFY SPATIAL SPLIT
# ============================================================

key_cols = [
    "lat",
    "lon",
    "label",
]

required_split_cols = key_cols + [GROUP, SPLIT]

missing_split_cols = [
    c for c in required_split_cols
    if c not in split_df.columns
]

if missing_split_cols:
    raise ValueError(
        f"Missing columns in spatial split: {missing_split_cols}"
    )


# Merge only split information.
split_info = split_df[
    key_cols + [GROUP, SPLIT]
].copy()

df = model_df.merge(
    split_info,
    on=key_cols,
    how="left",
    validate="one_to_one",
)

if len(df) != len(model_df):
    raise ValueError(
        "Merge changed row count. Check coordinate/label matching."
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\nChecking required columns...")

required = FEATURES + [TARGET, GROUP, SPLIT]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("All required columns found.")


print("\nChecking missing values...")

missing_counts = df[required].isna().sum()

print(missing_counts.to_string())

if missing_counts.sum() > 0:
    raise ValueError(
        "Missing values detected. Do not continue."
    )


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
# TRAIN / TEST SPLIT
# ============================================================

train_df = df[df[SPLIT] == "train"].copy()
test_df = df[df[SPLIT] == "test"].copy()

print("\n" + "=" * 70)
print("SPATIAL TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training samples : {len(train_df)}")
print(f"Test samples     : {len(test_df)}")

print(
    f"Training groups  : "
    f"{train_df[GROUP].nunique()}"
)

print(
    f"Test groups      : "
    f"{test_df[GROUP].nunique()}"
)

overlap = set(
    train_df[GROUP].unique()
).intersection(
    set(test_df[GROUP].unique())
)

print(f"Spatial group overlap: {len(overlap)}")

if overlap:
    raise ValueError(
        "Spatial leakage detected: train/test groups overlap."
    )


print("\nTraining class distribution:")
print(train_df[TARGET].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df[TARGET].value_counts().sort_index())


# ============================================================
# PREPARE X / y
# ============================================================

X_train = train_df[FEATURES].copy()
y_train = train_df[TARGET].copy()

X_test = test_df[FEATURES].copy()
y_test = test_df[TARGET].copy()

groups_train = train_df[GROUP].copy()


# ============================================================
# PREPROCESSING
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
print("MODEL CONFIGURATION")
print("=" * 70)

print(json.dumps(rf_params, indent=2))


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
# SPATIAL CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD SPATIAL CROSS-VALIDATION")
print("=" * 70)

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_rows = []

for fold, (train_idx, val_idx) in enumerate(
    cv.split(
        X_train,
        y_train,
        groups_train,
    ),
    start=1,
):

    print(f"\nFold {fold}/{N_SPLITS}")

    X_tr = X_train.iloc[train_idx]
    X_val = X_train.iloc[val_idx]

    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    groups_tr = groups_train.iloc[train_idx]
    groups_val = groups_train.iloc[val_idx]

    overlap_fold = set(
        groups_tr.unique()
    ).intersection(
        set(groups_val.unique())
    )

    if overlap_fold:
        raise ValueError(
            f"Fold {fold}: spatial group leakage detected."
        )

    fold_model = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
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
                ),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    **rf_params
                ),
            ),
        ]
    )

    fold_model.fit(X_tr, y_tr)

    prob = fold_model.predict_proba(X_val)[:, 1]

    pred = (
        prob >= 0.5
    ).astype(int)

    roc = roc_auc_score(
        y_val,
        prob,
    )

    pr = average_precision_score(
        y_val,
        prob,
    )

    acc = accuracy_score(
        y_val,
        pred,
    )

    bal = balanced_accuracy_score(
        y_val,
        pred,
    )

    precision = precision_score(
        y_val,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y_val,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_val,
        pred,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_val,
        pred,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    print(
        f"ROC-AUC={roc:.4f} | "
        f"PR-AUC={pr:.4f} | "
        f"Balanced Acc={bal:.4f} | "
        f"F1={f1:.4f}"
    )

    cv_rows.append(
        {
            "fold": fold,
            "n_train": len(train_idx),
            "n_validation": len(val_idx),
            "train_groups": groups_tr.nunique(),
            "validation_groups": groups_val.nunique(),
            "roc_auc": roc,
            "pr_auc": pr,
            "accuracy": acc,
            "balanced_accuracy": bal,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
        }
    )


cv_results = pd.DataFrame(cv_rows)

cv_results.to_csv(
    CV_RESULTS,
    index=False,
)


# ============================================================
# CV SUMMARY
# ============================================================

summary_rows = []

metrics = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
]

for metric in metrics:

    summary_rows.append(
        {
            "metric": metric,
            "mean": cv_results[metric].mean(),
            "std": cv_results[metric].std(
                ddof=1
            ),
            "min": cv_results[metric].min(),
            "max": cv_results[metric].max(),
        }
    )

cv_summary = pd.DataFrame(
    summary_rows
)

cv_summary.to_csv(
    CV_SUMMARY,
    index=False,
)

print("\n" + "=" * 70)
print("SPATIAL CV SUMMARY")
print("=" * 70)

for _, row in cv_summary.iterrows():

    print(
        f"{row['metric']:20s} "
        f"{row['mean']:.4f} ± "
        f"{row['std']:.4f}"
    )


# ============================================================
# FIT FINAL MODEL ON TRAINING DATA ONLY
# ============================================================

print("\n" + "=" * 70)
print("FITTING FINAL MODEL ON TRAINING DATA")
print("=" * 70)

model.fit(
    X_train,
    y_train,
)

train_prob = model.predict_proba(
    X_train
)[:, 1]

train_pred = (
    train_prob >= 0.5
).astype(int)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL UNTOUCHED SPATIAL TEST")
print("=" * 70)

test_prob = model.predict_proba(
    X_test
)[:, 1]

test_pred = (
    test_prob >= 0.5
).astype(int)


roc = roc_auc_score(
    y_test,
    test_prob,
)

pr = average_precision_score(
    y_test,
    test_prob,
)

acc = accuracy_score(
    y_test,
    test_pred,
)

bal = balanced_accuracy_score(
    y_test,
    test_pred,
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0,
)

recall = recall_score(
    y_test,
    test_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    test_pred,
    labels=[0, 1],
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else np.nan
)


# ============================================================
# TRAIN METRICS
# ============================================================

train_roc = roc_auc_score(
    y_train,
    train_prob,
)

train_pr = average_precision_score(
    y_train,
    train_prob,
)


metrics_dict = {
    "train_roc_auc": train_roc,
    "train_pr_auc": train_pr,
    "test_roc_auc": roc,
    "test_pr_auc": pr,
    "test_accuracy": acc,
    "test_balanced_accuracy": bal,
    "test_precision": precision,
    "test_recall": recall,
    "test_specificity": specificity,
    "test_f1": f1,
    "test_true_negative": tn,
    "test_false_positive": fp,
    "test_false_negative": fn,
    "test_true_positive": tp,
    "test_n": len(test_df),
}


metrics_df = pd.DataFrame(
    [metrics_dict]
)

metrics_df.to_csv(
    TEST_METRICS,
    index=False,
)

pd.DataFrame(
    cm,
    index=[
        "actual_0_background",
        "actual_1_landslide",
    ],
    columns=[
        "predicted_0_background",
        "predicted_1_landslide",
    ],
).to_csv(
    TEST_CM
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions = test_df[
    [
        "lat",
        "lon",
        "label",
        GROUP,
    ]
].copy()

predictions[
    "predicted_probability"
] = test_prob

predictions[
    "predicted_class"
] = test_pred

predictions.to_csv(
    TEST_RESULTS,
    index=False,
)


# ============================================================
# SAVE CONFIGURATION
# ============================================================

config = {
    "model": "RandomForestClassifier",
    "n_estimators": 700,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "max_depth": None,
    "random_state": RANDOM_STATE,
    "spatial_block_size": "0.5 degree",
    "cv": "StratifiedGroupKFold",
    "cv_folds": N_SPLITS,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "removed_factor": "Level_I LUCC",
    "unavailable_factor": "Lithology",
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        indent=2,
    )


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("8-FACTOR FINAL TEST RESULTS")
print("=" * 70)

print(f"ROC-AUC           : {roc:.4f}")
print(f"PR-AUC            : {pr:.4f}")
print(f"Accuracy           : {acc:.4f}")
print(f"Balanced Accuracy  : {bal:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"Specificity        : {specificity:.4f}")
print(f"F1 Score           : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nTrain/Test ROC-AUC gap:")
print(f"{train_roc - roc:.4f}")

print("\nSaved:")
print(CV_RESULTS)
print(CV_SUMMARY)
print(TEST_RESULTS)
print(TEST_METRICS)
print(TEST_CM)
print(CONFIG_FILE)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)