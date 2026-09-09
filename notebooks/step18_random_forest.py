import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "processed/training_data.csv"
)

print("Training samples:", len(df))


# ============================================================
# FEATURES
# ============================================================

features = [
    "elevation",
    "slope",
    "aspect",
    "ndvi",
    "rainfall_3day",
    "soil_moisture",
    "dist_to_road",
    "dist_to_river"
]

X = df[features]

y = df["label"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining labels:")
print(y_train.value_counts())

print("\nTesting labels:")
print(y_test.value_counts())


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# EVALUATION
# ============================================================

print("\n========================================")
print("MODEL RESULTS")
print("========================================")

print(
    "\nAccuracy:",
    accuracy_score(y_test, y_pred)
)

print("\nConfusion matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# ROC AUC
# ============================================================

if len(set(y_test)) == 2:

    print(
        "ROC-AUC:",
        roc_auc_score(
            y_test,
            y_probability
        )
    )

else:

    print(
        "ROC-AUC cannot be calculated because "
        "the test set contains only one class."
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_file = (
    "processed/random_forest_model.joblib"
)

joblib.dump(
    model,
    model_file
)

print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    "Saved:",
    model_file
)