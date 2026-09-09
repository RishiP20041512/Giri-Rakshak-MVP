import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

file = "processed/training_data.csv"

df = pd.read_csv(file)

print("========================================")
print("STEP 17 — DATA VALIDATION")
print("========================================")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\nShape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print(" -", column)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# DUPLICATES
# ============================================================

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())


# ============================================================
# FEATURE STATISTICS
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

print("\nFeature statistics:")
print(df[features].describe().T)


# ============================================================
# CHECK INFINITE VALUES
# ============================================================

print("\nInfinite values:")

for column in features:

    count = np.isinf(
        df[column].to_numpy()
    ).sum()

    print(column, ":", count)


# ============================================================
# CHECK LABELS
# ============================================================

if not set(df["label"].unique()).issubset({0, 1}):

    raise ValueError(
        "Label must contain only 0 and 1."
    )


# ============================================================
# FINAL CHECK
# ============================================================

print("\n========================================")
print("VALIDATION COMPLETE")
print("========================================")

print("Dataset is ready for Random Forest.")