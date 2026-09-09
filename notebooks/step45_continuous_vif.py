import pandas as pd
import numpy as np
from pathlib import Path

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# STEP 45 — CONTINUOUS FACTOR VIF
# Giri Rakshak Project
# ============================================================

print("=" * 70)
print("STEP 45 — CONTINUOUS FACTOR VIF")
print("=" * 70)


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "ner_master_factors.csv"
OUTPUT_FILE = PROCESSED_DIR / "ner_continuous_vif.csv"


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

print("\n[1] Loading master factor table...")

df = pd.read_csv(INPUT_FILE)

print(f"  Original rows: {len(df)}")


# ------------------------------------------------------------
# CONTINUOUS FACTORS
# ------------------------------------------------------------

factors = [
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]


# ------------------------------------------------------------
# COMPLETE CASES
# ------------------------------------------------------------

print("\n[2] Selecting complete observations...")

X = df[factors].copy()

X = X.dropna()

print(f"  Complete observations: {len(X)}")
print(f"  Excluded observations: {len(df) - len(X)}")


# ------------------------------------------------------------
# NUMERIC CONVERSION
# ------------------------------------------------------------

for col in factors:

    X[col] = pd.to_numeric(
        X[col],
        errors="coerce"
    )


if X.isna().any().any():

    raise ValueError(
        "Missing/non-numeric values remain."
    )


# ------------------------------------------------------------
# MATRIX RANK
# ------------------------------------------------------------

print("\n[3] Checking matrix rank...")

rank = np.linalg.matrix_rank(
    X.to_numpy(dtype=float)
)

print(f"  Matrix rank: {rank}")
print(f"  Number of factors: {X.shape[1]}")

if rank < X.shape[1]:

    raise ValueError(
        "Continuous predictor matrix is rank-deficient."
    )

print("  Full rank confirmed.")


# ------------------------------------------------------------
# VIF
# ------------------------------------------------------------

print("\n[4] Calculating VIF...")

results = []

X_values = X.to_numpy(dtype=float)

for i, factor in enumerate(factors):

    vif = variance_inflation_factor(
        X_values,
        i
    )

    tolerance = 1.0 / vif

    results.append({
        "factor": factor,
        "VIF": vif,
        "Tolerance": tolerance
    })


results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------

def classify_vif(vif):

    if vif < 5:
        return "Low"

    elif vif < 10:
        return "Moderate"

    else:
        return "High"


results_df["VIF_class"] = (
    results_df["VIF"]
    .apply(classify_vif)
)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------

print("\n[5] RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n[6] SUMMARY")
print("=" * 70)

print(
    f"Maximum VIF: "
    f"{results_df['VIF'].max():.4f}"
)

print(
    f"Minimum tolerance: "
    f"{results_df['Tolerance'].min():.4f}"
)

high = results_df[
    results_df["VIF"] >= 10
]

moderate = results_df[
    (results_df["VIF"] >= 5)
    &
    (results_df["VIF"] < 10)
]

print(
    f"VIF >= 10: {len(high)}"
)

print(
    f"VIF 5–10:  {len(moderate)}"
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

print("\n[7] Saving results...")

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"  Saved: {OUTPUT_FILE}"
)


print("\n" + "=" * 70)
print("STEP 45 COMPLETED SUCCESSFULLY")
print("=" * 70)