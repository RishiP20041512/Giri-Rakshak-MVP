import pandas as pd
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

print("=" * 70)
print("STEP 50 — CONTINUOUS VIF FOR 9-FACTOR DATASET")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_master_factors_9factor.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_continuous_vif_9factor.csv"
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("\n[1] Loading 9-factor master table...")

df = pd.read_csv(INPUT_FILE)

print(f"  Original rows: {len(df)}")

if len(df) != 1071:
    raise ValueError(
        f"Expected 1071 rows, found {len(df)}"
    )

# ---------------------------------------------------------
# 2. DEFINE CONTINUOUS FACTORS
# ---------------------------------------------------------
continuous_factors = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

print("\n[2] Continuous factors:")

for factor in continuous_factors:
    print(f"  - {factor}")

# ---------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ---------------------------------------------------------
print("\n[3] Checking required columns...")

missing_columns = [
    factor
    for factor in continuous_factors
    if factor not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("  All 7 continuous factors found.")

# ---------------------------------------------------------
# 4. SELECT COMPLETE OBSERVATIONS
# ---------------------------------------------------------
print("\n[4] Selecting complete observations...")

vif_df = df[continuous_factors].dropna().copy()

print(f"  Complete observations: {len(vif_df)}")
print(
    f"  Excluded observations: "
    f"{len(df) - len(vif_df)}"
)

# ---------------------------------------------------------
# 5. CHECK MISSING VALUES
# ---------------------------------------------------------
print("\n[5] Checking missing values...")

for factor in continuous_factors:

    missing = int(vif_df[factor].isna().sum())

    print(f"  {factor}: {missing}")

    if missing > 0:
        raise ValueError(
            f"Missing values remain in {factor}"
        )

# ---------------------------------------------------------
# 6. CHECK CONSTANT VARIABLES
# ---------------------------------------------------------
print("\n[6] Checking constant variables...")

for factor in continuous_factors:

    unique_count = vif_df[factor].nunique()

    print(
        f"  {factor}: "
        f"{unique_count} unique values"
    )

    if unique_count <= 1:
        raise ValueError(
            f"Constant variable detected: {factor}"
        )

# ---------------------------------------------------------
# 7. CONVERT TO NUMERIC
# ---------------------------------------------------------
print("\n[7] Checking numeric data types...")

X = vif_df[continuous_factors].apply(
    pd.to_numeric,
    errors="coerce"
)

if X.isna().any().any():
    raise ValueError(
        "Non-numeric or invalid values detected."
    )

print("  All continuous factors are numeric.")

# ---------------------------------------------------------
# 8. CHECK MATRIX RANK
# ---------------------------------------------------------
print("\n[8] Checking matrix rank...")

rank = np.linalg.matrix_rank(
    X.values
)

n_factors = X.shape[1]

print(f"  Matrix rank: {rank}")
print(f"  Number of factors: {n_factors}")

if rank == n_factors:
    print("  Full rank confirmed.")
else:
    print("  WARNING: Matrix is rank deficient.")

# ---------------------------------------------------------
# 9. CALCULATE VIF
# ---------------------------------------------------------
print("\n[9] Calculating VIF...")

results = []

for i, factor in enumerate(continuous_factors):

    vif_value = variance_inflation_factor(
        X.values,
        i
    )

    tolerance = 1.0 / vif_value

    if vif_value < 5:
        vif_class = "Low"
    elif vif_value < 10:
        vif_class = "Moderate"
    else:
        vif_class = "High"

    results.append({
        "factor": factor,
        "VIF": round(vif_value, 4),
        "Tolerance": round(tolerance, 4),
        "VIF_class": vif_class
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 10. DISPLAY RESULTS
# ---------------------------------------------------------
print("\n[10] RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 11. SUMMARY
# ---------------------------------------------------------
print("\n[11] SUMMARY")
print("=" * 70)

max_vif = results_df["VIF"].max()
min_tolerance = results_df["Tolerance"].min()

vif_high = (
    results_df["VIF"] >= 10
).sum()

vif_moderate = (
    (results_df["VIF"] >= 5)
    & (results_df["VIF"] < 10)
).sum()

print(f"Maximum VIF: {max_vif:.4f}")
print(f"Minimum tolerance: {min_tolerance:.4f}")
print(f"VIF >= 10: {vif_high}")
print(f"VIF 5–10:  {vif_moderate}")

# ---------------------------------------------------------
# 12. SAVE RESULTS
# ---------------------------------------------------------
print("\n[12] Saving results...")

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"  Saved: {OUTPUT_FILE}")

# ---------------------------------------------------------
# 13. FINAL INTERPRETATION
# ---------------------------------------------------------
print("\n[13] INTERPRETATION")
print("=" * 70)

if vif_high == 0 and vif_moderate == 0:

    print(
        "  ✓ No continuous factor has VIF >= 5."
    )

    print(
        "  ✓ Continuous predictors show low "
        "multicollinearity."
    )

else:

    print(
        "  ⚠ One or more continuous factors "
        "have VIF >= 5."
    )

print(
    "  Tolerance is calculated as 1 / VIF."
)

# ---------------------------------------------------------
# 14. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 50 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nDataset:")
print(f"  Original observations : {len(df)}")
print(f"  VIF observations      : {len(vif_df)}")
print(
    f"  Excluded observations : "
    f"{len(df) - len(vif_df)}"
)

print("\nContinuous factors analyzed:")

for i, factor in enumerate(
    continuous_factors,
    start=1
):
    print(f"  {i}. {factor}")

print("\nOutput:")
print(f"  {OUTPUT_FILE}")

print("=" * 70)