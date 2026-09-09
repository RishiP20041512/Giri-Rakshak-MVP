import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("=" * 70)
print("STEP 53 — CATEGORICAL + CONTINUOUS VIF")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_master_factors_9factor_geomorph_origin.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_vif_9factor_final_check.csv"
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("\n[1] Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"  Original observations: {len(df)}")

if len(df) != 1071:
    raise ValueError(
        f"Expected 1071 observations, found {len(df)}"
    )

# ---------------------------------------------------------
# 2. DEFINE FACTORS
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

categorical_factors = [
    "geomorph_origin",
    "Level_I",
]

all_factors = (
    continuous_factors
    + categorical_factors
)

print("\n[2] Factors")

print("\nContinuous:")
for factor in continuous_factors:
    print(f"  - {factor}")

print("\nCategorical:")
for factor in categorical_factors:
    print(f"  - {factor}")

# ---------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ---------------------------------------------------------
print("\n[3] Checking required columns...")

missing_columns = [
    factor
    for factor in all_factors
    if factor not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("  ✓ All required factors found.")

# ---------------------------------------------------------
# 4. COMPLETE-CASE SELECTION
# ---------------------------------------------------------
print("\n[4] Selecting complete observations...")

complete_mask = (
    df[all_factors]
    .notna()
    .all(axis=1)
)

model_df = df.loc[
    complete_mask
].copy()

excluded = len(df) - len(model_df)

print(f"  Complete observations: {len(model_df)}")
print(f"  Excluded observations: {excluded}")

print("\n  Labels:")
print(
    model_df["label"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# 5. CATEGORICAL FREQUENCIES
# ---------------------------------------------------------
print("\n[5] Categorical frequencies...")

for factor in categorical_factors:

    print(f"\n{factor}:")

    counts = (
        model_df[factor]
        .value_counts()
    )

    for category, count in counts.items():

        print(
            f"  {category}: {count}"
        )

# ---------------------------------------------------------
# 6. ONE-HOT ENCODING
# ---------------------------------------------------------
print("\n[6] One-hot encoding categorical factors...")

X_continuous = model_df[
    continuous_factors
].copy()

X_categorical = pd.get_dummies(
    model_df[categorical_factors],
    columns=categorical_factors,
    drop_first=True,
    dtype=float
)

print(
    f"  Continuous predictors: "
    f"{X_continuous.shape[1]}"
)

print(
    f"  Categorical dummy predictors: "
    f"{X_categorical.shape[1]}"
)

# ---------------------------------------------------------
# 7. COMBINE PREDICTORS
# ---------------------------------------------------------
X = pd.concat(
    [
        X_continuous,
        X_categorical
    ],
    axis=1
)

print(
    f"  Total predictors: "
    f"{X.shape[1]}"
)

print(
    f"  Observations: "
    f"{X.shape[0]}"
)

# ---------------------------------------------------------
# 8. CHECK MISSING VALUES
# ---------------------------------------------------------
print("\n[7] Checking encoded predictor matrix...")

missing_values = X.isna().sum().sum()

print(
    f"  Missing predictor values: "
    f"{missing_values}"
)

if missing_values > 0:
    raise ValueError(
        "Missing values remain after encoding."
    )

# ---------------------------------------------------------
# 9. CHECK CONSTANT VARIABLES
# ---------------------------------------------------------
print("\n[8] Checking constant predictors...")

constant_columns = [
    column
    for column in X.columns
    if X[column].nunique() <= 1
]

print(
    f"  Constant predictors: "
    f"{len(constant_columns)}"
)

if constant_columns:
    print("\nConstant columns:")
    for column in constant_columns:
        print(f"  - {column}")

# ---------------------------------------------------------
# 10. MATRIX RANK
# ---------------------------------------------------------
print("\n[9] Checking matrix rank...")

rank = np.linalg.matrix_rank(
    X.values
)

n_predictors = X.shape[1]

print(f"  Matrix rank: {rank}")
print(f"  Predictors : {n_predictors}")

if rank == n_predictors:
    print("  ✓ Full rank confirmed.")
else:
    print("  ⚠ Matrix is rank deficient.")

# ---------------------------------------------------------
# 11. CALCULATE VIF
# ---------------------------------------------------------
print("\n[10] Calculating VIF...")

results = []

for i, factor in enumerate(X.columns):

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
        "predictor": factor,
        "VIF": round(vif_value, 4),
        "Tolerance": round(tolerance, 4),
        "VIF_class": vif_class
    })

results_df = pd.DataFrame(results)

# ---------------------------------------------------------
# 12. DISPLAY CONTINUOUS RESULTS
# ---------------------------------------------------------
print("\n[11] CONTINUOUS FACTOR VIF")
print("=" * 70)

continuous_results = results_df[
    results_df["predictor"].isin(
        continuous_factors
    )
]

print(
    continuous_results.to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 13. DISPLAY CATEGORICAL RESULTS
# ---------------------------------------------------------
print("\n[12] CATEGORICAL DUMMY VIF")
print("=" * 70)

categorical_results = results_df[
    results_df["predictor"].str.startswith(
        "geomorph_origin_"
    )
    |
    results_df["predictor"].str.startswith(
        "Level_I_"
    )
]

print(
    categorical_results.to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 14. SUMMARY
# ---------------------------------------------------------
print("\n[13] OVERALL VIF SUMMARY")
print("=" * 70)

print(
    f"Maximum VIF: "
    f"{results_df['VIF'].max():.4f}"
)

print(
    f"Minimum tolerance: "
    f"{results_df['Tolerance'].min():.4f}"
)

high_vif = (
    results_df["VIF"] >= 10
).sum()

moderate_vif = (
    (results_df["VIF"] >= 5)
    & (results_df["VIF"] < 10)
).sum()

print(
    f"VIF >= 10: {high_vif}"
)

print(
    f"VIF 5–10:  {moderate_vif}"
)

# ---------------------------------------------------------
# 15. HIGH VIF DETAILS
# ---------------------------------------------------------
print("\n[14] VIF FLAGS")
print("=" * 70)

flags = results_df[
    results_df["VIF"] >= 5
]

if len(flags) == 0:

    print(
        "  ✓ No predictors have VIF >= 5."
    )

else:

    print(flags.to_string(index=False))

# ---------------------------------------------------------
# 16. SAVE RESULTS
# ---------------------------------------------------------
print("\n[15] Saving VIF results...")

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"  Saved: {OUTPUT_FILE}"
)

# ---------------------------------------------------------
# 17. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 53 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nDataset:")
print(f"  Original observations : {len(df)}")
print(f"  Complete observations : {len(model_df)}")
print(f"  Excluded observations : {excluded}")

print("\nPredictor structure:")
print(f"  Continuous factors   : {len(continuous_factors)}")
print(f"  Categorical factors  : {len(categorical_factors)}")
print(f"  Total encoded inputs : {X.shape[1]}")

print("\nImportant:")
print("  Categorical variables were one-hot encoded.")
print("  One reference category was dropped per factor.")
print("  No arbitrary ordinal ranking was applied.")
print("  Original geomorphology was retained.")
print("  No factor was automatically removed.")

print("\nOutput:")
print(f"  {OUTPUT_FILE}")

print("=" * 70)