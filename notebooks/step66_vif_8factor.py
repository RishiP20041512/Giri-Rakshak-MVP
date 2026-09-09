from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

INPUT = PROCESSED / "ner_final_modeling_table_8factor.csv"
OUTPUT = PROCESSED / "step66_vif_8factor_final.csv"


# ============================================================
# SETTINGS
# ============================================================

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


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("8-FACTOR FINAL VIF ANALYSIS")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"\nInput rows: {len(df)}")

required = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# COMPLETE CASE CHECK
# ============================================================

vif_df = df[required].dropna().copy()

print(
    f"Complete observations: {len(vif_df)}"
)

if len(vif_df) == 0:
    raise ValueError(
        "No complete observations available."
    )


# ============================================================
# NUMERIC VARIABLES
# ============================================================

numeric_df = vif_df[
    NUMERIC_FEATURES
].astype(float).copy()


# ============================================================
# CATEGORICAL VARIABLES
# ============================================================

categorical_df = pd.get_dummies(
    vif_df[CATEGORICAL_FEATURES],
    columns=CATEGORICAL_FEATURES,
    drop_first=True,
    dtype=float,
)


# ============================================================
# COMBINE
# ============================================================

X = pd.concat(
    [
        numeric_df,
        categorical_df,
    ],
    axis=1,
)

X = X.astype(float)

print(
    f"Final VIF design matrix: "
    f"{X.shape[0]} rows × {X.shape[1]} predictors"
)


# ============================================================
# REMOVE CONSTANT COLUMNS
# ============================================================

constant_columns = [
    c for c in X.columns
    if X[c].nunique() <= 1
]

if constant_columns:

    print(
        "\nRemoving constant columns:"
    )

    for c in constant_columns:
        print(f"  {c}")

    X = X.drop(
        columns=constant_columns
    )


# ============================================================
# RANK CHECK
# ============================================================

matrix_rank = np.linalg.matrix_rank(
    X.values
)

print(
    f"\nMatrix rank: {matrix_rank}"
)

print(
    f"Number of predictors: {X.shape[1]}"
)

if matrix_rank < X.shape[1]:

    print(
        "\nWARNING: VIF matrix is rank deficient."
    )

else:

    print(
        "Rank check: PASSED"
    )


# ============================================================
# ADD INTERCEPT
# ============================================================

X_with_const = sm.add_constant(
    X,
    has_constant="add",
)


# ============================================================
# CALCULATE VIF
# ============================================================

results = []

for i, column in enumerate(
    X_with_const.columns
):

    if column == "const":
        continue

    vif = variance_inflation_factor(
        X_with_const.values,
        i,
    )

    tolerance = (
        1.0 / vif
        if np.isfinite(vif) and vif != 0
        else np.nan
    )

    if vif < 5:
        interpretation = "Low"
    elif vif < 10:
        interpretation = "Moderate"
    else:
        interpretation = "High"

    results.append(
        {
            "predictor": column,
            "vif": vif,
            "tolerance": tolerance,
            "interpretation": interpretation,
        }
    )


results_df = pd.DataFrame(
    results
).sort_values(
    "vif",
    ascending=False,
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("VIF RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)


# ============================================================
# SUMMARY
# ============================================================

max_vif = results_df["vif"].max()

min_tolerance = (
    results_df["tolerance"].min()
)

high_vif = results_df[
    results_df["vif"] >= 10
]

moderate_vif = results_df[
    (results_df["vif"] >= 5)
    & (results_df["vif"] < 10)
]


print("\n" + "=" * 70)
print("VIF SUMMARY")
print("=" * 70)

print(
    f"Maximum VIF       : {max_vif:.4f}"
)

print(
    f"Minimum tolerance : {min_tolerance:.4f}"
)

print(
    f"VIF >= 10         : {len(high_vif)}"
)

print(
    f"VIF 5-10          : {len(moderate_vif)}"
)


if len(high_vif) == 0:

    print(
        "\nNo severe multicollinearity "
        "(VIF >= 10) detected."
    )

else:

    print(
        "\nWARNING: Severe "
        "multicollinearity detected."
    )


print(
    f"\nSaved: {OUTPUT}"
)

print("\nDONE")