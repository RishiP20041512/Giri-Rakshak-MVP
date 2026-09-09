import pandas as pd
import numpy as np
from pathlib import Path

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# STEP 44 — CORRECTED VIF AND TOLERANCE CALCULATION
# Giri Rakshak Project
# ============================================================

print("=" * 75)
print("STEP 44 — CORRECTED VIF AND TOLERANCE CALCULATION")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"

# IMPORTANT:
# Read the original master table because it contains the
# original categorical variables.
INPUT_FILE = PROCESSED_DIR / "ner_master_factors.csv"

OUTPUT_FILE = (
    PROCESSED_DIR /
    "ner_vif_results_corrected.csv"
)


# ------------------------------------------------------------
# 2. LOAD MASTER DATA
# ------------------------------------------------------------

print("\n[1] Loading master factor table...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Master factor table not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"  Rows:    {len(df)}")
print(f"  Columns: {len(df.columns)}")


if len(df) != 1071:
    raise ValueError(
        f"Expected 1071 rows, found {len(df)}"
    )


# ------------------------------------------------------------
# 3. DEFINE FACTORS
# ------------------------------------------------------------

continuous_factors = [
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

categorical_factors = [
    "geomorphology",
    "Level_I",
]


# ------------------------------------------------------------
# 4. CHECK COLUMNS
# ------------------------------------------------------------

print("\n[2] Checking factor columns...")

for col in continuous_factors + categorical_factors:

    if col not in df.columns:
        raise ValueError(
            f"Missing factor column: {col}"
        )

    print(f"  OK: {col}")


# ------------------------------------------------------------
# 5. REMOVE ROWS WITH MISSING FACTORS
# ------------------------------------------------------------

print("\n[3] Selecting complete observations...")

before = len(df)

complete_mask = (
    df[continuous_factors + categorical_factors]
    .notna()
    .all(axis=1)
)

df_complete = df.loc[
    complete_mask
].copy()

after = len(df_complete)

print(f"  Original observations: {before}")
print(f"  Complete observations: {after}")
print(f"  Excluded observations: {before - after}")


# ------------------------------------------------------------
# 6. LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\n[4] Label distribution in VIF dataset...")

print(
    df_complete["label"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 7. CONTINUOUS VARIABLES
# ------------------------------------------------------------

print("\n[5] Preparing continuous factors...")

X_continuous = df_complete[
    continuous_factors
].copy()

for col in continuous_factors:

    X_continuous[col] = pd.to_numeric(
        X_continuous[col],
        errors="coerce"
    )

print(
    f"  Continuous variables: "
    f"{len(continuous_factors)}"
)


# ------------------------------------------------------------
# 8. CATEGORICAL VARIABLES
# ------------------------------------------------------------

print("\n[6] Preparing categorical factors...")

X_categorical = df_complete[
    categorical_factors
].copy()

print("\n  Geomorphology categories:")
print(
    X_categorical["geomorphology"]
    .value_counts()
    .to_string()
)

print("\n  LULC categories:")
print(
    X_categorical["Level_I"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# 9. ONE-HOT ENCODING
# ------------------------------------------------------------

print("\n[7] One-hot encoding categorical factors...")

print(
    "  Using drop_first=True."
)

print(
    "  This prevents the dummy-variable trap."
)

X_categorical_encoded = pd.get_dummies(
    X_categorical,
    columns=categorical_factors,
    prefix=[
        "geomorphology",
        "lulc"
    ],
    drop_first=True,
    dtype=float
)

print(
    f"  Encoded categorical variables: "
    f"{X_categorical_encoded.shape[1]}"
)


# ------------------------------------------------------------
# 10. COMBINE PREDICTORS
# ------------------------------------------------------------

print("\n[8] Combining predictor variables...")

X = pd.concat(
    [
        X_continuous.reset_index(drop=True),
        X_categorical_encoded.reset_index(drop=True)
    ],
    axis=1
)

print(
    f"  Predictor matrix shape: {X.shape}"
)


# ------------------------------------------------------------
# 11. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n[9] Checking predictor missing values...")

missing_counts = X.isna().sum()

if missing_counts.sum() > 0:

    print(
        missing_counts[
            missing_counts > 0
        ].to_string()
    )

    raise ValueError(
        "Missing values remain in the VIF matrix."
    )

else:

    print(
        "  No missing predictor values."
    )


# ------------------------------------------------------------
# 12. CHECK CONSTANT VARIABLES
# ------------------------------------------------------------

print("\n[10] Checking constant variables...")

constant_columns = [
    col
    for col in X.columns
    if X[col].nunique() <= 1
]

if constant_columns:

    print(
        f"  Found {len(constant_columns)} constant variables."
    )

    for col in constant_columns:
        print(f"    {col}")

    X = X.drop(
        columns=constant_columns
    )

else:

    print(
        "  No constant variables."
    )


# ------------------------------------------------------------
# 13. CHECK MATRIX RANK
# ------------------------------------------------------------

print("\n[11] Checking matrix rank...")

X_values = X.to_numpy(
    dtype=float
)

rank = np.linalg.matrix_rank(
    X_values
)

n_predictors = X.shape[1]

print(
    f"  Matrix rank:       {rank}"
)

print(
    f"  Number predictors: {n_predictors}"
)

if rank < n_predictors:

    raise ValueError(
        "Matrix is still rank-deficient. "
        "VIF calculation stopped."
    )

else:

    print(
        "  Full column rank confirmed."
    )


# ------------------------------------------------------------
# 14. CALCULATE VIF
# ------------------------------------------------------------

print("\n[12] Calculating VIF...")

results = []

for i, column in enumerate(X.columns):

    vif_value = variance_inflation_factor(
        X_values,
        i
    )

    if (
        np.isfinite(vif_value)
        and vif_value > 0
    ):

        tolerance = 1.0 / vif_value

    else:

        tolerance = np.nan

    results.append(
        {
            "variable": column,
            "VIF": vif_value,
            "Tolerance": tolerance
        }
    )


vif_df = pd.DataFrame(
    results
)


# ------------------------------------------------------------
# 15. SORT RESULTS
# ------------------------------------------------------------

vif_df = vif_df.sort_values(
    by="VIF",
    ascending=False
).reset_index(drop=True)


# ------------------------------------------------------------
# 16. CLASSIFY VIF
# ------------------------------------------------------------

def classify_vif(vif):

    if pd.isna(vif):
        return "Undefined"

    if vif < 5:
        return "Low"

    elif vif < 10:
        return "Moderate"

    else:
        return "High"


vif_df["VIF_class"] = (
    vif_df["VIF"]
    .apply(classify_vif)
)


# ------------------------------------------------------------
# 17. CLASSIFY TOLERANCE
# ------------------------------------------------------------

def classify_tolerance(tolerance):

    if pd.isna(tolerance):
        return "Undefined"

    if tolerance >= 0.20:
        return "Acceptable"

    elif tolerance >= 0.10:
        return "Potential concern"

    else:
        return "Low tolerance"


vif_df["Tolerance_class"] = (
    vif_df["Tolerance"]
    .apply(classify_tolerance)
)


# ------------------------------------------------------------
# 18. PRINT RESULTS
# ------------------------------------------------------------

print("\n[13] CORRECTED VIF RESULTS")
print("=" * 75)

print(
    vif_df[
        [
            "variable",
            "VIF",
            "Tolerance",
            "VIF_class",
            "Tolerance_class"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# 19. SUMMARY
# ------------------------------------------------------------

print("\n[14] VIF SUMMARY")
print("=" * 75)

print(
    f"Minimum VIF: "
    f"{vif_df['VIF'].min():.4f}"
)

print(
    f"Maximum VIF: "
    f"{vif_df['VIF'].max():.4f}"
)

print(
    f"Mean VIF: "
    f"{vif_df['VIF'].mean():.4f}"
)

print(
    f"Median VIF: "
    f"{vif_df['VIF'].median():.4f}"
)


# ------------------------------------------------------------
# 20. MULTICOLLINEARITY FLAGS
# ------------------------------------------------------------

high_vif = vif_df[
    vif_df["VIF"] >= 10
]

moderate_vif = vif_df[
    (vif_df["VIF"] >= 5)
    &
    (vif_df["VIF"] < 10)
]

low_tolerance = vif_df[
    vif_df["Tolerance"] < 0.10
]


print("\n[15] MULTICOLLINEARITY FLAGS")
print("=" * 75)

print(
    f"VIF >= 10:        {len(high_vif)}"
)

print(
    f"VIF 5–10:         {len(moderate_vif)}"
)

print(
    f"Tolerance < 0.10: {len(low_tolerance)}"
)


if len(high_vif) > 0:

    print("\nVariables with VIF >= 10:")

    print(
        high_vif[
            [
                "variable",
                "VIF",
                "Tolerance"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

else:

    print(
        "\nNo variables have VIF >= 10."
    )


if len(moderate_vif) > 0:

    print("\nVariables with VIF between 5 and 10:")

    print(
        moderate_vif[
            [
                "variable",
                "VIF",
                "Tolerance"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

else:

    print(
        "\nNo variables have VIF between 5 and 10."
    )


# ------------------------------------------------------------
# 21. SAVE RESULTS
# ------------------------------------------------------------

print("\n[16] Saving VIF results...")

vif_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"  Saved: {OUTPUT_FILE}"
)


# ------------------------------------------------------------
# 22. FINAL
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 44 COMPLETED SUCCESSFULLY")
print("=" * 75)

print(
    "\nNo conditioning factor has been removed automatically."
)

print(
    "VIF results will be interpreted before factor selection."
)

print("=" * 75)