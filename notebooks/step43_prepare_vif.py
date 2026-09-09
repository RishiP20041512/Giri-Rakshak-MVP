import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 43 — VIF PREPARATION & MISSING-DATA AUDIT
# Giri Rakshak Project
# ============================================================

print("=" * 75)
print("STEP 43 — VIF PREPARATION & MISSING-DATA AUDIT")
print("=" * 75)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"

INPUT_FILE = PROCESSED_DIR / "ner_master_factors.csv"

VIF_READY_FILE = PROCESSED_DIR / "ner_vif_ready.csv"
MISSING_FILE = PROCESSED_DIR / "ner_vif_missing_audit.csv"
CATEGORICAL_FILE = PROCESSED_DIR / "ner_vif_categorical_summary.csv"


# ------------------------------------------------------------
# 2. LOAD MASTER TABLE
# ------------------------------------------------------------

print("\n[1] Loading master factor table...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Master factor table not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")


if len(df) != 1071:
    raise ValueError(
        f"Expected 1071 rows, found {len(df)}"
    )


# ------------------------------------------------------------
# 3. DEFINE MODEL FACTORS
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


all_factors = continuous_factors + categorical_factors


# ------------------------------------------------------------
# 4. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

print("\n[2] Checking required factor columns...")

missing_columns = [
    col for col in all_factors
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing factor columns: {missing_columns}"
    )

for col in all_factors:
    print(f"  OK: {col}")


# ------------------------------------------------------------
# 5. MISSING-DATA AUDIT
# ------------------------------------------------------------

print("\n[3] Missing-data audit...")

missing_records = []

for col in all_factors:

    missing = int(df[col].isna().sum())
    valid = int(df[col].notna().sum())
    percentage = (missing / len(df)) * 100

    missing_records.append(
        {
            "factor": col,
            "total_rows": len(df),
            "valid_rows": valid,
            "missing_rows": missing,
            "missing_percent": percentage,
        }
    )

    print(
        f"  {col:<25} "
        f"valid={valid:>4}  "
        f"missing={missing:>4}  "
        f"({percentage:.2f}%)"
    )


missing_df = pd.DataFrame(missing_records)

missing_df.to_csv(
    MISSING_FILE,
    index=False
)

print(f"\n  Saved: {MISSING_FILE}")


# ------------------------------------------------------------
# 6. MISSINGNESS BY LABEL
# ------------------------------------------------------------

print("\n[4] Missingness by landslide/background class...")

for col in all_factors:

    print(f"\n  {col}")

    table = (
        df.groupby("label")[col]
        .apply(lambda x: x.isna().sum())
    )

    for label, count in table.items():
        label_name = (
            "Background (0)"
            if label == 0
            else "Landslide (1)"
        )

        print(
            f"    {label_name:<20}: {int(count)}"
        )


# ------------------------------------------------------------
# 7. STATE INFORMATION
# ------------------------------------------------------------

print("\n[5] Checking state information...")

if "region" in df.columns:

    print(
        "\n  Region counts:"
    )

    print(
        df["region"]
        .value_counts(dropna=False)
        .to_string()
    )

else:
    print("  Region column not available.")


# ------------------------------------------------------------
# 8. CONTINUOUS FACTOR CHECK
# ------------------------------------------------------------

print("\n[6] Checking continuous factors...")

for col in continuous_factors:

    series = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    print(f"\n  {col}")

    print(f"    Valid:   {series.notna().sum()}")
    print(f"    Missing: {series.isna().sum()}")

    if series.notna().sum() > 0:

        print(
            f"    Min:     {series.min():.6f}"
        )

        print(
            f"    Max:     {series.max():.6f}"
        )

        print(
            f"    Mean:    {series.mean():.6f}"
        )

        print(
            f"    Median:  {series.median():.6f}"
        )

        print(
            f"    Std:     {series.std():.6f}"
        )


# ------------------------------------------------------------
# 9. CATEGORICAL FACTOR CHECK
# ------------------------------------------------------------

print("\n[7] Checking categorical factors...")

categorical_records = []

for col in categorical_factors:

    print(f"\n  {col}")

    counts = (
        df[col]
        .value_counts(dropna=False)
    )

    print(
        counts.to_string()
    )

    for category, count in counts.items():

        category_name = (
            "MISSING"
            if pd.isna(category)
            else str(category)
        )

        categorical_records.append(
            {
                "factor": col,
                "category": category_name,
                "count": int(count),
            }
        )


categorical_df = pd.DataFrame(
    categorical_records
)

categorical_df.to_csv(
    CATEGORICAL_FILE,
    index=False
)

print(
    f"\n  Saved: {CATEGORICAL_FILE}"
)


# ------------------------------------------------------------
# 10. ONE-HOT ENCODING
# ------------------------------------------------------------

print("\n[8] Preparing categorical encoding...")

print(
    "  Geomorphology and LULC will be one-hot encoded."
)

print(
    "  Missing categories are retained as missing."
)


# ------------------------------------------------------------
# IMPORTANT:
# We do NOT impute missing values here.
#
# For VIF, complete cases will be used only for the
# calculation. The original master table remains unchanged.
# ------------------------------------------------------------

categorical_encoded = pd.get_dummies(
    df[categorical_factors],
    columns=categorical_factors,
    prefix=["geomorphology", "lulc"],
    dummy_na=True,
    dtype=float
)


# ------------------------------------------------------------
# 11. PREPARE CONTINUOUS VARIABLES
# ------------------------------------------------------------

continuous_df = df[continuous_factors].copy()

for col in continuous_factors:

    continuous_df[col] = pd.to_numeric(
        continuous_df[col],
        errors="coerce"
    )


# ------------------------------------------------------------
# 12. COMBINE VIF VARIABLES
# ------------------------------------------------------------

vif_variables = pd.concat(
    [
        continuous_df,
        categorical_encoded
    ],
    axis=1
)


# ------------------------------------------------------------
# 13. REMOVE CONSTANT COLUMNS
# ------------------------------------------------------------

print("\n[9] Checking for constant encoded variables...")

constant_columns = []

for col in vif_variables.columns:

    if vif_variables[col].nunique(dropna=False) <= 1:
        constant_columns.append(col)


if constant_columns:

    print(
        f"  Constant columns found: {len(constant_columns)}"
    )

    for col in constant_columns:
        print(f"    Removing: {col}")

    vif_variables = vif_variables.drop(
        columns=constant_columns
    )

else:

    print("  No constant columns found.")


# ------------------------------------------------------------
# 14. COMPLETE-CASE DATA FOR VIF
# ------------------------------------------------------------

print("\n[10] Preparing complete cases for VIF...")

complete_case_mask = (
    vif_variables.notna().all(axis=1)
)

vif_complete = vif_variables.loc[
    complete_case_mask
].copy()

print(
    f"  Original rows:       {len(df)}"
)

print(
    f"  Complete-case rows:   {len(vif_complete)}"
)

print(
    f"  Excluded rows:        "
    f"{len(df) - len(vif_complete)}"
)


# ------------------------------------------------------------
# 15. LABEL DISTRIBUTION IN COMPLETE CASES
# ------------------------------------------------------------

print("\n[11] Label distribution among complete cases...")

complete_labels = df.loc[
    complete_case_mask,
    "label"
]

print(
    complete_labels
    .value_counts()
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 16. CHECK FOR ZERO-VARIANCE AFTER COMPLETE CASE FILTER
# ------------------------------------------------------------

print("\n[12] Checking zero-variance variables...")

zero_variance = []

for col in vif_complete.columns:

    if vif_complete[col].nunique() <= 1:
        zero_variance.append(col)


if zero_variance:

    print(
        f"  Zero-variance variables: {len(zero_variance)}"
    )

    for col in zero_variance:
        print(f"    {col}")

else:

    print("  No zero-variance variables.")


# ------------------------------------------------------------
# 17. SAVE VIF-READY DATA
# ------------------------------------------------------------

print("\n[13] Saving VIF-ready dataset...")

# Add original identifiers for traceability
vif_output = pd.concat(
    [
        df.loc[
            complete_case_mask,
            [
                "lat",
                "lon",
                "label"
            ]
        ].reset_index(drop=True),

        vif_complete.reset_index(drop=True)
    ],
    axis=1
)

vif_output.to_csv(
    VIF_READY_FILE,
    index=False
)

print(
    f"  Saved: {VIF_READY_FILE}"
)


# ------------------------------------------------------------
# 18. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("STEP 43 SUMMARY")
print("=" * 75)

print(
    f"Original observations:       {len(df)}"
)

print(
    f"Complete VIF observations:   {len(vif_complete)}"
)

print(
    f"Excluded due to missingness: "
    f"{len(df) - len(vif_complete)}"
)

print(
    f"Continuous factors:          "
    f"{len(continuous_factors)}"
)

print(
    f"Categorical factors:         "
    f"{len(categorical_factors)}"
)

print(
    f"VIF variables after encoding:"
    f" {len(vif_complete.columns)}"
)

print("\nOutputs:")

print(
    f"  {VIF_READY_FILE}"
)

print(
    f"  {MISSING_FILE}"
)

print(
    f"  {CATEGORICAL_FILE}"
)

print("\nImportant methodological notes:")

print(
    "  1. No missing values were imputed."
)

print(
    "  2. Geomorphology was one-hot encoded."
)

print(
    "  3. LULC Level_I was one-hot encoded."
)

print(
    "  4. Complete cases are used only for VIF preparation."
)

print(
    "  5. The original master dataset remains unchanged."
)

print(
    "  6. Lithology is not included because it is currently unavailable."
)

print("=" * 75)
print("STEP 43 COMPLETED")
print("=" * 75)