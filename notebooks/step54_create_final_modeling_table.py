import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 54 — CREATE FINAL MODELING TABLE")
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
    / "ner_final_modeling_table.csv"
)

AUDIT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_final_modeling_audit.csv"
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
# 2. DEFINE FINAL PREDICTORS
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

target = "label"

all_model_columns = (
    ["lat", "lon", target]
    + continuous_factors
    + categorical_factors
)

print("\n[2] Final predictor structure")

print("\nContinuous factors:")
for factor in continuous_factors:
    print(f"  - {factor}")

print("\nCategorical factors:")
for factor in categorical_factors:
    print(f"  - {factor}")

print("\nTarget:")
print(f"  - {target}")

# ---------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ---------------------------------------------------------
print("\n[3] Checking required columns...")

missing_columns = [
    column
    for column in all_model_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("  ✓ All required columns found.")

# ---------------------------------------------------------
# 4. SELECT MODELING COLUMNS
# ---------------------------------------------------------
print("\n[4] Selecting modeling columns...")

model_df = df[
    all_model_columns
].copy()

# ---------------------------------------------------------
# 5. COMPLETE CASE FILTER
# ---------------------------------------------------------
print("\n[5] Selecting complete observations...")

factor_columns = (
    continuous_factors
    + categorical_factors
)

complete_mask = (
    model_df[factor_columns]
    .notna()
    .all(axis=1)
)

model_df = model_df.loc[
    complete_mask
].copy()

print(
    f"  Complete observations: "
    f"{len(model_df)}"
)

print(
    f"  Excluded observations: "
    f"{len(df) - len(model_df)}"
)

if len(model_df) != 991:
    raise ValueError(
        f"Expected 991 complete observations, "
        f"found {len(model_df)}"
    )

# ---------------------------------------------------------
# 6. CHECK TARGET
# ---------------------------------------------------------
print("\n[6] Checking target labels...")

label_counts = (
    model_df["label"]
    .value_counts()
    .sort_index()
)

print(label_counts)

if set(label_counts.index) != {0, 1}:
    raise ValueError(
        "Target must contain both 0 and 1 classes."
    )

# ---------------------------------------------------------
# 7. CHECK DUPLICATES
# ---------------------------------------------------------
print("\n[7] Checking duplicate coordinates...")

duplicate_count = (
    model_df
    .duplicated(
        subset=["lat", "lon"]
    )
    .sum()
)

print(
    f"  Duplicate coordinates: "
    f"{duplicate_count}"
)

if duplicate_count > 0:
    raise ValueError(
        "Duplicate coordinates detected."
    )

# ---------------------------------------------------------
# 8. CHECK CONTINUOUS FACTORS
# ---------------------------------------------------------
print("\n[8] Checking continuous predictors...")

for factor in continuous_factors:

    model_df[factor] = pd.to_numeric(
        model_df[factor],
        errors="coerce"
    )

    missing = model_df[factor].isna().sum()

    print(
        f"  {factor}: "
        f"{missing} missing"
    )

    if missing > 0:
        raise ValueError(
            f"Missing values in {factor}"
        )

# ---------------------------------------------------------
# 9. CHECK CATEGORICAL FACTORS
# ---------------------------------------------------------
print("\n[9] Checking categorical predictors...")

for factor in categorical_factors:

    missing = model_df[factor].isna().sum()

    print(
        f"  {factor}: "
        f"{missing} missing"
    )

    if missing > 0:
        raise ValueError(
            f"Missing values in {factor}"
        )

    print(
        f"  {factor} categories: "
        f"{model_df[factor].nunique()}"
    )

# ---------------------------------------------------------
# 10. CLASS BALANCE
# ---------------------------------------------------------
print("\n[10] Class balance")

background = (
    model_df["label"] == 0
).sum()

landslide = (
    model_df["label"] == 1
).sum()

total = len(model_df)

print(
    f"  Background : "
    f"{background} "
    f"({background / total * 100:.2f}%)"
)

print(
    f"  Landslide  : "
    f"{landslide} "
    f"({landslide / total * 100:.2f}%)"
)

# ---------------------------------------------------------
# 11. SAVE MODELING TABLE
# ---------------------------------------------------------
print("\n[11] Saving final modeling table...")

model_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"  Saved: {OUTPUT_FILE}"
)

# ---------------------------------------------------------
# 12. CREATE AUDIT
# ---------------------------------------------------------
print("\n[12] Creating modeling audit...")

audit_rows = []

for factor in continuous_factors:

    audit_rows.append({
        "factor": factor,
        "type": "continuous",
        "valid_rows": int(
            model_df[factor].notna().sum()
        ),
        "missing_rows": int(
            model_df[factor].isna().sum()
        ),
        "unique_values": int(
            model_df[factor].nunique()
        )
    })

for factor in categorical_factors:

    audit_rows.append({
        "factor": factor,
        "type": "categorical",
        "valid_rows": int(
            model_df[factor].notna().sum()
        ),
        "missing_rows": int(
            model_df[factor].isna().sum()
        ),
        "unique_values": int(
            model_df[factor].nunique()
        )
    })

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    AUDIT_FILE,
    index=False
)

print(
    f"  Saved: {AUDIT_FILE}"
)

# ---------------------------------------------------------
# 13. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 54 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFINAL MODELING DATASET")
print("-" * 70)

print(
    f"Observations: {len(model_df)}"
)

print(
    f"Background:   {background}"
)

print(
    f"Landslide:    {landslide}"
)

print("\nPredictors: 9")

for i, factor in enumerate(
    continuous_factors,
    start=1
):
    print(f"  {i}. {factor}")

print("  8. geomorph_origin")
print("  9. Level_I")

print("\nTarget:")
print("  label")

print("\nExcluded from current model:")
print("  lithology — unavailable GSI/NGDR dataset")

print("\nOriginal geomorphology:")
print(
    "  Retained in master table for traceability, "
    "but not used as a separate predictor."
)

print("\nOutputs:")
print(f"  {OUTPUT_FILE}")
print(f"  {AUDIT_FILE}")

print("=" * 70)