import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 46 — PREPARE FINAL MODELING DATASET")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "processed" / "ner_master_factors.csv"
OUTPUT_FILE = BASE_DIR / "processed" / "ner_modeling_dataset.csv"
MISSING_FILE = BASE_DIR / "processed" / "ner_modeling_missing_summary.csv"

# ---------------------------------------------------------
# 1. LOAD MASTER TABLE
# ---------------------------------------------------------
print("\n[1] Loading master factor table...")

df = pd.read_csv(INPUT_FILE)

print(f"  Rows: {len(df)}")
print(f"  Columns: {len(df.columns)}")

# ---------------------------------------------------------
# 2. DEFINE FINAL AVAILABLE FACTORS
# ---------------------------------------------------------
print("\n[2] Selecting final available factors...")

# Lithology is intentionally excluded for now because
# the official GSI/NGDR lithology dataset has not yet
# been incorporated into the project.

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

required_columns = [
    "lat",
    "lon",
    "label",
] + continuous_factors + categorical_factors

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from master table: {missing_columns}"
    )

model_df = df[required_columns].copy()

print("  Factors included:")
for col in continuous_factors + categorical_factors:
    print(f"    - {col}")

print("\n  Lithology:")
print("    - Excluded temporarily due to unavailable GSI/NGDR layer.")

# ---------------------------------------------------------
# 3. CHECK LABELS
# ---------------------------------------------------------
print("\n[3] Checking landslide/background labels...")

print(model_df["label"].value_counts(dropna=False).sort_index())

invalid_labels = model_df[
    ~model_df["label"].isin([0, 1])
]

if len(invalid_labels) > 0:
    raise ValueError(
        f"Found {len(invalid_labels)} records with invalid labels."
    )

# ---------------------------------------------------------
# 4. MISSING-VALUE AUDIT
# ---------------------------------------------------------
print("\n[4] Missing-value audit...")

missing_rows = []

for col in continuous_factors + categorical_factors:

    missing_count = int(model_df[col].isna().sum())
    valid_count = int(model_df[col].notna().sum())

    missing_rows.append({
        "factor": col,
        "total_rows": len(model_df),
        "valid_rows": valid_count,
        "missing_rows": missing_count,
        "missing_percent": round(
            missing_count / len(model_df) * 100, 4
        )
    })

    print(
        f"  {col}: "
        f"{valid_count} valid, "
        f"{missing_count} missing "
        f"({missing_count / len(model_df) * 100:.2f}%)"
    )

missing_summary = pd.DataFrame(missing_rows)

missing_summary.to_csv(
    MISSING_FILE,
    index=False
)

print(f"\n  Missing-value audit saved:")
print(f"  {MISSING_FILE}")

# ---------------------------------------------------------
# 5. CHECK DUPLICATES
# ---------------------------------------------------------
print("\n[5] Checking duplicate coordinates...")

duplicate_count = model_df.duplicated(
    subset=["lat", "lon"]
).sum()

print(f"  Duplicate coordinate rows: {duplicate_count}")

if duplicate_count > 0:
    raise ValueError(
        "Duplicate coordinates detected."
    )

# ---------------------------------------------------------
# 6. CHECK CONTINUOUS FACTORS
# ---------------------------------------------------------
print("\n[6] Continuous factor statistics...")

for col in continuous_factors:

    series = pd.to_numeric(
        model_df[col],
        errors="coerce"
    )

    print(f"\n  {col}")
    print(f"    Min    : {series.min()}")
    print(f"    Max    : {series.max()}")
    print(f"    Mean   : {series.mean()}")
    print(f"    Median : {series.median()}")
    print(f"    Std    : {series.std()}")

# ---------------------------------------------------------
# 7. CHECK CATEGORICAL FACTORS
# ---------------------------------------------------------
print("\n[7] Categorical factor classes...")

for col in categorical_factors:

    print(f"\n  {col}")

    counts = (
        model_df[col]
        .value_counts(dropna=False)
    )

    for category, count in counts.items():
        print(f"    {category}: {count}")

# ---------------------------------------------------------
# 8. COMPLETE CASE DATASET
# ---------------------------------------------------------
print("\n[8] Creating complete-case modeling dataset...")

factor_columns = continuous_factors + categorical_factors

complete_mask = model_df[factor_columns].notna().all(axis=1)

complete_df = model_df.loc[
    complete_mask
].copy()

excluded_df = model_df.loc[
    ~complete_mask
].copy()

print(f"  Original observations : {len(model_df)}")
print(f"  Complete observations: {len(complete_df)}")
print(f"  Excluded observations: {len(excluded_df)}")

print("\n  Complete-case labels:")
print(
    complete_df["label"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# 9. DO NOT NUMERICALLY ENCODE CATEGORIES
# ---------------------------------------------------------
print("\n[9] Preserving categorical factors...")

print("  Geomorphology remains categorical.")
print("  LUCC Level_I remains categorical.")
print("  No arbitrary ordinal ranking is applied.")

# ---------------------------------------------------------
# 10. SAVE FULL MODELING DATASET
# ---------------------------------------------------------
print("\n[10] Saving modeling dataset...")

complete_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"  Saved: {OUTPUT_FILE}")

# ---------------------------------------------------------
# 11. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 46 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFINAL MODELING DATASET")
print("-" * 70)

print(f"Total complete observations : {len(complete_df)}")
print(
    f"Background observations     : "
    f"{(complete_df['label'] == 0).sum()}"
)
print(
    f"Landslide observations      : "
    f"{(complete_df['label'] == 1).sum()}"
)

print("\nFactors included:")
for factor in continuous_factors + categorical_factors:
    print(f"  ✓ {factor}")

print("\nFactor excluded:")
print("  - lithology (temporary limitation)")

print("\nOutput files:")
print(f"  ✓ {OUTPUT_FILE}")
print(f"  ✓ {MISSING_FILE}")

print("=" * 70)
