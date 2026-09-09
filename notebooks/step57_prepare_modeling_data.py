"""
STEP 57 — PREPARE MODELING DATA

Purpose:
- Load the spatially separated train/test dataset.
- Verify the 9 available predictors.
- Separate predictors and target.
- Identify numerical and categorical variables.
- Check missing values.
- Check class balance.
- Check predictor ranges.
- Verify spatial-group separation.
- Save clean train/test datasets for the modeling pipeline.

Important:
- Lithology is NOT included because the lithology dataset is currently unavailable.
- The final available model therefore uses 9 conditioning factors.
- No scaling, encoding, imputation, or feature selection is performed here.
- Those preprocessing operations will be performed inside the ML pipeline
  in the next modeling step to avoid data leakage.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"processed\ner_spatial_split_05deg.csv"

TRAIN_OUTPUT = r"processed\ner_train_modeling.csv"
TEST_OUTPUT = r"processed\ner_test_modeling.csv"
AUDIT_OUTPUT = r"processed\ner_modeling_data_audit.csv"


# ============================================================
# EXPECTED MODEL VARIABLES
# ============================================================

NUMERICAL_FEATURES = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density"
]

CATEGORICAL_FEATURES = [
    "geomorph_origin",
    "Level_I"
]

TARGET = "label"

ALL_FEATURES = (
    NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 57 — PREPARE MODELING DATA")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading spatially separated dataset...")
print("-" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"  Observations : {len(df)}")
print(f"  Columns      : {len(df.columns)}")


# ============================================================
# 2. VERIFY REQUIRED COLUMNS
# ============================================================

print("\n[2] Checking required modeling columns...")
print("-" * 70)

required_columns = (
    ["lat", "lon", TARGET, "split", "spatial_group"]
    + ALL_FEATURES
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        + "\n".join(f"  - {c}" for c in missing_columns)
    )

print("  All required columns are present.")


# ============================================================
# 3. CHECK SPLIT VALUES
# ============================================================

print("\n[3] Checking train/test split...")
print("-" * 70)

split_values = sorted(
    df["split"].dropna().unique().tolist()
)

print(f"  Split values found : {split_values}")

if set(split_values) != {"train", "test"}:
    raise ValueError(
        "Expected exactly two split values: train and test."
    )

train = df[df["split"] == "train"].copy()
test = df[df["split"] == "test"].copy()

print(f"  Training rows : {len(train)}")
print(f"  Testing rows  : {len(test)}")


# ============================================================
# 4. CHECK TARGET
# ============================================================

print("\n[4] Checking target variable...")
print("-" * 70)

target_values = sorted(
    df[TARGET].dropna().unique().tolist()
)

print(f"  Target values : {target_values}")

if set(target_values) != {0, 1}:
    raise ValueError(
        "Target must contain exactly two classes: 0 and 1."
    )

print("\n  Overall:")
print(
    f"    Background (0) : "
    f"{int((df[TARGET] == 0).sum())}"
)
print(
    f"    Landslide (1)  : "
    f"{int((df[TARGET] == 1).sum())}"
)

print("\n  Training:")
print(
    f"    Background (0) : "
    f"{int((train[TARGET] == 0).sum())}"
)
print(
    f"    Landslide (1)  : "
    f"{int((train[TARGET] == 1).sum())}"
)

print("\n  Testing:")
print(
    f"    Background (0) : "
    f"{int((test[TARGET] == 0).sum())}"
)
print(
    f"    Landslide (1)  : "
    f"{int((test[TARGET] == 1).sum())}"
)


# ============================================================
# 5. CHECK SPATIAL GROUP SEPARATION
# ============================================================

print("\n[5] Verifying spatial-group separation...")
print("-" * 70)

train_groups = set(train["spatial_group"].dropna())
test_groups = set(test["spatial_group"].dropna())

overlap = train_groups.intersection(test_groups)

print(f"  Training spatial groups : {len(train_groups)}")
print(f"  Testing spatial groups  : {len(test_groups)}")
print(f"  Overlapping groups      : {len(overlap)}")

if len(overlap) != 0:
    raise RuntimeError(
        "Spatial leakage detected: train/test groups overlap."
    )

print("  Spatial separation PASSED.")


# ============================================================
# 6. CHECK DUPLICATE COORDINATES
# ============================================================

print("\n[6] Checking duplicate coordinates...")
print("-" * 70)

duplicate_all = df.duplicated(
    subset=["lat", "lon"]
).sum()

duplicate_train = train.duplicated(
    subset=["lat", "lon"]
).sum()

duplicate_test = test.duplicated(
    subset=["lat", "lon"]
).sum()

print(f"  Overall duplicates : {duplicate_all}")
print(f"  Train duplicates   : {duplicate_train}")
print(f"  Test duplicates    : {duplicate_test}")

if (
    duplicate_all != 0
    or duplicate_train != 0
    or duplicate_test != 0
):
    raise RuntimeError(
        "Duplicate coordinates detected."
    )

print("  Coordinate uniqueness PASSED.")


# ============================================================
# 7. CHECK NUMERICAL FEATURES
# ============================================================

print("\n[7] Checking numerical predictors...")
print("-" * 70)

numeric_audit = []

for feature in NUMERICAL_FEATURES:

    overall_missing = int(df[feature].isna().sum())
    train_missing = int(train[feature].isna().sum())
    test_missing = int(test[feature].isna().sum())

    overall_min = df[feature].min()
    overall_max = df[feature].max()
    overall_mean = df[feature].mean()
    overall_std = df[feature].std()

    print(f"\n  {feature}")
    print(f"    Missing overall : {overall_missing}")
    print(f"    Missing train   : {train_missing}")
    print(f"    Missing test    : {test_missing}")
    print(f"    Min             : {overall_min}")
    print(f"    Max             : {overall_max}")
    print(f"    Mean            : {overall_mean}")
    print(f"    Std             : {overall_std}")

    numeric_audit.append({
        "variable": feature,
        "type": "numeric",
        "missing_overall": overall_missing,
        "missing_train": train_missing,
        "missing_test": test_missing,
        "min": overall_min,
        "max": overall_max,
        "mean": overall_mean,
        "std": overall_std
    })


# ============================================================
# 8. CHECK NUMERICAL VALUES FOR INF/NON-FINITE
# ============================================================

print("\n[8] Checking numerical values for non-finite values...")
print("-" * 70)

for feature in NUMERICAL_FEATURES:

    values = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

    n_nonfinite = int(
        (~np.isfinite(values.dropna())).sum()
    )

    print(
        f"  {feature:<25} "
        f"non-finite values: {n_nonfinite}"
    )

    if n_nonfinite != 0:
        raise ValueError(
            f"Non-finite values found in {feature}."
        )

print("  Numerical finiteness check PASSED.")


# ============================================================
# 9. CHECK CATEGORICAL FEATURES
# ============================================================

print("\n[9] Checking categorical predictors...")
print("-" * 70)

categorical_audit = []

for feature in CATEGORICAL_FEATURES:

    overall_missing = int(df[feature].isna().sum())
    train_missing = int(train[feature].isna().sum())
    test_missing = int(test[feature].isna().sum())

    categories = (
        df[feature]
        .dropna()
        .astype(str)
        .value_counts()
    )

    print(f"\n  {feature}")
    print(f"    Missing overall : {overall_missing}")
    print(f"    Missing train   : {train_missing}")
    print(f"    Missing test    : {test_missing}")
    print(f"    Number classes  : {len(categories)}")

    print("    Classes:")

    for category, count in categories.items():
        print(
            f"      {category} : {count}"
        )

    for category, count in categories.items():
        categorical_audit.append({
            "variable": feature,
            "type": "categorical",
            "category": category,
            "count": int(count)
        })


# ============================================================
# 10. CHECK CATEGORICAL TRAIN/TEST COVERAGE
# ============================================================

print("\n[10] Checking categorical train/test coverage...")
print("-" * 70)

for feature in CATEGORICAL_FEATURES:

    train_categories = set(
        train[feature]
        .dropna()
        .astype(str)
        .unique()
    )

    test_categories = set(
        test[feature]
        .dropna()
        .astype(str)
        .unique()
    )

    unseen_in_train = test_categories - train_categories

    print(f"\n  {feature}")
    print(
        f"    Train categories : "
        f"{len(train_categories)}"
    )
    print(
        f"    Test categories  : "
        f"{len(test_categories)}"
    )

    if unseen_in_train:
        print(
            "    Categories present in test but absent "
            "from train:"
        )

        for category in sorted(unseen_in_train):
            print(f"      - {category}")

        print(
            "    NOTE: These will be handled later using "
            "OneHotEncoder(handle_unknown='ignore')."
        )
    else:
        print(
            "    All test categories are represented "
            "in training data."
        )


# ============================================================
# 11. MISSING-DATA SUMMARY
# ============================================================

print("\n[11] Overall missing-data summary...")
print("-" * 70)

model_columns = ALL_FEATURES + [TARGET]

missing_summary = df[model_columns].isna().sum()

for feature, count in missing_summary.items():

    print(
        f"  {feature:<25} "
        f"missing: {int(count)}"
    )


# ============================================================
# 12. VERIFY FINAL MODELING FEATURES
# ============================================================

print("\n[12] Verifying final 9-factor predictor set...")
print("-" * 70)

print("\n  Numerical:")
for feature in NUMERICAL_FEATURES:
    print(f"    - {feature}")

print("\n  Categorical:")
for feature in CATEGORICAL_FEATURES:
    print(f"    - {feature}")

print("\n  Target:")
print(f"    - {TARGET}")

print("\n  Total predictors:", len(ALL_FEATURES))

if len(ALL_FEATURES) != 9:
    raise RuntimeError(
        "Expected exactly 9 available predictors."
    )


# ============================================================
# 13. LITHOLOGY STATUS
# ============================================================

print("\n[13] Lithology status...")
print("-" * 70)

print(
    "  Lithology is NOT included in the current modeling table."
)

print(
    "  Reason: the required GSI/NGDR lithology dataset "
    "has not yet been obtained."
)

print(
    "  This is a documented data-availability limitation, "
    "not a claim that lithology is unimportant."
)


# ============================================================
# 14. PREPROCESSING STATUS
# ============================================================

print("\n[14] Preprocessing status...")
print("-" * 70)

print(
    "  No imputation performed."
)

print(
    "  No scaling performed."
)

print(
    "  No categorical encoding performed."
)

print(
    "  No feature selection performed."
)

print(
    "  These operations will be performed inside the "
    "modeling pipeline using training data only."
)


# ============================================================
# 15. CREATE CLEAN TRAIN/TEST TABLES
# ============================================================

print("\n[15] Creating clean train/test modeling tables...")
print("-" * 70)

# Keep coordinates and spatial group for auditing.
# The ML pipeline itself will only use ALL_FEATURES.

metadata_columns = [
    "lat",
    "lon",
    "spatial_group"
]

train_output_columns = (
    metadata_columns
    + ALL_FEATURES
    + [TARGET]
)

test_output_columns = (
    metadata_columns
    + ALL_FEATURES
    + [TARGET]
)

train_model = train[
    train_output_columns
].copy()

test_model = test[
    test_output_columns
].copy()

print(
    f"  Training table shape : "
    f"{train_model.shape}"
)

print(
    f"  Testing table shape  : "
    f"{test_model.shape}"
)


# ============================================================
# 16. SAVE TRAIN/TEST TABLES
# ============================================================

print("\n[16] Saving modeling tables...")
print("-" * 70)

train_model.to_csv(
    TRAIN_OUTPUT,
    index=False
)

test_model.to_csv(
    TEST_OUTPUT,
    index=False
)

print(f"  Saved: {TRAIN_OUTPUT}")
print(f"  Saved: {TEST_OUTPUT}")


# ============================================================
# 17. CREATE AUDIT REPORT
# ============================================================

print("\n[17] Creating modeling-data audit...")
print("-" * 70)

audit_rows = [
    {
        "item": "total_observations",
        "value": len(df)
    },
    {
        "item": "training_observations",
        "value": len(train)
    },
    {
        "item": "testing_observations",
        "value": len(test)
    },
    {
        "item": "total_predictors",
        "value": len(ALL_FEATURES)
    },
    {
        "item": "numerical_predictors",
        "value": len(NUMERICAL_FEATURES)
    },
    {
        "item": "categorical_predictors",
        "value": len(CATEGORICAL_FEATURES)
    },
    {
        "item": "overall_background",
        "value": int((df[TARGET] == 0).sum())
    },
    {
        "item": "overall_landslide",
        "value": int((df[TARGET] == 1).sum())
    },
    {
        "item": "train_background",
        "value": int((train[TARGET] == 0).sum())
    },
    {
        "item": "train_landslide",
        "value": int((train[TARGET] == 1).sum())
    },
    {
        "item": "test_background",
        "value": int((test[TARGET] == 0).sum())
    },
    {
        "item": "test_landslide",
        "value": int((test[TARGET] == 1).sum())
    },
    {
        "item": "spatial_group_overlap",
        "value": len(overlap)
    },
    {
        "item": "duplicate_coordinates",
        "value": duplicate_all
    },
    {
        "item": "lithology_available",
        "value": False
    }
]

# Add missing-data information
for feature in ALL_FEATURES:

    audit_rows.append({
        "item": f"missing_{feature}",
        "value": int(df[feature].isna().sum())
    })

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    AUDIT_OUTPUT,
    index=False
)

print(f"  Saved: {AUDIT_OUTPUT}")


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

print("\n[18] Final validation...")
print("-" * 70)

# Ensure train/test contain both classes
if set(train_model[TARGET].unique()) != {0, 1}:
    raise RuntimeError(
        "Training data does not contain both target classes."
    )

if set(test_model[TARGET].unique()) != {0, 1}:
    raise RuntimeError(
        "Testing data does not contain both target classes."
    )

# Ensure no spatial overlap
if (
    set(train_model["spatial_group"])
    .intersection(set(test_model["spatial_group"]))
):
    raise RuntimeError(
        "Spatial groups overlap between train and test."
    )

# Ensure no duplicate coordinates
if train_model.duplicated(
    subset=["lat", "lon"]
).any():
    raise RuntimeError(
        "Duplicate training coordinates detected."
    )

if test_model.duplicated(
    subset=["lat", "lon"]
).any():
    raise RuntimeError(
        "Duplicate testing coordinates detected."
    )

print("  Target classes             : PASSED")
print("  Spatial separation         : PASSED")
print("  Coordinate uniqueness      : PASSED")
print("  Predictor configuration    : PASSED")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 57 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal modeling configuration:")
print("  Available conditioning factors : 9")
print("  Numerical factors              : 7")
print("  Categorical factors            : 2")
print("  Lithology                      : unavailable")
print("  Target                         : label")

print("\nDataset:")
print(f"  Train : {len(train_model)}")
print(f"  Test  : {len(test_model)}")

print("\nClass balance:")
print(
    f"  Train landslide proportion : "
    f"{train_model[TARGET].mean():.4f}"
)

print(
    f"  Test landslide proportion  : "
    f"{test_model[TARGET].mean():.4f}"
)

print("\nOutputs:")
print(f"  {TRAIN_OUTPUT}")
print(f"  {TEST_OUTPUT}")
print(f"  {AUDIT_OUTPUT}")

print("\nNext stage:")
print(
    "  Build leakage-safe Logistic Regression "
    "and Random Forest pipelines."
)

print("=" * 70)