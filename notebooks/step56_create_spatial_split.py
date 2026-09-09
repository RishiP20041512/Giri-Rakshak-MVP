"""
STEP 56 — CREATE SPATIALLY GROUPED TRAIN/TEST SPLIT

Purpose:
- Create 0.50-degree spatial blocks.
- Keep all observations from the same spatial block
  entirely within either train or test.
- Search multiple grouped splits.
- Select a split with approximately 20% test observations
  and a class distribution close to the overall dataset.
- Verify zero spatial-group overlap.
- Save the final split and audit information.

Input:
    processed/ner_final_modeling_table.csv

Output:
    processed/ner_spatial_split_05deg.csv
    processed/ner_spatial_split_audit.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = r"processed\ner_final_modeling_table.csv"

OUTPUT_SPLIT = r"processed\ner_spatial_split_05deg.csv"
OUTPUT_AUDIT = r"processed\ner_spatial_split_audit.csv"

GRID_SIZE = 0.50
TEST_SIZE_TARGET = 0.20

# Number of candidate grouped splits to examine
N_SPLITS = 100

# Reproducible random seed
RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 56 — CREATE SPATIALLY GROUPED TRAIN/TEST SPLIT")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading final modeling table...")
print("-" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print(f"  Observations loaded: {len(df)}")
print(f"  Columns: {len(df.columns)}")


# ============================================================
# 2. BASIC VALIDATION
# ============================================================

print("\n[2] Validating required columns...")
print("-" * 70)

required_columns = [
    "lat",
    "lon",
    "label"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Latitude, longitude, or label contains missing values."
    )

print("  Required columns present.")
print("  No missing coordinates/labels.")


# ============================================================
# 3. CREATE 0.50-DEGREE SPATIAL GROUPS
# ============================================================

print("\n[3] Creating 0.50-degree spatial blocks...")
print("-" * 70)

# Floor coordinates into fixed geographic grid cells.
#
# Example:
# longitude 92.63 -> block 185
# latitude 27.45  -> block 54
#
# Every point falling inside the same cell receives
# the same spatial group ID.

df["lat_block"] = np.floor(df["lat"] / GRID_SIZE).astype(int)
df["lon_block"] = np.floor(df["lon"] / GRID_SIZE).astype(int)

df["spatial_group"] = (
    df["lat_block"].astype(str)
    + "_"
    + df["lon_block"].astype(str)
)

n_groups = df["spatial_group"].nunique()

print(f"  Grid size              : {GRID_SIZE:.2f} degrees")
print(f"  Spatial groups         : {n_groups}")
print(f"  Largest group          : {df['spatial_group'].value_counts().max()}")
print(
    f"  Median group size      : "
    f"{df['spatial_group'].value_counts().median():.1f}"
)


# ============================================================
# 4. OVERALL CLASS DISTRIBUTION
# ============================================================

print("\n[4] Overall class distribution...")
print("-" * 70)

overall_counts = df["label"].value_counts().sort_index()

n_total = len(df)

overall_background = int(overall_counts.get(0, 0))
overall_landslide = int(overall_counts.get(1, 0))

overall_ls_ratio = overall_landslide / n_total

print(f"  Total observations     : {n_total}")
print(f"  Background (0)         : {overall_background}")
print(f"  Landslide (1)          : {overall_landslide}")
print(f"  Landslide proportion   : {overall_ls_ratio:.4f}")


# ============================================================
# 5. GENERATE CANDIDATE GROUPED SPLITS
# ============================================================

print("\n[5] Searching candidate spatially grouped splits...")
print("-" * 70)

groups = df["spatial_group"].values
y = df["label"].values

gss = GroupShuffleSplit(
    n_splits=N_SPLITS,
    test_size=TEST_SIZE_TARGET,
    random_state=RANDOM_STATE
)

candidates = []

for split_number, (train_idx, test_idx) in enumerate(
    gss.split(df, y, groups),
    start=1
):

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    train_groups = set(train_df["spatial_group"])
    test_groups = set(test_df["spatial_group"])

    # --------------------------------------------------------
    # Verify that no spatial group occurs in both sets
    # --------------------------------------------------------

    overlap = train_groups.intersection(test_groups)

    if len(overlap) != 0:
        raise RuntimeError(
            "Spatial leakage detected: "
            f"{len(overlap)} groups overlap."
        )

    # --------------------------------------------------------
    # Test size
    # --------------------------------------------------------

    test_fraction = len(test_df) / n_total

    # --------------------------------------------------------
    # Test class proportion
    # --------------------------------------------------------

    test_ls_ratio = test_df["label"].mean()

    # --------------------------------------------------------
    # Train class proportion
    # --------------------------------------------------------

    train_ls_ratio = train_df["label"].mean()

    # --------------------------------------------------------
    # Difference from desired test size
    # --------------------------------------------------------

    size_error = abs(
        test_fraction - TEST_SIZE_TARGET
    )

    # --------------------------------------------------------
    # Difference from overall landslide proportion
    # --------------------------------------------------------

    class_error = abs(
        test_ls_ratio - overall_ls_ratio
    )

    # --------------------------------------------------------
    # Combined score
    #
    # We give somewhat greater importance to class balance
    # while still requiring approximately 20% test data.
    # --------------------------------------------------------

    score = (
        class_error * 2.0
        + size_error
    )

    candidates.append({
        "split_number": split_number,
        "train_n": len(train_df),
        "test_n": len(test_df),
        "train_fraction": len(train_df) / n_total,
        "test_fraction": test_fraction,
        "train_ls_ratio": train_ls_ratio,
        "test_ls_ratio": test_ls_ratio,
        "size_error": size_error,
        "class_error": class_error,
        "score": score
    })


candidates_df = pd.DataFrame(candidates)

best_row = candidates_df.loc[
    candidates_df["score"].idxmin()
]

best_split_number = int(
    best_row["split_number"]
)

print(f"  Candidate splits tested : {N_SPLITS}")
print(f"  Best split              : {best_split_number}")
print(f"  Best score              : {best_row['score']:.6f}")


# ============================================================
# 6. RECREATE THE SELECTED SPLIT
# ============================================================

print("\n[6] Creating final selected split...")
print("-" * 70)

# Recreate exactly the same sequence of splits.
gss_final = GroupShuffleSplit(
    n_splits=N_SPLITS,
    test_size=TEST_SIZE_TARGET,
    random_state=RANDOM_STATE
)

selected_train_idx = None
selected_test_idx = None

for split_number, (train_idx, test_idx) in enumerate(
    gss_final.split(df, y, groups),
    start=1
):

    if split_number == best_split_number:
        selected_train_idx = train_idx
        selected_test_idx = test_idx
        break

if selected_train_idx is None:
    raise RuntimeError(
        "Could not recreate selected split."
    )

train_mask = np.zeros(len(df), dtype=bool)
test_mask = np.zeros(len(df), dtype=bool)

train_mask[selected_train_idx] = True
test_mask[selected_test_idx] = True

df["split"] = "train"
df.loc[test_mask, "split"] = "test"


# ============================================================
# 7. REMOVE TEMPORARY GRID COLUMNS
# ============================================================

# Keep spatial_group because it is useful for audit and
# leakage verification.

df = df.drop(
    columns=["lat_block", "lon_block"]
)


# ============================================================
# 8. FINAL SPLIT STATISTICS
# ============================================================

print("\n[7] Final split statistics...")
print("-" * 70)

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

print(f"  Training observations : {len(train_df)}")
print(f"  Testing observations  : {len(test_df)}")

print(
    f"  Training fraction     : "
    f"{len(train_df) / len(df):.4f}"
)

print(
    f"  Testing fraction      : "
    f"{len(test_df) / len(df):.4f}"
)


# ============================================================
# 9. CLASS DISTRIBUTION
# ============================================================

print("\n[8] Class distribution by split...")
print("-" * 70)

train_counts = (
    train_df["label"]
    .value_counts()
    .sort_index()
)

test_counts = (
    test_df["label"]
    .value_counts()
    .sort_index()
)

print("\n  TRAIN")
print(f"    Background (0) : {int(train_counts.get(0, 0))}")
print(f"    Landslide (1)  : {int(train_counts.get(1, 0))}")
print(f"    LS proportion  : {train_df['label'].mean():.4f}")

print("\n  TEST")
print(f"    Background (0) : {int(test_counts.get(0, 0))}")
print(f"    Landslide (1)  : {int(test_counts.get(1, 0))}")
print(f"    LS proportion  : {test_df['label'].mean():.4f}")

print("\n  OVERALL")
print(f"    Background (0) : {overall_background}")
print(f"    Landslide (1)  : {overall_landslide}")
print(f"    LS proportion  : {overall_ls_ratio:.4f}")


# ============================================================
# 10. SPATIAL LEAKAGE CHECK
# ============================================================

print("\n[9] Checking spatial leakage...")
print("-" * 70)

train_groups = set(
    train_df["spatial_group"]
)

test_groups = set(
    test_df["spatial_group"]
)

overlap_groups = train_groups.intersection(
    test_groups
)

print(
    f"  Training spatial groups : "
    f"{len(train_groups)}"
)

print(
    f"  Testing spatial groups  : "
    f"{len(test_groups)}"
)

print(
    f"  Overlapping groups      : "
    f"{len(overlap_groups)}"
)

if len(overlap_groups) != 0:
    raise RuntimeError(
        "CRITICAL ERROR: spatial groups overlap!"
    )

print("  Spatial leakage check PASSED.")


# ============================================================
# 11. COORDINATE DUPLICATE CHECK
# ============================================================

print("\n[10] Checking coordinate duplication...")
print("-" * 70)

duplicate_coordinates = df.duplicated(
    subset=["lat", "lon"]
).sum()

print(
    f"  Duplicate coordinates : "
    f"{duplicate_coordinates}"
)

if duplicate_coordinates != 0:
    raise RuntimeError(
        "Duplicate coordinates detected."
    )

print("  Coordinate uniqueness check PASSED.")


# ============================================================
# 12. VERIFY BOTH CLASSES EXIST IN BOTH SETS
# ============================================================

print("\n[11] Checking class availability...")
print("-" * 70)

train_classes = set(train_df["label"].unique())
test_classes = set(test_df["label"].unique())

print(f"  Training classes : {sorted(train_classes)}")
print(f"  Testing classes  : {sorted(test_classes)}")

if train_classes != {0, 1}:
    raise RuntimeError(
        "Training set does not contain both classes."
    )

if test_classes != {0, 1}:
    raise RuntimeError(
        "Testing set does not contain both classes."
    )

print("  Both classes present in train and test.")


# ============================================================
# 13. SAVE FINAL SPLIT
# ============================================================

print("\n[12] Saving final spatial split...")
print("-" * 70)

df.to_csv(
    OUTPUT_SPLIT,
    index=False
)

print(f"  Saved: {OUTPUT_SPLIT}")


# ============================================================
# 14. CREATE AUDIT TABLE
# ============================================================

print("\n[13] Creating audit report...")
print("-" * 70)

audit_rows = [
    {
        "parameter": "input_observations",
        "value": len(df)
    },
    {
        "parameter": "grid_size_degrees",
        "value": GRID_SIZE
    },
    {
        "parameter": "spatial_groups_total",
        "value": df["spatial_group"].nunique()
    },
    {
        "parameter": "train_observations",
        "value": len(train_df)
    },
    {
        "parameter": "test_observations",
        "value": len(test_df)
    },
    {
        "parameter": "train_fraction",
        "value": len(train_df) / len(df)
    },
    {
        "parameter": "test_fraction",
        "value": len(test_df) / len(df)
    },
    {
        "parameter": "overall_landslide_fraction",
        "value": overall_ls_ratio
    },
    {
        "parameter": "train_landslide_fraction",
        "value": train_df["label"].mean()
    },
    {
        "parameter": "test_landslide_fraction",
        "value": test_df["label"].mean()
    },
    {
        "parameter": "train_spatial_groups",
        "value": len(train_groups)
    },
    {
        "parameter": "test_spatial_groups",
        "value": len(test_groups)
    },
    {
        "parameter": "overlapping_spatial_groups",
        "value": len(overlap_groups)
    },
    {
        "parameter": "duplicate_coordinates",
        "value": duplicate_coordinates
    },
    {
        "parameter": "candidate_splits",
        "value": N_SPLITS
    },
    {
        "parameter": "selected_split_number",
        "value": best_split_number
    }
]

audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    OUTPUT_AUDIT,
    index=False
)

print(f"  Saved: {OUTPUT_AUDIT}")


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 56 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nSpatial validation design:")
print(f"  Spatial block size : {GRID_SIZE:.2f} degrees")
print(f"  Train observations : {len(train_df)}")
print(f"  Test observations  : {len(test_df)}")

print("\nClass balance:")
print(
    f"  Overall LS ratio : "
    f"{overall_ls_ratio:.4f}"
)

print(
    f"  Train LS ratio   : "
    f"{train_df['label'].mean():.4f}"
)

print(
    f"  Test LS ratio    : "
    f"{test_df['label'].mean():.4f}"
)

print("\nLeakage checks:")
print("  Spatial overlap  : 0")
print("  Duplicate coords : 0")

print("\nOutput files:")
print(f"  {OUTPUT_SPLIT}")
print(f"  {OUTPUT_AUDIT}")

print("=" * 70)