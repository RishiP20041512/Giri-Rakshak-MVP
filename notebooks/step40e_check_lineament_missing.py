# ============================================================
# STEP 40E — CHECK MISSING LINEAMENT VALUES
# ============================================================
#
# Purpose:
#   1. Read the lineament-density CSV
#   2. Identify all missing values
#   3. Verify their states using the NER boundary
#   4. Check whether missing values are ONLY Mizoram
#   5. Save a diagnostic CSV
#
# ============================================================

from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DENSITY_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ner_lineament_density.csv"
)

POINTS_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ner_training_points.geojson"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "boundaries"
    / "geoBoundaries-IND-ADM1.geojson"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ner_lineament_missing_check.csv"
)


# ============================================================
# 2. HEADER
# ============================================================

print("=" * 70)
print("STEP 40E — LINEAMENT MISSING-VALUE CHECK")
print("=" * 70)


# ============================================================
# 3. CHECK FILES
# ============================================================

print("\nChecking files...")

for file in [
    DENSITY_FILE,
    POINTS_FILE,
    BOUNDARY_FILE,
]:

    if not file.exists():

        raise FileNotFoundError(
            f"File not found:\n{file}"
        )

    print(
        "FOUND:",
        file
    )


# ============================================================
# 4. READ DATA
# ============================================================

print("\nReading lineament-density CSV...")

density = pd.read_csv(
    DENSITY_FILE
)

print(
    "Rows:",
    len(density)
)

print(
    "Columns:",
    list(density.columns)
)


print("\nReading training points...")

points = gpd.read_file(
    POINTS_FILE
)

print(
    "Training points:",
    len(points)
)


print("\nReading NER boundary...")

boundary = gpd.read_file(
    BOUNDARY_FILE
)


# ============================================================
# 5. IDENTIFY MISSING VALUES
# ============================================================

missing_mask = (
    density[
        "lineament_density"
    ]
    .isna()
)

missing = density[
    missing_mask
].copy()


print("\n" + "-" * 70)
print("MISSING-VALUE SUMMARY")
print("-" * 70)

print(
    "Total records:",
    len(density)
)

print(
    "Valid:",
    (~missing_mask).sum()
)

print(
    "Missing:",
    missing_mask.sum()
)


# ============================================================
# 6. JOIN DENSITY TO TRAINING POINTS
# ============================================================
#
# The CSV preserves the same row order as the training points.
#
# Therefore we can safely attach the missing-value status to
# the original geometries.
#
# ============================================================

if len(points) != len(density):

    raise ValueError(
        "Training point count and density CSV count differ."
    )


points = points.copy()

points[
    "lineament_density"
] = density[
    "lineament_density"
].values


# ============================================================
# 7. ASSIGN STATE
# ============================================================

print("\nAssigning state to missing records...")

points_4326 = points.to_crs(
    "EPSG:4326"
)

boundary_4326 = boundary.to_crs(
    "EPSG:4326"
)

joined = gpd.sjoin(
    points_4326,
    boundary_4326[
        [
            "shapeName",
            "geometry"
        ]
    ],
    how="left",
    predicate="within"
)


# Remove duplicate index entries

joined = joined[
    ~joined.index.duplicated(
        keep="first"
    )
].copy()


points_4326[
    "state"
] = (
    joined[
        "shapeName"
    ]
    .reindex(
        points_4326.index
    )
    .values
)


# ============================================================
# 8. EXTRACT ONLY MISSING
# ============================================================

missing_points = points_4326[
    points_4326[
        "lineament_density"
    ].isna()
].copy()


# ============================================================
# 9. PRINT MISSING STATE DISTRIBUTION
# ============================================================

print("\n" + "-" * 70)
print("STATE DISTRIBUTION OF MISSING VALUES")
print("-" * 70)

print(
    missing_points[
        "state"
    ]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 10. PRINT MISSING COORDINATES
# ============================================================

print("\n" + "-" * 70)
print("MISSING RECORDS")
print("-" * 70)

print(
    missing_points[
        [
            "lat",
            "lon",
            "state",
            "label",
            "source"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE DIAGNOSTIC CSV
# ============================================================

diagnostic_columns = [
    "lat",
    "lon",
    "state",
    "label",
    "source",
    "lineament_density"
]

available_columns = [
    column
    for column in diagnostic_columns
    if column in missing_points.columns
]

missing_points[
    available_columns
].to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nDiagnostic file saved:")
print(OUTPUT_FILE)


# ============================================================
# 12. FINAL CHECK
# ============================================================

print("\n" + "-" * 70)
print("FINAL CHECK")
print("-" * 70)

print(
    "Missing records:",
    len(missing_points)
)

print(
    "Diagnostic CSV rows:",
    len(
        pd.read_csv(
            OUTPUT_FILE
        )
    )
)


print("\n" + "=" * 70)
print("STEP 40E COMPLETED")
print("=" * 70)

print(
    "\nSend me the complete terminal output."
)

print("=" * 70)