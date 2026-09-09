# ============================================================
# STEP 39 — EXTRACT DISTANCE TO NEAREST ROAD
# ============================================================
#
# Input:
#   processed/ner_training_points.geojson
#   raw_data/roads/north-eastern-zone.gpkg
#
# Road layer:
#   gis_osm_roads_free
#
# Output:
#   processed/ner_distance_to_road.csv
#
# Purpose:
#   Calculate the distance from every NER training point
#   to its nearest road.
#
# ============================================================

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POINTS_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ner_training_points.geojson"
)

ROADS_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "roads"
    / "north-eastern-zone.gpkg"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ner_distance_to_road.csv"
)


# ------------------------------------------------------------
# 2. CHECK INPUT FILES
# ------------------------------------------------------------

print("=" * 70)
print("STEP 39 — DISTANCE TO ROAD")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nChecking input files...")

if not POINTS_FILE.exists():
    raise FileNotFoundError(
        f"Training points file not found:\n{POINTS_FILE}"
    )

if not ROADS_FILE.exists():
    raise FileNotFoundError(
        f"Road GeoPackage not found:\n{ROADS_FILE}"
    )

print("Training points file found.")
print("Road GeoPackage found.")


# ------------------------------------------------------------
# 3. READ TRAINING POINTS
# ------------------------------------------------------------

print("\nReading training points...")

points = gpd.read_file(POINTS_FILE)

print("Training points loaded:", len(points))
print("CRS:", points.crs)

if points.empty:
    raise ValueError("Training points file is empty.")


# ------------------------------------------------------------
# 4. CHECK POINT GEOMETRY
# ------------------------------------------------------------

print("\nChecking point geometry...")

if points.geometry.is_empty.any():
    print(
        "WARNING:",
        points.geometry.is_empty.sum(),
        "empty geometries found."
    )

if points.geometry.isna().any():
    print(
        "WARNING:",
        points.geometry.isna().sum(),
        "missing geometries found."
    )

print("Geometry type(s):")
print(points.geometry.geom_type.value_counts())


# ------------------------------------------------------------
# 5. READ ROAD LAYER
# ------------------------------------------------------------

ROAD_LAYER = "gis_osm_roads_free"

print("\nReading road layer:")
print(ROAD_LAYER)

roads = gpd.read_file(
    ROADS_FILE,
    layer=ROAD_LAYER
)

print("Road features loaded:", len(roads))
print("Road CRS:", roads.crs)

if roads.empty:
    raise ValueError("Road layer is empty.")


# ------------------------------------------------------------
# 6. REMOVE EMPTY ROAD GEOMETRIES
# ------------------------------------------------------------

print("\nCleaning road geometries...")

before = len(roads)

roads = roads[
    roads.geometry.notna()
    & (~roads.geometry.is_empty)
].copy()

after = len(roads)

print("Road features before cleaning:", before)
print("Road features after cleaning:", after)


# ------------------------------------------------------------
# 7. CHECK / SET CRS
# ------------------------------------------------------------

if points.crs is None:
    raise ValueError(
        "Training points do not have a CRS."
    )

if roads.crs is None:
    raise ValueError(
        "Road layer does not have a CRS."
    )


# ------------------------------------------------------------
# 8. REPROJECT TO A METRIC CRS
# ------------------------------------------------------------
#
# EPSG:3857 uses metres as the distance unit.
#
# IMPORTANT:
# NER spans multiple UTM zones. EPSG:3857 is therefore being
# used here specifically for the point-to-road distance
# calculation. The final raster CRS/grid strategy will be
# handled separately for the complete susceptibility model.
#
# ------------------------------------------------------------

METRIC_CRS = "EPSG:3857"

print("\nReprojecting data for distance calculation...")
print("Metric CRS:", METRIC_CRS)

points_metric = points.to_crs(METRIC_CRS)
roads_metric = roads.to_crs(METRIC_CRS)

print("Training points CRS:", points_metric.crs)
print("Road CRS:", roads_metric.crs)


# ------------------------------------------------------------
# 9. CALCULATE NEAREST ROAD
# ------------------------------------------------------------

print("\nCalculating nearest-road distance...")

nearest = gpd.sjoin_nearest(
    points_metric,
    roads_metric,
    how="left",
    distance_col="distance_to_road_m"
)

print(
    "\nRaw nearest-road result rows:",
    len(nearest)
)


# ------------------------------------------------------------
# 10. HANDLE MULTIPLE EQUALLY-NEAR ROADS
# ------------------------------------------------------------
#
# A point can have more than one road at exactly the same
# minimum distance. Therefore sjoin_nearest() can return
# more rows than the number of training points.
#
# We keep the minimum distance for each original point.
#
# ------------------------------------------------------------

distance_by_point = (
    nearest
    .groupby(nearest.index)["distance_to_road_m"]
    .min()
)

print(
    "Unique training points with road distance:",
    len(distance_by_point)
)


# ------------------------------------------------------------
# 11. CREATE FINAL OUTPUT TABLE
# ------------------------------------------------------------

print("\nCreating final output table...")

output = pd.DataFrame({
    "lat": points.geometry.y.values,
    "lon": points.geometry.x.values,
})


# Preserve useful original columns

for column in ["date", "source", "region", "label"]:

    if column in points.columns:
        output[column] = points[column].values


# Add nearest-road distance

output["distance_to_road_m"] = [
    distance_by_point.get(idx, np.nan)
    for idx in points.index
]


# ------------------------------------------------------------
# 12. VERIFY OUTPUT SIZE
# ------------------------------------------------------------

print(
    "\nFinal output rows:",
    len(output)
)

if len(output) != len(points):

    raise ValueError(
        "Output row count does not match "
        "training-point count."
    )


# ------------------------------------------------------------
# 13. CHECK MISSING VALUES
# ------------------------------------------------------------

missing = output["distance_to_road_m"].isna().sum()

valid = (
    output["distance_to_road_m"]
    .notna()
    .sum()
)

print("\nDistance-to-road quality check:")
print("Total points :", len(output))
print("Valid        :", valid)
print("Missing      :", missing)


# ------------------------------------------------------------
# 14. DISTANCE STATISTICS
# ------------------------------------------------------------

if valid > 0:

    distance_values = (
        output["distance_to_road_m"]
        .dropna()
    )

    print("\nDistance statistics (metres):")

    print(
        "Minimum :", 
        distance_values.min()
    )

    print(
        "Maximum :",
        distance_values.max()
    )

    print(
        "Mean    :",
        distance_values.mean()
    )

    print(
        "Median  :",
        distance_values.median()
    )


# ------------------------------------------------------------
# 15. SAVE CSV
# ------------------------------------------------------------

print("\nSaving output...")

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nOutput saved successfully:")
print(OUTPUT_FILE)


# ------------------------------------------------------------
# 16. VERIFY SAVED FILE
# ------------------------------------------------------------

if not OUTPUT_FILE.exists():

    raise RuntimeError(
        "Output CSV was not created."
    )

check = pd.read_csv(OUTPUT_FILE)

print("\nSaved file verification:")
print("Rows:", len(check))
print("Columns:", list(check.columns))

print("\nFirst 5 rows:")
print(check.head())


# ------------------------------------------------------------
# 17. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 39 COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nDistance-to-road values are ready for the NER model."
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print("=" * 70)