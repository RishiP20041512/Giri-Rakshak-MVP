import pandas as pd
import geopandas as gpd
import rasterio
from pathlib import Path
from rasterio.sample import sample_gen

print("=" * 70)
print("STEP 48 — EXTRACT ELEVATION AND SLOPE")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

POINTS_FILE = (
    BASE_DIR
    / "processed"
    / "ner_training_points.geojson"
)

DEM_FILE = (
    BASE_DIR
    / "raw_data"
    / "dem_ner"
    / "dem_ner_projected.tif"
)

SLOPE_FILE = (
    BASE_DIR
    / "raw_data"
    / "dem_ner"
    / "slope_ner.tif"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_elevation_slope.csv"
)

# ---------------------------------------------------------
# 1. LOAD TRAINING POINTS
# ---------------------------------------------------------
print("\n[1] Loading training points...")

points = gpd.read_file(POINTS_FILE)

print(f"  Training points: {len(points)}")
print(f"  Point CRS: {points.crs}")

if len(points) != 1071:
    raise ValueError(
        f"Expected 1071 training points, found {len(points)}"
    )

# ---------------------------------------------------------
# 2. FUNCTION FOR RASTER EXTRACTION
# ---------------------------------------------------------
def extract_raster_values(points_gdf, raster_path, factor_name):

    print(f"\n[2] Extracting {factor_name}...")
    print(f"  Raster: {raster_path}")

    with rasterio.open(raster_path) as src:

        print(f"  Raster CRS: {src.crs}")
        print(f"  Raster size: {src.width} x {src.height}")
        print(f"  Raster resolution: {src.res}")
        print(f"  NoData value: {src.nodata}")

        # Transform points to raster CRS
        points_raster_crs = points_gdf.to_crs(src.crs)

        coordinates = [
            (geom.x, geom.y)
            for geom in points_raster_crs.geometry
        ]

        values = []

        for value in src.sample(coordinates):

            value = value[0]

            if src.nodata is not None and value == src.nodata:
                values.append(float("nan"))
            else:
                values.append(float(value))

        values = pd.Series(values)

        print(f"  Valid values: {values.notna().sum()}")
        print(f"  Missing values: {values.isna().sum()}")

        if values.notna().sum() > 0:

            print(f"  Minimum: {values.min()}")
            print(f"  Maximum: {values.max()}")
            print(f"  Mean: {values.mean()}")
            print(f"  Median: {values.median()}")

        return values


# ---------------------------------------------------------
# 3. EXTRACT ELEVATION
# ---------------------------------------------------------
elevation = extract_raster_values(
    points,
    DEM_FILE,
    "elevation"
)

# ---------------------------------------------------------
# 4. EXTRACT SLOPE
# ---------------------------------------------------------
slope = extract_raster_values(
    points,
    SLOPE_FILE,
    "slope"
)

# ---------------------------------------------------------
# 5. BUILD OUTPUT TABLE
# ---------------------------------------------------------
print("\n[3] Creating elevation/slope table...")

# Use the original point coordinates rather than
# transformed coordinates.

output = pd.DataFrame({
    "lat": points.geometry.y,
    "lon": points.geometry.x,
    "label": points["label"],
    "elevation_m": elevation,
    "slope_deg": slope
})

print(f"  Output rows: {len(output)}")

# ---------------------------------------------------------
# 6. CHECK COORDINATES
# ---------------------------------------------------------
print("\n[4] Checking coordinates...")

duplicate_count = output.duplicated(
    subset=["lat", "lon"]
).sum()

print(f"  Duplicate coordinates: {duplicate_count}")

if duplicate_count > 0:
    raise ValueError(
        "Duplicate coordinates detected."
    )

# ---------------------------------------------------------
# 7. CHECK MISSING VALUES
# ---------------------------------------------------------
print("\n[5] Missing-value check...")

for column in [
    "elevation_m",
    "slope_deg"
]:

    missing = output[column].isna().sum()

    print(
        f"  {column}: "
        f"{missing} missing / "
        f"{len(output)} total"
    )

# ---------------------------------------------------------
# 8. CHECK SLOPE RANGE
# ---------------------------------------------------------
print("\n[6] Checking slope values...")

invalid_slope = output[
    (output["slope_deg"] < 0)
    | (output["slope_deg"] > 90)
]

print(f"  Invalid slope values: {len(invalid_slope)}")

if len(invalid_slope) > 0:
    raise ValueError(
        "Slope values outside the valid 0–90 degree range detected."
    )

# ---------------------------------------------------------
# 9. SAVE
# ---------------------------------------------------------
print("\n[7] Saving output...")

output.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"  Saved: {OUTPUT_FILE}")

# ---------------------------------------------------------
# 10. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 48 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput columns:")
print("  lat")
print("  lon")
print("  label")
print("  elevation_m")
print("  slope_deg")

print("\nFinal statistics:")

for column in [
    "elevation_m",
    "slope_deg"
]:

    print(f"\n{column}")
    print(f"  Min    : {output[column].min()}")
    print(f"  Max    : {output[column].max()}")
    print(f"  Mean   : {output[column].mean()}")
    print(f"  Median : {output[column].median()}")
    print(f"  Std    : {output[column].std()}")

print("\nOutput file:")
print(f"  {OUTPUT_FILE}")

print("=" * 70)