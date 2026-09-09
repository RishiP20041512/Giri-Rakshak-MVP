import geopandas as gpd
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import glob

# --------------------------------------------------
# Paths
# --------------------------------------------------

points_path = Path("processed/ner_training_points.geojson")
rainfall_dir = Path("raw_data/rainfall")
output_path = Path("processed/ner_rainfall.csv")

# --------------------------------------------------
# Load NER training points
# --------------------------------------------------

print("Loading NER training points...")

points = gpd.read_file(points_path)

print("Number of points:", len(points))
print("CRS:", points.crs)

# Make sure coordinates are WGS84
if points.crs is None:
    points = points.set_crs("EPSG:4326")
else:
    points = points.to_crs("EPSG:4326")

# Extract coordinates
points["lon"] = points.geometry.x
points["lat"] = points.geometry.y

# --------------------------------------------------
# Find rainfall files
# --------------------------------------------------

files = sorted(glob.glob(str(rainfall_dir / "*.nc4")))

if len(files) != 3:
    raise ValueError(
        f"Expected exactly 3 rainfall files, found {len(files)}"
    )

print("\nRainfall files:")
for f in files:
    print(Path(f).name)

# --------------------------------------------------
# Extract rainfall for each day
# --------------------------------------------------

daily_values = []

for f in files:

    print("\nProcessing:", Path(f).name)

    ds = xr.open_dataset(f)

    rainfall = ds["precipitation"].squeeze("time")

    values = rainfall.interp(
        lon=xr.DataArray(
            points["lon"].values,
            dims="points"
        ),
        lat=xr.DataArray(
            points["lat"].values,
            dims="points"
        ),
        method="linear"
    ).values

    daily_values.append(values)

    ds.close()

# --------------------------------------------------
# Calculate 3-day cumulative rainfall
# --------------------------------------------------

daily_values = np.vstack(daily_values)

rainfall_3day = np.sum(
    daily_values,
    axis=0
)

points["rainfall_3day"] = rainfall_3day

# --------------------------------------------------
# Create output table
# --------------------------------------------------

result = points[
    ["lat", "lon", "label", "rainfall_3day"]
].copy()

result.to_csv(output_path, index=False)

# --------------------------------------------------
# Statistics
# --------------------------------------------------

print("\n========================================")
print("SUCCESS!")
print("========================================")

print("Saved:", output_path)
print("Rows:", len(result))

print("\nRainfall statistics:")
print("Minimum:", result["rainfall_3day"].min())
print("Maximum:", result["rainfall_3day"].max())
print("Mean:", result["rainfall_3day"].mean())

print("\nMissing values:")
print(result["rainfall_3day"].isna().sum())

print("\nFirst 5 rows:")
print(result.head())