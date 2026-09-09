import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import glob

# --------------------------------------------------
# Paths
# --------------------------------------------------

points_path = Path("processed/ner_training_points.geojson")
sm_dir = Path("raw_data/soil_moisture")
output_path = Path("processed/ner_soil_moisture.csv")

# --------------------------------------------------
# Load NER points
# --------------------------------------------------

print("Loading NER training points...")

points = gpd.read_file(points_path)

print("Number of points:", len(points))
print("CRS:", points.crs)

points = points.to_crs("EPSG:4326")

points["lon"] = points.geometry.x
points["lat"] = points.geometry.y

# --------------------------------------------------
# Find SMAP files
# --------------------------------------------------

files = sorted(glob.glob(str(sm_dir / "*.h5")))

if len(files) != 3:
    raise ValueError(
        f"Expected exactly 3 SMAP files, found {len(files)}"
    )

print("\nSMAP files:")
for f in files:
    print(Path(f).name)

# --------------------------------------------------
# SMAP dataset paths
# --------------------------------------------------

base = "Soil_Moisture_Retrieval_Data_AM"

sm_path = base + "/soil_moisture"
lat_path = base + "/latitude"
lon_path = base + "/longitude"

# --------------------------------------------------
# Extract soil moisture
# --------------------------------------------------

daily_values = []

for f in files:

    print("\nProcessing:", Path(f).name)

    with h5py.File(f, "r") as h5:

        soil = h5[sm_path][:]
        lat = h5[lat_path][:]
        lon = h5[lon_path][:]

        soil = np.asarray(soil, dtype="float32")
        lat = np.asarray(lat, dtype="float32")
        lon = np.asarray(lon, dtype="float32")

        # Invalid SMAP values
        soil[soil < 0] = np.nan
        soil[soil > 1] = np.nan

        values = []

        # --------------------------------------------------
        # Nearest-neighbour extraction
        # --------------------------------------------------

        for point_lat, point_lon in zip(
            points["lat"],
            points["lon"]
        ):

            distance = (
                (lat - point_lat) ** 2
                + (lon - point_lon) ** 2
            )

            distance[~np.isfinite(soil)] = np.inf

            index = np.unravel_index(
                np.argmin(distance),
                distance.shape
            )

            value = soil[index]

            values.append(value)

        daily_values.append(values)

# --------------------------------------------------
# Convert to array
# --------------------------------------------------

daily_values = np.array(daily_values, dtype="float32")

# --------------------------------------------------
# 3-day mean soil moisture
# --------------------------------------------------

soil_moisture_3day = np.nanmean(
    daily_values,
    axis=0
)

points["soil_moisture"] = soil_moisture_3day

# --------------------------------------------------
# Output
# --------------------------------------------------

result = points[
    ["lat", "lon", "label", "soil_moisture"]
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

print("\nSoil moisture statistics:")
print("Minimum:", result["soil_moisture"].min())
print("Maximum:", result["soil_moisture"].max())
print("Mean:", result["soil_moisture"].mean())

print("\nMissing values:")
print(result["soil_moisture"].isna().sum())

print("\nFirst 5 rows:")
print(result.head())