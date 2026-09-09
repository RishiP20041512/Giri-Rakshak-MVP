import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
import h5py
import glob
import os


# ============================================================
# LOAD TRAINING POINTS
# ============================================================

points = gpd.read_file(
    "processed/training_features.geojson"
)

print("Training points:", len(points))


# Get coordinates from geometry
points["lat"] = points.geometry.y
points["lon"] = points.geometry.x


# ============================================================
# PART 1 — RAINFALL
# ============================================================

print("\n========================================")
print("EXTRACTING RAINFALL")
print("========================================")

rainfall_files = sorted(
    glob.glob("raw_data/rainfall/*.nc4")
)

print(
    "Rainfall files found:",
    len(rainfall_files)
)

rainfall_values = []


for file in rainfall_files:

    print(
        "\nReading rainfall:",
        os.path.basename(file)
    )

    ds = xr.open_dataset(file)

    rain = ds["precipitation"]

    print(
        "Units:",
        rain.attrs.get("units", "unknown")
    )

    daily_values = []

    for _, point in points.iterrows():

        value = rain.sel(
            lat=point["lat"],
            lon=point["lon"],
            method="nearest"
        ).values

        value = float(
            np.asarray(value).squeeze()
        )

        if not np.isfinite(value):
            value = np.nan

        daily_values.append(value)

    rainfall_values.append(
        daily_values
    )

    ds.close()


rainfall_array = np.array(
    rainfall_values,
    dtype=float
)

points["rainfall_3day"] = np.nansum(
    rainfall_array,
    axis=0
)

print("\nRainfall statistics:")

print(
    points["rainfall_3day"].describe()
)


# ============================================================
# PART 2 — SMAP SOIL MOISTURE
# ============================================================

print("\n========================================")
print("EXTRACTING SMAP SOIL MOISTURE")
print("========================================")


smap_files = sorted(
    glob.glob("raw_data/soil_moisture/*.h5")
)

print(
    "SMAP files found:",
    len(smap_files)
)


soil_values_all_days = []


for file in smap_files:

    print(
        "\nReading SMAP:",
        os.path.basename(file)
    )

    with h5py.File(file, "r") as h5:

        # ----------------------------------------------------
        # CORRECT DATASETS
        # ----------------------------------------------------

        soil = h5[
            "Soil_Moisture_Retrieval_Data_AM/soil_moisture"
        ][:]

        lat = h5[
            "Soil_Moisture_Retrieval_Data_AM/latitude"
        ][:]

        lon = h5[
            "Soil_Moisture_Retrieval_Data_AM/longitude"
        ][:]

        print(
            "Soil moisture shape:",
            soil.shape
        )

        print(
            "Latitude shape:",
            lat.shape
        )

        print(
            "Longitude shape:",
            lon.shape
        )

        # ----------------------------------------------------
        # CLEAN INVALID VALUES
        # ----------------------------------------------------

        soil = soil.astype(np.float32)

        valid = (
            np.isfinite(soil)
            & np.isfinite(lat)
            & np.isfinite(lon)
            & (soil >= 0.0)
            & (soil <= 1.0)
        )

        print(
            "Valid SMAP pixels:",
            int(valid.sum())
        )

        # ----------------------------------------------------
        # EXTRACT NEAREST VALUE FOR EACH TRAINING POINT
        # ----------------------------------------------------

        day_values = []

        for _, point in points.iterrows():

            target_lat = point["lat"]
            target_lon = point["lon"]

            # Restrict search to a small geographic window
            # around the target point to avoid unnecessary work.

            window = (
                valid
                & (lat >= target_lat - 0.5)
                & (lat <= target_lat + 0.5)
                & (lon >= target_lon - 0.5)
                & (lon <= target_lon + 0.5)
            )

            if not np.any(window):

                day_values.append(np.nan)
                continue

            rows, cols = np.where(window)

            lat_values = lat[rows, cols]
            lon_values = lon[rows, cols]

            distance = (
                (lat_values - target_lat) ** 2
                +
                (lon_values - target_lon) ** 2
            )

            nearest = np.argmin(distance)

            r = rows[nearest]
            c = cols[nearest]

            value = float(
                soil[r, c]
            )

            day_values.append(value)

        soil_values_all_days.append(
            day_values
        )


# ============================================================
# AVERAGE 3 DAYS
# ============================================================

soil_array = np.array(
    soil_values_all_days,
    dtype=float
)

points["soil_moisture"] = np.nanmean(
    soil_array,
    axis=0
)


# ============================================================
# SHOW SMAP RESULTS
# ============================================================

print("\n========================================")
print("SMAP RESULTS")
print("========================================")

print(
    points["soil_moisture"].describe()
)


# ============================================================
# CHECK VALUES
# ============================================================

print("\nSample values:")

print(
    points[
        [
            "lat",
            "lon",
            "rainfall_3day",
            "soil_moisture"
        ]
    ].to_string(index=False)
)


# ============================================================
# CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    points[
        [
            "rainfall_3day",
            "soil_moisture"
        ]
    ].isnull().sum()
)


# ============================================================
# REMOVE TEMPORARY COORDINATE COLUMNS
# ============================================================

points = points.drop(
    columns=["lat", "lon"]
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "processed/training_features_weather.geojson"
)

points.to_file(
    output_file,
    driver="GeoJSON"
)


print("\n========================================")
print("STEP 21 COMPLETE")
print("========================================")

print(
    "Saved:",
    output_file
)