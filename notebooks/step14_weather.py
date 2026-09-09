import geopandas as gpd
import xarray as xr
import h5py
import numpy as np
import glob
import os


# ============================================================
# LOAD TRAINING POINTS
# ============================================================

points = gpd.read_file(
    "processed/training_features.geojson"
)

print("Training points:", len(points))


# ============================================================
# STEP 1 — RAINFALL
# ============================================================

rainfall_files = sorted(
    glob.glob("raw_data/rainfall/*.nc4")
)

print("\nRainfall files found:", len(rainfall_files))

for f in rainfall_files:
    print("  ", os.path.basename(f))

if len(rainfall_files) == 0:
    raise FileNotFoundError(
        "No .nc4 rainfall files found"
    )


rainfall_values = np.zeros(len(points))


for file in rainfall_files:

    print(
        "\nReading rainfall:",
        os.path.basename(file)
    )

    ds = xr.open_dataset(file)

    rain = ds["precipitation"]

    print("Dimensions:", rain.dims)
    print(
        "Units:",
        rain.attrs.get("units")
    )

    # --------------------------------------------------------
    # Extract nearest rainfall pixel
    # --------------------------------------------------------

    for i, point in enumerate(points.geometry):

        value = rain.sel(
            lon=point.x,
            lat=point.y,
            method="nearest"
        ).mean().item()

        rainfall_values[i] += value

    ds.close()


points["rainfall_3day"] = rainfall_values


print("\nRainfall extraction complete.")

print(
    points["rainfall_3day"].describe()
)


# ============================================================
# STEP 2 — SMAP SOIL MOISTURE
# ============================================================

smap_files = sorted(
    glob.glob("raw_data/soil_moisture/*.h5")
)

print("\nSMAP files found:", len(smap_files))

for f in smap_files:
    print("  ", os.path.basename(f))

if len(smap_files) == 0:
    raise FileNotFoundError(
        "No .h5 SMAP files found"
    )


soil_values = [[] for _ in range(len(points))]


for file in smap_files:

    print(
        "\nReading SMAP:",
        os.path.basename(file)
    )

    with h5py.File(file, "r") as h5:

        # ----------------------------------------------------
        # Use EXACT dataset names
        # ----------------------------------------------------

        soil_path = (
            "Soil_Moisture_Retrieval_Data_AM/"
            "soil_moisture"
        )

        lat_path = (
            "Soil_Moisture_Retrieval_Data_AM/"
            "latitude"
        )

        lon_path = (
            "Soil_Moisture_Retrieval_Data_AM/"
            "longitude"
        )

        if soil_path not in h5:
            raise KeyError(
                f"Dataset not found: {soil_path}"
            )

        if lat_path not in h5:
            raise KeyError(
                f"Dataset not found: {lat_path}"
            )

        if lon_path not in h5:
            raise KeyError(
                f"Dataset not found: {lon_path}"
            )

        soil = h5[soil_path][:]
        latitude = h5[lat_path][:]
        longitude = h5[lon_path][:]

        print(
            "Soil moisture shape:",
            soil.shape
        )

        print(
            "Soil moisture units:",
            h5[soil_path].attrs.get("units")
        )

        print(
            "Raw soil moisture min:",
            np.nanmin(soil)
        )

        print(
            "Raw soil moisture max:",
            np.nanmax(soil)
        )

        # ----------------------------------------------------
        # Replace SMAP fill values
        # ----------------------------------------------------

        fill_value = h5[soil_path].attrs.get(
            "_FillValue"
        )

        if fill_value is not None:
            soil = np.where(
                soil == fill_value,
                np.nan,
                soil
            )

        # Keep physically meaningful SMAP values
        soil = np.where(
            (soil >= 0.0) &
            (soil <= 1.0),
            soil,
            np.nan
        )

        # ----------------------------------------------------
        # Extract nearest valid SMAP pixel
        # ----------------------------------------------------

        for i, point in enumerate(points.geometry):

            distance = (
                (latitude - point.y) ** 2
                +
                (longitude - point.x) ** 2
            )

            # Ignore invalid coordinates
            distance = np.where(
                np.isfinite(distance),
                distance,
                np.inf
            )

            # Find nearest pixel
            index = np.unravel_index(
                np.argmin(distance),
                distance.shape
            )

            value = soil[index]

            if np.isfinite(value):
                soil_values[i].append(
                    float(value)
                )


# ============================================================
# AVERAGE SMAP OBSERVATIONS
# ============================================================

soil_moisture_values = []

for values in soil_values:

    if len(values) == 0:
        soil_moisture_values.append(np.nan)

    else:
        soil_moisture_values.append(
            float(np.mean(values))
        )


points["soil_moisture"] = (
    soil_moisture_values
)


print(
    "\nSoil moisture extraction complete."
)

print(
    points["soil_moisture"].describe()
)


# ============================================================
# CHECK FINAL VALUES
# ============================================================

print("\nSample weather values:")

print(
    points[
        [
            "label",
            "rainfall_3day",
            "soil_moisture"
        ]
    ].head(20)
)


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
print("STEP 14 COMPLETE")
print("========================================")

print(
    "Saved:",
    output_file
)