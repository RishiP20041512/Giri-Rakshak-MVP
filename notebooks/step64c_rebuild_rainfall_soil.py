"""
STEP 64C — REBUILD RAINFALL AND SOIL-MOISTURE RASTERS

Purpose
-------
Rebuild the rainfall and soil-moisture rasters after coordinate QA.

Rainfall:
- IMERG precipitation arrays are stored as (longitude, latitude)
- Coordinate arrays are (latitude, longitude)
- Therefore precipitation is transposed before spatial processing.

Soil moisture:
- SMAP latitude/longitude are 2-D geolocated arrays
- -9999 values are fill/NoData
- Valid geolocated observations are extracted directly
- No 0.1-degree intermediate grid is used

Target:
- Existing final NER grid
- EPSG:6933
- 250 m
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, calculate_default_transform
from netCDF4 import Dataset
import h5py
from scipy.interpolate import griddata
from pyproj import Transformer


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main")

GRID_FILE = PROJECT / "processed" / "step64a_final_ner_grid.tif"
MASK_FILE = PROJECT / "processed" / "step64a_final_ner_mask.tif"

RAINFALL_DIR = PROJECT / "raw_data" / "rainfall"
SOIL_DIR = PROJECT / "raw_data" / "soil_moisture"

OUT_DIR = PROJECT / "processed" / "predictors"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAINFALL_OUT = OUT_DIR / "rainfall_3day_250m.tif"
SOIL_OUT = OUT_DIR / "soil_moisture_3day_250m.tif"


# ============================================================
# LOAD FINAL GRID
# ============================================================

print("=" * 70)
print("STEP 64C — REBUILD RAINFALL AND SOIL MOISTURE")
print("=" * 70)

print("\n[1] Loading final NER grid")
print("-" * 70)

with rasterio.open(GRID_FILE) as src:
    grid_crs = src.crs
    grid_transform = src.transform
    grid_width = src.width
    grid_height = src.height
    grid_bounds = src.bounds

print(f"CRS       : {grid_crs}")
print(f"Width     : {grid_width}")
print(f"Height    : {grid_height}")
print(f"Resolution: {grid_transform.a} x {abs(grid_transform.e)} m")
print(f"Bounds    : {grid_bounds}")


# ============================================================
# LOAD NER MASK
# ============================================================

with rasterio.open(MASK_FILE) as src:
    ner_mask = src.read(1)

ner_mask_bool = ner_mask == 1

print(f"NER cells : {np.sum(ner_mask_bool):,}")


# ============================================================
# NER BOUNDS IN WGS84
# ============================================================

with rasterio.open(GRID_FILE) as src:
    transformer_to_wgs84 = Transformer.from_crs(
        src.crs,
        "EPSG:4326",
        always_xy=True
    )

    left, bottom = transformer_to_wgs84.transform(
        src.bounds.left,
        src.bounds.bottom
    )

    right, top = transformer_to_wgs84.transform(
        src.bounds.right,
        src.bounds.top
    )

print("\nNER geographic bounds:")
print(f"Longitude: {left:.6f} to {right:.6f}")
print(f"Latitude : {bottom:.6f} to {top:.6f}")


# ============================================================
# 2. RAINFALL
# ============================================================

print("\n\n[2] REBUILDING IMERG RAINFALL")
print("-" * 70)

rain_files = sorted(RAINFALL_DIR.glob("*.nc4"))

if len(rain_files) != 3:
    raise RuntimeError(
        f"Expected 3 IMERG files, found {len(rain_files)}"
    )

rain_arrays = []

for file in rain_files:

    print(f"\nReading: {file.name}")

    with Dataset(file, "r") as ds:

        lat = np.asarray(ds.variables["lat"][:], dtype=np.float32)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float32)

        precip_var = ds.variables["precipitation"]
        precip = np.asarray(precip_var[:], dtype=np.float32)

    # Remove time dimension if present
    precip = np.squeeze(precip)

    print(f"Original precipitation shape: {precip.shape}")
    print(f"Latitude shape             : {lat.shape}")
    print(f"Longitude shape            : {lon.shape}")

    # --------------------------------------------------------
    # IMPORTANT:
    # Diagnostic showed:
    #
    # precipitation = (3600, 1800)
    # coordinates   = lat(1800), lon(3600)
    #
    # Therefore transpose.
    # --------------------------------------------------------

    expected_shape = (len(lat), len(lon))

    if precip.shape == expected_shape:
        print("Shape already matches lat x lon.")

    elif precip.shape == (len(lon), len(lat)):
        print("Transposing precipitation lon x lat -> lat x lon.")
        precip = precip.T

    else:
        raise RuntimeError(
            f"Unexpected precipitation shape {precip.shape}; "
            f"expected {expected_shape} or {(len(lon), len(lat))}"
        )

    # Remove invalid values
    precip = np.where(
        np.isfinite(precip),
        precip,
        np.nan
    )

    print(
        f"Valid precipitation min/max: "
        f"{np.nanmin(precip):.3f} / {np.nanmax(precip):.3f}"
    )

    rain_arrays.append(precip)


# ------------------------------------------------------------
# 3-day rainfall sum
# ------------------------------------------------------------

print("\nCalculating 3-day rainfall sum...")

rain_3day = np.nansum(
    np.stack(rain_arrays, axis=0),
    axis=0
)

print(
    f"3-day rainfall min : {np.nanmin(rain_3day):.3f}"
)
print(
    f"3-day rainfall max : {np.nanmax(rain_3day):.3f}"
)
print(
    f"3-day rainfall mean: {np.nanmean(rain_3day):.3f}"
)


# ------------------------------------------------------------
# Crop to NER geographic extent
# ------------------------------------------------------------

lon_mask = (lon >= left - 0.2) & (lon <= right + 0.2)
lat_mask = (lat >= bottom - 0.2) & (lat <= top + 0.2)

lon_sub = lon[lon_mask]
lat_sub = lat[lat_mask]

rain_sub = rain_3day[
    np.ix_(lat_mask, lon_mask)
]

print("\nRainfall cropped to NER:")
print(f"Latitude cells : {len(lat_sub)}")
print(f"Longitude cells: {len(lon_sub)}")
print(f"Array shape    : {rain_sub.shape}")


# ------------------------------------------------------------
# Create geographic raster
# ------------------------------------------------------------

lon_res = float(np.median(np.diff(lon)))
lat_res = float(np.median(np.diff(lat)))

print(f"\nIMERG resolution:")
print(f"Longitude: {lon_res}")
print(f"Latitude : {lat_res}")


# Raster transform
rain_transform = from_origin(
    float(lon_sub.min() - lon_res / 2),
    float(lat_sub.max() + lat_res / 2),
    lon_res,
    lat_res
)

# ------------------------------------------------------------
# Reproject rainfall directly to final grid
# ------------------------------------------------------------

rain_target = np.full(
    (grid_height, grid_width),
    np.nan,
    dtype=np.float32
)

reproject(
    source=rain_sub.astype(np.float32),
    destination=rain_target,
    src_transform=rain_transform,
    src_crs="EPSG:4326",
    dst_transform=grid_transform,
    dst_crs=grid_crs,
    src_nodata=np.nan,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear
)

# Apply NER mask
rain_target[~ner_mask_bool] = np.nan

print("\nFinal rainfall raster:")
print(f"Valid cells: {np.sum(np.isfinite(rain_target)):,}")
print(f"Min        : {np.nanmin(rain_target):.3f}")
print(f"Max        : {np.nanmax(rain_target):.3f}")
print(f"Mean       : {np.nanmean(rain_target):.3f}")


# ------------------------------------------------------------
# Save rainfall
# ------------------------------------------------------------

with rasterio.open(
    RAINFALL_OUT,
    "w",
    driver="GTiff",
    height=grid_height,
    width=grid_width,
    count=1,
    dtype="float32",
    crs=grid_crs,
    transform=grid_transform,
    nodata=np.nan,
    compress="deflate",
    predictor=2
) as dst:

    dst.write(rain_target, 1)

print(f"\nSaved rainfall:")
print(RAINFALL_OUT)


# ============================================================
# 3. SOIL MOISTURE
# ============================================================

print("\n\n[3] REBUILDING SMAP SOIL MOISTURE")
print("-" * 70)

soil_files = sorted(SOIL_DIR.glob("*.h5"))

if len(soil_files) != 3:
    raise RuntimeError(
        f"Expected 3 SMAP files, found {len(soil_files)}"
    )


# ------------------------------------------------------------
# Collect valid geolocated observations
# ------------------------------------------------------------

soil_points = []

for file in soil_files:

    print(f"\nReading: {file.name}")

    with h5py.File(file, "r") as f:

        group = f["Soil_Moisture_Retrieval_Data_AM"]

        sm = np.asarray(
            group["soil_moisture"][:],
            dtype=np.float32
        )

        lat2d = np.asarray(
            group["latitude"][:],
            dtype=np.float32
        )

        lon2d = np.asarray(
            group["longitude"][:],
            dtype=np.float32
        )

    print(f"Shape: {sm.shape}")

    # --------------------------------------------------------
    # Mask invalid SMAP fill values
    # --------------------------------------------------------

    valid = (
        np.isfinite(sm)
        & np.isfinite(lat2d)
        & np.isfinite(lon2d)
        & (sm > -1000)
        & (lat2d > -90)
        & (lat2d < 90)
        & (lon2d > -180)
        & (lon2d < 180)
    )

    # NER extent
    valid &= (
        (lon2d >= left - 0.2)
        & (lon2d <= right + 0.2)
        & (lat2d >= bottom - 0.2)
        & (lat2d <= top + 0.2)
    )

    n_valid = np.sum(valid)

    print(f"Valid observations in NER: {n_valid:,}")

    soil_points.append(
        pd.DataFrame({
            "lon": lon2d[valid].astype(np.float32),
            "lat": lat2d[valid].astype(np.float32),
            "soil_moisture": sm[valid].astype(np.float32)
        })
    )


soil_df = pd.concat(
    soil_points,
    ignore_index=True
)

print("\nCombined SMAP observations:")
print(f"Rows: {len(soil_df):,}")


# ------------------------------------------------------------
# Aggregate duplicate/near-duplicate coordinates
# ------------------------------------------------------------

soil_df["lon_round"] = soil_df["lon"].round(5)
soil_df["lat_round"] = soil_df["lat"].round(5)

soil_df = (
    soil_df
    .groupby(["lon_round", "lat_round"], as_index=False)
    ["soil_moisture"]
    .mean()
    .rename(columns={
        "lon_round": "lon",
        "lat_round": "lat"
    })
)

print(
    f"Unique spatial observations: {len(soil_df):,}"
)


# ============================================================
# INTERPOLATE SMAP TO FINAL GRID
# ============================================================

print("\nCreating target grid coordinates...")

# Generate target raster cell centers
rows, cols = np.indices(
    (grid_height, grid_width)
)

xs = (
    grid_transform.c
    + (cols + 0.5) * grid_transform.a
)

ys = (
    grid_transform.f
    + (rows + 0.5) * grid_transform.e
)

# Transform target grid to WGS84
transformer_from_grid = Transformer.from_crs(
    grid_crs,
    "EPSG:4326",
    always_xy=True
)

target_lon, target_lat = transformer_from_grid.transform(
    xs,
    ys
)

# Only interpolate where NER mask exists
target_lon_ner = target_lon[ner_mask_bool]
target_lat_ner = target_lat[ner_mask_bool]


# ------------------------------------------------------------
# Transform SMAP points into EPSG:6933
# ------------------------------------------------------------

transformer_to_grid = Transformer.from_crs(
    "EPSG:4326",
    grid_crs,
    always_xy=True
)

soil_x, soil_y = transformer_to_grid.transform(
    soil_df["lon"].values,
    soil_df["lat"].values
)

target_x = xs[ner_mask_bool]
target_y = ys[ner_mask_bool]


# ------------------------------------------------------------
# Nearest-neighbour interpolation
# ------------------------------------------------------------

print("\nInterpolating SMAP observations...")
print("Method: nearest-neighbour")

soil_values = griddata(
    points=np.column_stack([soil_x, soil_y]),
    values=soil_df["soil_moisture"].values,
    xi=np.column_stack([target_x, target_y]),
    method="nearest"
)

soil_target = np.full(
    (grid_height, grid_width),
    np.nan,
    dtype=np.float32
)

soil_target[ner_mask_bool] = soil_values.astype(np.float32)

# Apply physically plausible range
soil_target[
    (soil_target < 0)
    | (soil_target > 1)
] = np.nan

print("\nFinal soil-moisture raster:")
print(
    f"Valid cells: {np.sum(np.isfinite(soil_target)):,}"
)
print(f"Min : {np.nanmin(soil_target):.4f}")
print(f"Max : {np.nanmax(soil_target):.4f}")
print(f"Mean: {np.nanmean(soil_target):.4f}")


# ------------------------------------------------------------
# Save soil moisture
# ------------------------------------------------------------

with rasterio.open(
    SOIL_OUT,
    "w",
    driver="GTiff",
    height=grid_height,
    width=grid_width,
    count=1,
    dtype="float32",
    crs=grid_crs,
    transform=grid_transform,
    nodata=np.nan,
    compress="deflate",
    predictor=2
) as dst:

    dst.write(soil_target, 1)

print(f"\nSaved soil moisture:")
print(SOIL_OUT)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n\n[4] FINAL BASIC VALIDATION")
print("-" * 70)

for name, path in [
    ("Rainfall", RAINFALL_OUT),
    ("Soil moisture", SOIL_OUT)
]:

    with rasterio.open(path) as src:

        print(f"\n{name}:")
        print(f"  CRS        : {src.crs}")
        print(f"  Size       : {src.width} x {src.height}")
        print(
            f"  Resolution : "
            f"{src.res[0]} x {src.res[1]} m"
        )
        print(f"  Bounds     : {src.bounds}")

        arr = src.read(1)

        valid = np.isfinite(arr)

        print(f"  Valid cells: {np.sum(valid):,}")

        if np.any(valid):
            print(f"  Min        : {np.nanmin(arr):.6f}")
            print(f"  Max        : {np.nanmax(arr):.6f}")
            print(f"  Mean       : {np.nanmean(arr):.6f}")


print("\n" + "=" * 70)
print("STEP 64C REBUILD COMPLETED")
print("=" * 70)
print("\nIMPORTANT:")
print("Run the point-level QA script next.")
print("Do NOT proceed to Step 64D until rainfall and soil QA pass.")