"""
STEP 64C — CORRECTED IMERG RAINFALL RASTER

Purpose
-------
Create the 3-day IMERG rainfall raster on the established
EPSG:6933 / 250 m final grid.

Important:
- IMERG precipitation dimensions are (lon, lat)
- Coordinates are regular 0.1-degree cell centers
- Fill value is -9999.9
- Precipitation is transposed to (lat, lon)
- Cell-center coordinates are converted to the correct
  raster transform
- Raster is reprojected directly to the final grid
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from netCDF4 import Dataset


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

GRID_FILE = (
    PROJECT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

MASK_FILE = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

RAIN_DIR = (
    PROJECT
    / "raw_data"
    / "rainfall"
)

OUT_FILE = (
    PROJECT
    / "processed"
    / "predictors"
    / "rainfall_3day_250m.tif"
)


# ============================================================
# LOAD FINAL GRID
# ============================================================

print("=" * 70)
print("STEP 64C — CORRECTED IMERG RAINFALL")
print("=" * 70)

print("\n[1] Loading final grid")
print("-" * 70)

with rasterio.open(GRID_FILE) as src:

    dst_crs = src.crs
    dst_transform = src.transform
    dst_width = src.width
    dst_height = src.height

    print(f"CRS       : {dst_crs}")
    print(f"Width     : {dst_width}")
    print(f"Height    : {dst_height}")
    print(f"Resolution: {src.res}")


# ============================================================
# LOAD NER MASK
# ============================================================

with rasterio.open(MASK_FILE) as src:

    ner_mask = src.read(1)

ner_mask_bool = ner_mask == 1

print(
    f"NER cells : {np.sum(ner_mask_bool):,}"
)


# ============================================================
# ACTUAL NER GEOGRAPHIC BOUNDS
# ============================================================

# These are the established NER analysis bounds.
NER_WEST = 88.0
NER_EAST = 97.5
NER_SOUTH = 21.5
NER_NORTH = 29.5


# ============================================================
# READ IMERG
# ============================================================

print("\n[2] Reading IMERG files")
print("-" * 70)

files = sorted(
    RAIN_DIR.glob("*.nc4")
)

if len(files) != 3:

    raise RuntimeError(
        f"Expected 3 IMERG files, found {len(files)}"
    )


daily_arrays = []

source_lat = None
source_lon = None


for file in files:

    print(f"\nReading: {file.name}")

    with Dataset(file, "r") as ds:

        lat = np.asarray(
            ds.variables["lat"][:],
            dtype=np.float64
        )

        lon = np.asarray(
            ds.variables["lon"][:],
            dtype=np.float64
        )

        variable = ds.variables["precipitation"]

        raw = np.asarray(
            variable[:],
            dtype=np.float64
        )

        fill_value = getattr(
            variable,
            "_FillValue",
            -9999.9
        )

    raw = np.squeeze(raw)

    print(f"Raw shape: {raw.shape}")
    print(f"Fill value: {fill_value}")

    # --------------------------------------------------------
    # Correct dimension order
    # --------------------------------------------------------

    if raw.shape == (
        len(lon),
        len(lat)
    ):

        print(
            "Transposing lon x lat -> lat x lon"
        )

        raw = raw.T

    elif raw.shape == (
        len(lat),
        len(lon)
    ):

        print(
            "Already lat x lon"
        )

    else:

        raise RuntimeError(
            f"Unexpected shape {raw.shape}"
        )

    # --------------------------------------------------------
    # Mask fill values
    # --------------------------------------------------------

    invalid = (
        ~np.isfinite(raw)
        | np.isclose(
            raw,
            fill_value,
            atol=1.0
        )
        | (raw < -1000)
    )

    raw[invalid] = np.nan

    print(
        f"Valid source pixels: "
        f"{np.sum(np.isfinite(raw)):,}"
    )

    print(
        f"Valid min: {np.nanmin(raw):.6f}"
    )

    print(
        f"Valid max: {np.nanmax(raw):.6f}"
    )

    daily_arrays.append(raw)

    if source_lat is None:

        source_lat = lat
        source_lon = lon


# ============================================================
# CHECK COORDINATES
# ============================================================

print("\n[3] Checking IMERG coordinates")
print("-" * 70)

lat = source_lat
lon = source_lon

print(
    f"Latitude : {lat.min():.4f} "
    f"to {lat.max():.4f}"
)

print(
    f"Longitude: {lon.min():.4f} "
    f"to {lon.max():.4f}"
)

print(
    f"Latitude spacing : "
    f"{np.median(np.diff(lat)):.10f}"
)

print(
    f"Longitude spacing: "
    f"{np.median(np.diff(lon)):.10f}"
)

if not np.all(np.diff(lat) > 0):

    raise RuntimeError(
        "Latitude is not increasing."
    )

if not np.all(np.diff(lon) > 0):

    raise RuntimeError(
        "Longitude is not increasing."
    )


# ============================================================
# CROP TO ACTUAL NER BOUNDS
# ============================================================

print("\n[4] Cropping IMERG to NER")
print("-" * 70)

lat_mask = (
    (lat >= NER_SOUTH)
    & (lat <= NER_NORTH)
)

lon_mask = (
    (lon >= NER_WEST)
    & (lon <= NER_EAST)
)

lat_sub = lat[lat_mask]
lon_sub = lon[lon_mask]

print(
    f"Latitude cells : {len(lat_sub)}"
)

print(
    f"Longitude cells: {len(lon_sub)}"
)

print(
    f"Latitude range : "
    f"{lat_sub.min():.4f} "
    f"to {lat_sub.max():.4f}"
)

print(
    f"Longitude range: "
    f"{lon_sub.min():.4f} "
    f"to {lon_sub.max():.4f}"
)


# ============================================================
# 3-DAY SUM
# ============================================================

print("\n[5] Calculating 3-day rainfall")
print("-" * 70)

cropped_days = []

for arr in daily_arrays:

    cropped = arr[
        np.ix_(lat_mask, lon_mask)
    ]

    cropped_days.append(cropped)


stack = np.stack(
    cropped_days,
    axis=0
)

# All three days must be available.
# Do not let missing pixels silently become zero.

rain_3day = np.where(
    np.all(np.isfinite(stack), axis=0),
    np.sum(stack, axis=0),
    np.nan
)

print(
    f"3-day source min : "
    f"{np.nanmin(rain_3day):.6f}"
)

print(
    f"3-day source max : "
    f"{np.nanmax(rain_3day):.6f}"
)

print(
    f"3-day source mean: "
    f"{np.nanmean(rain_3day):.6f}"
)


# ============================================================
# CREATE CORRECT SOURCE TRANSFORM
# ============================================================

print("\n[6] Creating IMERG source raster geometry")
print("-" * 70)

xres = float(
    np.median(np.diff(lon_sub))
)

yres = float(
    np.median(np.diff(lat_sub))
)

# Coordinates are CELL CENTERS.
#
# Raster transform needs OUTER EDGES.
#
# left edge  = first center - half pixel
# top edge   = last center + half pixel

left = float(
    lon_sub.min() - xres / 2
)

top = float(
    lat_sub.max() + yres / 2
)

source_transform = from_origin(
    left,
    top,
    xres,
    yres
)

print(
    f"Source transform: {source_transform}"
)

print(
    f"Source width : {len(lon_sub)}"
)

print(
    f"Source height: {len(lat_sub)}"
)


# ============================================================
# REPROJECT TO FINAL GRID
# ============================================================

print("\n[7] Reprojecting to EPSG:6933 / 250 m")
print("-" * 70)

rain_target = np.full(
    (dst_height, dst_width),
    np.nan,
    dtype=np.float32
)

reproject(
    source=rain_3day.astype(np.float32),
    destination=rain_target,
    src_transform=source_transform,
    src_crs="EPSG:4326",
    src_nodata=np.nan,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear
)


# ============================================================
# APPLY NER MASK
# ============================================================

rain_target[
    ~ner_mask_bool
] = np.nan


# ============================================================
# SANITY CHECK
# ============================================================

valid = np.isfinite(
    rain_target
)

print("\nFinal rainfall raster:")
print(
    f"Valid cells: {np.sum(valid):,}"
)

print(
    f"Min        : {np.nanmin(rain_target):.6f}"
)

print(
    f"Max        : {np.nanmax(rain_target):.6f}"
)

print(
    f"Mean       : {np.nanmean(rain_target):.6f}"
)

if np.nanmin(rain_target) < -0.001:

    raise RuntimeError(
        "Negative rainfall detected."
    )


# ============================================================
# SAVE
# ============================================================

print("\n[8] Saving rainfall raster")
print("-" * 70)

with rasterio.open(
    OUT_FILE,
    "w",
    driver="GTiff",
    height=dst_height,
    width=dst_width,
    count=1,
    dtype="float32",
    crs=dst_crs,
    transform=dst_transform,
    nodata=np.nan,
    compress="deflate",
    predictor=2
) as dst:

    dst.write(
        rain_target,
        1
    )

print(f"Saved:")
print(OUT_FILE)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("CORRECTED RAINFALL RASTER COMPLETED")
print("=" * 70)

print("\nNext:")
print("Run the environmental raster QA script.")
print("Do not proceed to Step 64D until rainfall QA is reviewed.")