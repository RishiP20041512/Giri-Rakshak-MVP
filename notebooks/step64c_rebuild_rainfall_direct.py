"""
STEP 64C — DIRECT IMERG RAINFALL TO FINAL GRID

Purpose
-------
Create the rainfall raster directly from IMERG source coordinates
onto the established EPSG:6933 / 250 m final grid.

Method
------
1. Read the three IMERG files.
2. Correct their dimension order:
       precipitation = (lon, lat)
       -> transpose -> (lat, lon)
3. Remove IMERG fill values.
4. Calculate the 3-day rainfall sum.
5. Generate the CENTER coordinates of every final-grid cell.
6. Transform final-grid centers from EPSG:6933 to WGS84.
7. Find the nearest IMERG latitude/longitude cell for every
   final-grid cell.
8. Assign that IMERG 3-day value directly.
9. Apply the NER mask.
10. Save as the final 250-m rainfall raster.

This avoids rasterio.warp.reproject() for the IMERG source.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from netCDF4 import Dataset
from pyproj import Transformer


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
# NER BOUNDS
# ============================================================

NER_WEST = 88.0
NER_EAST = 97.5
NER_SOUTH = 21.5
NER_NORTH = 29.5


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 64C — DIRECT IMERG RAINFALL RECONSTRUCTION")
print("=" * 70)


# ============================================================
# LOAD FINAL GRID
# ============================================================

print("\n[1] Loading final grid")
print("-" * 70)

with rasterio.open(GRID_FILE) as src:

    dst_crs = src.crs
    dst_transform = src.transform
    dst_width = src.width
    dst_height = src.height

    print(f"CRS        : {dst_crs}")
    print(f"Dimensions : {dst_width} x {dst_height}")
    print(f"Resolution : {src.res}")
    print(f"Bounds     : {src.bounds}")


# ============================================================
# LOAD MASK
# ============================================================

with rasterio.open(MASK_FILE) as src:

    ner_mask = src.read(1)

ner_mask_bool = (
    ner_mask == 1
)

print(
    f"NER cells: {np.sum(ner_mask_bool):,}"
)


# ============================================================
# READ IMERG
# ============================================================

print("\n[2] Reading IMERG")
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

    print(
        f"\nReading: {file.name}"
    )

    with Dataset(file, "r") as ds:

        lat = np.asarray(
            ds.variables["lat"][:],
            dtype=np.float64
        )

        lon = np.asarray(
            ds.variables["lon"][:],
            dtype=np.float64
        )

        variable = ds.variables[
            "precipitation"
        ]

        fill_value = getattr(
            variable,
            "_FillValue",
            -9999.9
        )

        arr = np.asarray(
            variable[:],
            dtype=np.float64
        )

    arr = np.squeeze(arr)

    print(
        f"Raw array shape: {arr.shape}"
    )

    print(
        f"Expected lat x lon: "
        f"{len(lat)} x {len(lon)}"
    )

    # --------------------------------------------------------
    # Correct dimension order
    # --------------------------------------------------------

    if arr.shape == (
        len(lon),
        len(lat)
    ):

        print(
            "Correcting lon x lat -> lat x lon"
        )

        arr = arr.T

    elif arr.shape == (
        len(lat),
        len(lon)
    ):

        print(
            "Array already lat x lon"
        )

    else:

        raise RuntimeError(
            f"Unexpected precipitation shape "
            f"{arr.shape}"
        )

    # --------------------------------------------------------
    # Remove fill values
    # --------------------------------------------------------

    invalid = (
        ~np.isfinite(arr)
        | np.isclose(
            arr,
            fill_value,
            atol=1.0
        )
        | (arr < -1000)
    )

    arr[invalid] = np.nan

    print(
        f"Valid pixels: "
        f"{np.sum(np.isfinite(arr)):,}"
    )

    print(
        f"Min: {np.nanmin(arr):.6f}"
    )

    print(
        f"Max: {np.nanmax(arr):.6f}"
    )

    daily_arrays.append(arr)

    if source_lat is None:

        source_lat = lat
        source_lon = lon


# ============================================================
# VERIFY SOURCE COORDINATES
# ============================================================

lat = source_lat
lon = source_lon

print("\n[3] IMERG coordinate verification")
print("-" * 70)

print(
    f"Latitude : {lat.min():.4f} "
    f"to {lat.max():.4f}"
)

print(
    f"Longitude: {lon.min():.4f} "
    f"to {lon.max():.4f}"
)

if not np.all(
    np.diff(lat) > 0
):

    raise RuntimeError(
        "Latitude is not strictly increasing."
    )

if not np.all(
    np.diff(lon) > 0
):

    raise RuntimeError(
        "Longitude is not strictly increasing."
    )


# ============================================================
# CALCULATE 3-DAY SUM
# ============================================================

print("\n[4] Calculating 3-day rainfall")
print("-" * 70)

stack = np.stack(
    daily_arrays,
    axis=0
)

# Require all three days to be valid.
rain_3day = np.where(
    np.all(
        np.isfinite(stack),
        axis=0
    ),
    np.sum(
        stack,
        axis=0
    ),
    np.nan
)

print(
    f"3-day min : "
    f"{np.nanmin(rain_3day):.6f}"
)

print(
    f"3-day max : "
    f"{np.nanmax(rain_3day):.6f}"
)

print(
    f"3-day mean: "
    f"{np.nanmean(rain_3day):.6f}"
)


# ============================================================
# CROP SOURCE TO NER
# ============================================================

print("\n[5] Cropping source coordinates to NER")
print("-" * 70)

lat_mask = (
    (lat >= NER_SOUTH)
    & (lat <= NER_NORTH)
)

lon_mask = (
    (lon >= NER_WEST)
    & (lon <= NER_EAST)
)

lat_ner = lat[lat_mask]
lon_ner = lon[lon_mask]

rain_ner = rain_3day[
    np.ix_(
        lat_mask,
        lon_mask
    )
]

print(
    f"Latitude cells : {len(lat_ner)}"
)

print(
    f"Longitude cells: {len(lon_ner)}"
)

print(
    f"Source shape   : {rain_ner.shape}"
)

print(
    f"Source extent  : "
    f"{lon_ner.min():.4f} to "
    f"{lon_ner.max():.4f} E"
)

print(
    f"Source extent  : "
    f"{lat_ner.min():.4f} to "
    f"{lat_ner.max():.4f} N"
)


# ============================================================
# GENERATE FINAL GRID CELL CENTERS
# ============================================================

print("\n[6] Generating final-grid cell centers")
print("-" * 70)

rows, cols = np.indices(
    (
        dst_height,
        dst_width
    )
)

x = (
    dst_transform.c
    + (cols + 0.5)
    * dst_transform.a
)

y = (
    dst_transform.f
    + (rows + 0.5)
    * dst_transform.e
)


# ============================================================
# TRANSFORM FINAL GRID TO WGS84
# ============================================================

print(
    "Transforming final-grid centers "
    "to WGS84..."
)

transformer = Transformer.from_crs(
    dst_crs,
    "EPSG:4326",
    always_xy=True
)

lon_grid, lat_grid = transformer.transform(
    x,
    y
)


# ============================================================
# ONLY PROCESS NER CELLS
# ============================================================

print("\n[7] Selecting NER cells")
print("-" * 70)

ner_rows, ner_cols = np.where(
    ner_mask_bool
)

ner_lon = lon_grid[
    ner_rows,
    ner_cols
]

ner_lat = lat_grid[
    ner_rows,
    ner_cols
]

print(
    f"NER cells to process: "
    f"{len(ner_rows):,}"
)


# ============================================================
# FIND NEAREST IMERG CELL
# ============================================================

print("\n[8] Direct nearest-neighbour assignment")
print("-" * 70)

print(
    "Finding nearest IMERG longitude..."
)

# Because longitude is regularly spaced,
# nearest index can be obtained mathematically.

lon_res = float(
    np.median(
        np.diff(lon_ner)
    )
)

lat_res = float(
    np.median(
        np.diff(lat_ner)
    )
)

print(
    f"IMERG longitude spacing: "
    f"{lon_res:.10f}"
)

print(
    f"IMERG latitude spacing: "
    f"{lat_res:.10f}"
)


# Use searchsorted for nearest source coordinate.

lon_indices = np.searchsorted(
    lon_ner,
    ner_lon
)

lon_indices = np.clip(
    lon_indices,
    1,
    len(lon_ner) - 1
)

left_lon = lon_ner[
    lon_indices - 1
]

right_lon = lon_ner[
    lon_indices
]

choose_right_lon = (
    np.abs(
        ner_lon - right_lon
    )
    <
    np.abs(
        ner_lon - left_lon
    )
)

lon_indices = np.where(
    choose_right_lon,
    lon_indices,
    lon_indices - 1
)


print(
    "Finding nearest IMERG latitude..."
)

lat_indices = np.searchsorted(
    lat_ner,
    ner_lat
)

lat_indices = np.clip(
    lat_indices,
    1,
    len(lat_ner) - 1
)

lower_lat = lat_ner[
    lat_indices - 1
]

upper_lat = lat_ner[
    lat_indices
]

choose_upper_lat = (
    np.abs(
        ner_lat - upper_lat
    )
    <
    np.abs(
        ner_lat - lower_lat
    )
)

lat_indices = np.where(
    choose_upper_lat,
    lat_indices,
    lat_indices - 1
)


# ============================================================
# EXTRACT SOURCE VALUES
# ============================================================

print(
    "Assigning IMERG values..."
)

values = rain_ner[
    lat_indices,
    lon_indices
]


# ============================================================
# CREATE OUTPUT RASTER
# ============================================================

rain_target = np.full(
    (
        dst_height,
        dst_width
    ),
    np.nan,
    dtype=np.float32
)

rain_target[
    ner_rows,
    ner_cols
] = values.astype(
    np.float32
)


# ============================================================
# SANITY CHECK
# ============================================================

valid = np.isfinite(
    rain_target
)

print("\n[9] Final raster statistics")
print("-" * 70)

print(
    f"Valid cells: "
    f"{np.sum(valid):,}"
)

print(
    f"Min        : "
    f"{np.nanmin(rain_target):.6f}"
)

print(
    f"Max        : "
    f"{np.nanmax(rain_target):.6f}"
)

print(
    f"Mean       : "
    f"{np.nanmean(rain_target):.6f}"
)


if np.nanmin(
    rain_target
) < 0:

    raise RuntimeError(
        "Negative rainfall detected."
    )


# ============================================================
# SAVE
# ============================================================

print("\n[10] Saving raster")
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

print(
    f"Saved:\n{OUT_FILE}"
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("DIRECT IMERG RAINFALL RECONSTRUCTION COMPLETED")
print("=" * 70)

print("\nNext:")
print(
    "Run: python notebooks\\step64c_qa_environmental_rasters.py"
)

print(
    "\nDo NOT proceed to Step 64D until rainfall QA "
    "is reviewed."
)