"""
STEP 64C — PREPARE RAINFALL, SOIL MOISTURE AND NDVI
====================================================

Creates the environmental predictor rasters on the final
NER analytical grid.

Final grid:
    CRS       = EPSG:6933
    Resolution = 250 m x 250 m

Predictors:
    1. rainfall_3day
    2. soil_moisture
    3. NDVI

Rainfall:
    IMERG daily data for 2023-06-01, 02, 03.
    Three daily precipitation values are summed.

Soil moisture:
    SMAP L3 AM, 2023-06-01, 02, 03.
    Three daily soil-moisture values are averaged.

NDVI:
    MODIS/Terra MOD13Q1 V6.1.
    250 m 16-day composite covering approximately
    2023-05-25 to 2023-06-09.
    Scale factor = 10000.

Important:
    Resampling does not create new native-resolution
    information.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.enums import Resampling

import netCDF4
import h5py

from pyhdf.SD import SD


# ================================================================
# 1. PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GRID_FILE = (
    PROJECT_ROOT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

MASK_FILE = (
    PROJECT_ROOT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

RAINFALL_DIR = (
    PROJECT_ROOT
    / "raw_data"
    / "rainfall"
)

SOIL_DIR = (
    PROJECT_ROOT
    / "raw_data"
    / "soil_moisture"
)

NDVI_DIR = (
    PROJECT_ROOT
    / "raw_data"
    / "ndvi"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "predictors"
)

RAINFALL_OUTPUT = (
    OUTPUT_DIR
    / "rainfall_3day_250m.tif"
)

SOIL_OUTPUT = (
    OUTPUT_DIR
    / "soil_moisture_3day_250m.tif"
)

NDVI_OUTPUT = (
    OUTPUT_DIR
    / "ndvi_250m.tif"
)


# ================================================================
# 2. SETTINGS
# ================================================================

TARGET_CRS = "EPSG:6933"
TARGET_RES = 250.0

# Coarse geographic grid used only as an intermediate
# representation for SMAP.
SOIL_INTERMEDIATE_RES = 0.1


# ================================================================
# 3. HEADER
# ================================================================

print("=" * 70)
print("STEP 64C — PREPARE ENVIRONMENTAL PREDICTORS")
print("=" * 70)


# ================================================================
# 4. CHECK REQUIRED FILES
# ================================================================

print("\n[1] Checking required inputs...")
print("-" * 70)

required = [
    GRID_FILE,
    MASK_FILE,
    RAINFALL_DIR,
    SOIL_DIR,
    NDVI_DIR,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{path}"
        )

    print(
        "  Found:",
        path.relative_to(PROJECT_ROOT)
    )


# ================================================================
# 5. READ FINAL GRID
# ================================================================

print("\n[2] Reading final grid...")
print("-" * 70)

with rasterio.open(
    GRID_FILE
) as src:

    target_crs = src.crs
    target_transform = src.transform
    target_width = src.width
    target_height = src.height
    target_res = src.res

print(
    f"  CRS        : {target_crs}"
)

print(
    f"  Dimensions : "
    f"{target_width} x {target_height}"
)

print(
    f"  Resolution : {target_res}"
)

if str(target_crs) != TARGET_CRS:
    raise ValueError(
        f"Unexpected target CRS: {target_crs}"
    )

if target_res != (
    TARGET_RES,
    TARGET_RES
):
    raise ValueError(
        f"Unexpected target resolution: "
        f"{target_res}"
    )


# ================================================================
# 6. READ NER MASK
# ================================================================

print("\n[3] Reading NER mask...")
print("-" * 70)

with rasterio.open(
    MASK_FILE
) as src:

    ner_mask = src.read(1)

if ner_mask.shape != (
    target_height,
    target_width
):

    raise ValueError(
        "NER mask does not match final grid."
    )

inside_cells = int(
    np.sum(ner_mask == 1)
)

print(
    f"  NER cells: {inside_cells:,}"
)


# ================================================================
# 7. CREATE OUTPUT DIRECTORY
# ================================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 8. RAINFALL
# ================================================================

print("\n" + "=" * 70)
print("PART A — RAINFALL")
print("=" * 70)

rainfall_files = sorted(
    RAINFALL_DIR.glob("*.nc4")
)

if len(rainfall_files) != 3:

    raise ValueError(
        f"Expected exactly 3 IMERG files, "
        f"found {len(rainfall_files)}."
    )

print(
    f"\n  Found {len(rainfall_files)} rainfall files."
)


rainfall_arrays = []

rainfall_source_transform = None
rainfall_source_crs = "EPSG:4326"


for file in rainfall_files:

    print(
        f"\n  Reading: {file.name}"
    )

    with netCDF4.Dataset(
        file,
        "r"
    ) as ds:

        precip = ds.variables[
            "precipitation"
        ]

        lats = np.asarray(
            ds.variables["lat"][:]
        )

        lons = np.asarray(
            ds.variables["lon"][:]
        )

        data = np.asarray(
            precip[0, :, :]
        ).astype(
            "float32"
        )

        # NetCDF dimensions are:
        # lat x lon
        #
        # The file metadata reports the variable as
        # (time, 3600, 1800), so explicitly use the
        # coordinate lengths rather than assuming order.

        if data.shape == (
            len(lats),
            len(lons)
        ):

            pass

        elif data.shape == (
            len(lons),
            len(lats)
        ):

            data = data.T

        else:

            raise ValueError(
                f"Unexpected precipitation shape "
                f"{data.shape}"
            )

        # Handle fill/missing values.
        fill_value = getattr(
            precip,
            "_FillValue",
            None
        )

        if fill_value is not None:

            data[
                data == fill_value
            ] = np.nan

        data[
            ~np.isfinite(data)
        ] = np.nan

        print(
            f"    Shape: {data.shape}"
        )

        print(
            f"    Min: {np.nanmin(data):.6f}"
        )

        print(
            f"    Max: {np.nanmax(data):.6f}"
        )

        rainfall_arrays.append(
            data
        )


# ---------------------------------------------------------------
# Align rainfall arrays
# ---------------------------------------------------------------

if not (
    rainfall_arrays[0].shape
    ==
    rainfall_arrays[1].shape
    ==
    rainfall_arrays[2].shape
):

    raise ValueError(
        "Rainfall source arrays do not "
        "have identical dimensions."
    )


# ---------------------------------------------------------------
# 3-day accumulation
# ---------------------------------------------------------------

rainfall_3day = np.nansum(
    np.stack(
        rainfall_arrays,
        axis=0
    ),
    axis=0
).astype(
    "float32"
)


print("\n  3-day rainfall:")
print(
    f"    Min : {np.nanmin(rainfall_3day):.6f}"
)
print(
    f"    Max : {np.nanmax(rainfall_3day):.6f}"
)
print(
    f"    Mean: {np.nanmean(rainfall_3day):.6f}"
)


# ---------------------------------------------------------------
# Construct geographic source transform
# ---------------------------------------------------------------

lon_res = abs(
    float(lons[1] - lons[0])
)

lat_res = abs(
    float(lats[1] - lats[0])
)

lon_min = float(
    np.min(lons)
)

lat_max = float(
    np.max(lats)
)

rainfall_transform = from_origin(
    lon_min - lon_res / 2,
    lat_max + lat_res / 2,
    lon_res,
    lat_res
)


# ---------------------------------------------------------------
# Reproject rainfall to final grid
# ---------------------------------------------------------------

rainfall_target = np.full(
    (
        target_height,
        target_width
    ),
    np.nan,
    dtype="float32"
)

reproject(
    source=rainfall_3day,
    destination=rainfall_target,
    src_transform=rainfall_transform,
    src_crs=rainfall_source_crs,
    src_nodata=np.nan,
    dst_transform=target_transform,
    dst_crs=target_crs,
    dst_nodata=np.nan,
    resampling=Resampling.bilinear,
)

rainfall_target[
    ner_mask != 1
] = np.nan


# ---------------------------------------------------------------
# Save rainfall
# ---------------------------------------------------------------

profile = {
    "driver": "GTiff",
    "height": target_height,
    "width": target_width,
    "count": 1,
    "dtype": "float32",
    "crs": target_crs,
    "transform": target_transform,
    "nodata": np.nan,
    "compress": "deflate",
    "predictor": 2,
}

with rasterio.open(
    RAINFALL_OUTPUT,
    "w",
    **profile
) as dst:

    dst.write(
        rainfall_target,
        1
    )

    dst.set_band_description(
        1,
        "rainfall_3day_mm"
    )

print(
    "\n  Saved:",
    RAINFALL_OUTPUT.relative_to(
        PROJECT_ROOT
    )
)


# ================================================================
# 9. SOIL MOISTURE
# ================================================================

print("\n" + "=" * 70)
print("PART B — SOIL MOISTURE")
print("=" * 70)

soil_files = sorted(
    list(
        SOIL_DIR.glob("*.h5")
    )
    +
    list(
        SOIL_DIR.glob("*.hdf5")
    )
)

if len(soil_files) != 3:

    raise ValueError(
        f"Expected exactly 3 SMAP HDF5 files, "
        f"found {len(soil_files)}."
    )

print(
    f"\n  Found {len(soil_files)} soil-moisture files."
)


soil_daily = []


for file in soil_files:

    print(
        f"\n  Reading: {file.name}"
    )

    with h5py.File(
        file,
        "r"
    ) as h5:

        group = h5[
            "Soil_Moisture_Retrieval_Data_AM"
        ]

        sm = np.asarray(
            group[
                "soil_moisture"
            ][:]
        ).astype(
            "float32"
        )

        lat = np.asarray(
            group[
                "latitude"
            ][:]
        ).astype(
            "float32"
        )

        lon = np.asarray(
            group[
                "longitude"
            ][:]
        ).astype(
            "float32"
        )

        # SMAP fill values are represented as
        # invalid/extreme values. Keep only
        # physically plausible retrievals.

        sm[
            ~np.isfinite(sm)
        ] = np.nan

        sm[
            (sm < 0)
            |
            (sm > 1)
        ] = np.nan

        print(
            f"    Shape: {sm.shape}"
        )

        print(
            f"    Valid: "
            f"{np.sum(np.isfinite(sm)):,}"
        )

        print(
            f"    Min: "
            f"{np.nanmin(sm):.6f}"
        )

        print(
            f"    Max: "
            f"{np.nanmax(sm):.6f}"
        )

        soil_daily.append(
            (
                sm,
                lat,
                lon
            )
        )


# ---------------------------------------------------------------
# Check source geometry consistency
# ---------------------------------------------------------------

base_lat = soil_daily[0][1]
base_lon = soil_daily[0][2]

for i in range(1, 3):

    if soil_daily[i][1].shape != base_lat.shape:
        raise ValueError(
            "SMAP latitude grids differ "
            "between dates."
        )

    if soil_daily[i][2].shape != base_lon.shape:
        raise ValueError(
            "SMAP longitude grids differ "
            "between dates."
        )


# ---------------------------------------------------------------
# 3-day mean
# ---------------------------------------------------------------

soil_stack = np.stack(
    [
        item[0]
        for item in soil_daily
    ],
    axis=0
)

soil_mean = np.nanmean(
    soil_stack,
    axis=0
).astype(
    "float32"
)


print("\n  3-day soil-moisture mean:")
print(
    f"    Min : {np.nanmin(soil_mean):.6f}"
)
print(
    f"    Max : {np.nanmax(soil_mean):.6f}"
)
print(
    f"    Mean: {np.nanmean(soil_mean):.6f}"
)


# ---------------------------------------------------------------
# Build coarse geographic grid
#
# SMAP has 2-D geolocation arrays.
# We convert the valid source observations into a
# 0.1-degree intermediate geographic grid.
#
# This is deliberately coarse and is NOT presented as
# native 250 m soil-moisture information.
# ---------------------------------------------------------------

valid = (
    np.isfinite(soil_mean)
    &
    np.isfinite(base_lat)
    &
    np.isfinite(base_lon)
)

source_lat = base_lat[
    valid
]

source_lon = base_lon[
    valid
]

source_sm = soil_mean[
    valid
]


# ---------------------------------------------------------------
# Limit to approximate NER geographic bounds
# ---------------------------------------------------------------

NER_WEST = 88.0
NER_EAST = 97.5
NER_SOUTH = 21.5
NER_NORTH = 29.5

inside = (
    (source_lon >= NER_WEST)
    &
    (source_lon <= NER_EAST)
    &
    (source_lat >= NER_SOUTH)
    &
    (source_lat <= NER_NORTH)
)

source_lat = source_lat[
    inside
]

source_lon = source_lon[
    inside
]

source_sm = source_sm[
    inside
]

print(
    f"\n  Valid SMAP source observations "
    f"inside NER: {len(source_sm):,}"
)

if len(source_sm) == 0:

    raise ValueError(
        "No valid SMAP observations "
        "found inside NER."
    )


# ---------------------------------------------------------------
# Create intermediate 0.1-degree grid
# ---------------------------------------------------------------

soil_lon_centers = np.arange(
    NER_WEST,
    NER_EAST + SOIL_INTERMEDIATE_RES,
    SOIL_INTERMEDIATE_RES
)

soil_lat_centers = np.arange(
    NER_SOUTH,
    NER_NORTH + SOIL_INTERMEDIATE_RES,
    SOIL_INTERMEDIATE_RES
)

soil_width = len(
    soil_lon_centers
)

soil_height = len(
    soil_lat_centers
)

soil_grid = np.full(
    (
        soil_height,
        soil_width
    ),
    np.nan,
    dtype="float32"
)


# ---------------------------------------------------------------
# Nearest-neighbor assignment from SMAP source observations
# to the coarse geographic grid.
#
# Process row-by-row to avoid excessive memory use.
# ---------------------------------------------------------------

print(
    "\n  Creating coarse SMAP intermediate grid..."
)

for r, lat_center in enumerate(
    soil_lat_centers
):

    # Select source points close to the
    # current latitude band.

    lat_band = np.abs(
        source_lat - lat_center
    ) <= (
        SOIL_INTERMEDIATE_RES / 2
    )

    if not np.any(lat_band):
        continue

    row_lon = source_lon[
        lat_band
    ]

    row_sm = source_sm[
        lat_band
    ]

    for c, lon_center in enumerate(
        soil_lon_centers
    ):

        distances = np.abs(
            row_lon - lon_center
        )

        if len(distances) == 0:
            continue

        nearest = np.argmin(
            distances
        )

        if (
            distances[nearest]
            <=
            SOIL_INTERMEDIATE_RES
        ):

            soil_grid[
                r,
                c
            ] = row_sm[
                nearest
            ]


print(
    f"  Intermediate grid: "
    f"{soil_width} x {soil_height}"
)

valid_intermediate = soil_grid[
    np.isfinite(soil_grid)
]

print(
    f"  Valid intermediate cells: "
    f"{len(valid_intermediate):,}"
)


# ---------------------------------------------------------------
# Geographic transform
# ---------------------------------------------------------------

soil_transform = from_origin(
    NER_WEST
    - SOIL_INTERMEDIATE_RES / 2,
    NER_NORTH
    + SOIL_INTERMEDIATE_RES / 2,
    SOIL_INTERMEDIATE_RES,
    SOIL_INTERMEDIATE_RES
)


# ---------------------------------------------------------------
# Reproject to final 250 m grid
# ---------------------------------------------------------------

soil_target = np.full(
    (
        target_height,
        target_width
    ),
    np.nan,
    dtype="float32"
)

reproject(
    source=soil_grid,
    destination=soil_target,
    src_transform=soil_transform,
    src_crs="EPSG:4326",
    src_nodata=np.nan,
    dst_transform=target_transform,
    dst_crs=target_crs,
    dst_nodata=np.nan,
    resampling=Resampling.nearest,
)

soil_target[
    ner_mask != 1
] = np.nan


# ---------------------------------------------------------------
# Save soil moisture
# ---------------------------------------------------------------

with rasterio.open(
    SOIL_OUTPUT,
    "w",
    **profile
) as dst:

    dst.write(
        soil_target,
        1
    )

    dst.set_band_description(
        1,
        "soil_moisture_cm3_cm3"
    )

print(
    "\n  Saved:",
    SOIL_OUTPUT.relative_to(
        PROJECT_ROOT
    )
)


# ================================================================
# 10. NDVI
# ================================================================

print("\n" + "=" * 70)
print("PART C — NDVI")
print("=" * 70)

ndvi_files = sorted(
    NDVI_DIR.glob("*.hdf")
)

if len(ndvi_files) != 3:

    raise ValueError(
        f"Expected exactly 3 MODIS HDF files, "
        f"found {len(ndvi_files)}."
    )

print(
    f"\n  Found {len(ndvi_files)} NDVI HDF files."
)


# ---------------------------------------------------------------
# Function to read MODIS NDVI
# ---------------------------------------------------------------

def read_modis_ndvi(
    file
):

    hdf = SD(
        str(file)
    )

    datasets = hdf.datasets()

    if (
        "250m 16 days NDVI"
        not in datasets
    ):

        hdf.end()

        raise ValueError(
            f"NDVI dataset not found in {file.name}"
        )

    ndvi_dataset = hdf.select(
        "250m 16 days NDVI"
    )

    data = np.asarray(
        ndvi_dataset[:]
    ).astype(
        "float32"
    )

    attrs = (
        ndvi_dataset.attributes()
    )

    scale_factor = float(
        attrs.get(
            "scale_factor",
            10000.0
        )
    )

    fill_value = attrs.get(
        "_FillValue",
        None
    )

    hdf.end()

    if fill_value is not None:

        data[
            data == fill_value
        ] = np.nan

    data[
        ~np.isfinite(data)
    ] = np.nan

    # Apply MODIS scale factor.

    data = (
        data
        /
        scale_factor
    )

    # NDVI physically lies approximately in
    # [-1, 1]. Remove impossible values.

    data[
        (data < -1.0)
        |
        (data > 1.0)
    ] = np.nan

    return data


# ---------------------------------------------------------------
# Read all three tiles
# ---------------------------------------------------------------

ndvi_arrays = []

for file in ndvi_files:

    print(
        f"\n  Reading: {file.name}"
    )

    data = read_modis_ndvi(
        file
    )

    print(
        f"    Shape: {data.shape}"
    )

    print(
        f"    Valid: "
        f"{np.sum(np.isfinite(data)):,}"
    )

    if np.any(
        np.isfinite(data)
    ):

        print(
            f"    Min: "
            f"{np.nanmin(data):.6f}"
        )

        print(
            f"    Max: "
            f"{np.nanmax(data):.6f}"
        )

    ndvi_arrays.append(
        data
    )


# ---------------------------------------------------------------
# MODIS sinusoidal tile handling
#
# MOD13Q1 tiles are in the MODIS Sinusoidal projection.
#
# The three tiles together cover the relevant NER area.
# Each tile is 4800 x 4800 at 250 m.
#
# Instead of mosaicking in geographic coordinates manually,
# use the MODIS tile geolocation metadata to establish the
# tile transform.
# ---------------------------------------------------------------

# MODIS Sinusoidal parameters.
MODIS_R = 6371007.181
MODIS_FALSE_EASTING = 0.0
MODIS_FALSE_NORTHING = 0.0

# Standard MODIS tile dimensions.
TILE_SIZE_M = 1111950.5196666666

PIXEL_SIZE_MODIS = (
    TILE_SIZE_M / 4800.0
)


# ---------------------------------------------------------------
# Determine tile h/v from filenames
# ---------------------------------------------------------------

import re

tile_info = []

for file, data in zip(
    ndvi_files,
    ndvi_arrays
):

    match = re.search(
        r"\.h(\d+)v(\d+)\.",
        file.name
    )

    if match is None:

        raise ValueError(
            f"Could not determine MODIS tile "
            f"from {file.name}"
        )

    h = int(
        match.group(1)
    )

    v = int(
        match.group(2)
    )

    tile_info.append(
        (
            h,
            v,
            data,
            file
        )
    )

    print(
        f"    Tile h{h:02d}v{v:02d}"
    )


# ---------------------------------------------------------------
# Convert MODIS tile indices to sinusoidal bounds
# ---------------------------------------------------------------

def modis_tile_transform(
    h,
    v
):

    # MODIS sinusoidal global origin.
    x_min_global = (
        -20015109.354
    )

    y_max_global = (
        10007554.677
    )

    x_min = (
        x_min_global
        +
        h * TILE_SIZE_M
    )

    y_max = (
        y_max_global
        -
        v * TILE_SIZE_M
    )

    return from_origin(
        x_min,
        y_max,
        PIXEL_SIZE_MODIS,
        PIXEL_SIZE_MODIS
    )


# ---------------------------------------------------------------
# Reproject each MODIS tile directly to final grid
# and combine using valid pixels.
# ---------------------------------------------------------------

ndvi_target = np.full(
    (
        target_height,
        target_width
    ),
    np.nan,
    dtype="float32"
)


for h, v, data, file in tile_info:

    print(
        f"\n  Reprojecting tile "
        f"h{h:02d}v{v:02d}..."
    )

    tile_target = np.full(
        (
            target_height,
            target_width
        ),
        np.nan,
        dtype="float32"
    )

    transform = modis_tile_transform(
        h,
        v
    )

    reproject(
        source=data,
        destination=tile_target,
        src_transform=transform,
        src_crs=(
            "+proj=sinu "
            "+R=6371007.181 "
            "+nadgrids=@null "
            "+wktext "
            "+units=m "
            "+no_defs"
        ),
        src_nodata=np.nan,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )

    valid = np.isfinite(
        tile_target
    )

    # Fill only currently empty cells.
    replace = (
        valid
        &
        ~np.isfinite(ndvi_target)
    )

    ndvi_target[
        replace
    ] = tile_target[
        replace
    ]


# ---------------------------------------------------------------
# Apply NER mask
# ---------------------------------------------------------------

ndvi_target[
    ner_mask != 1
] = np.nan


# ---------------------------------------------------------------
# Save NDVI
# ---------------------------------------------------------------

with rasterio.open(
    NDVI_OUTPUT,
    "w",
    **profile
) as dst:

    dst.write(
        ndvi_target,
        1
    )

    dst.set_band_description(
        1,
        "NDVI"
    )


print(
    "\n  Saved:",
    NDVI_OUTPUT.relative_to(
        PROJECT_ROOT
    )
)


# ================================================================
# 11. OUTPUT STATISTICS
# ================================================================

print("\n" + "=" * 70)
print("PART D — OUTPUT STATISTICS")
print("=" * 70)


outputs = [
    (
        RAINFALL_OUTPUT,
        "rainfall_3day"
    ),
    (
        SOIL_OUTPUT,
        "soil_moisture"
    ),
    (
        NDVI_OUTPUT,
        "ndvi"
    ),
]


for file, name in outputs:

    print(
        f"\n{name}:"
    )

    with rasterio.open(
        file
    ) as src:

        data = src.read(
            1
        )

        valid = data[
            np.isfinite(data)
        ]

        print(
            f"  CRS       : {src.crs}"
        )

        print(
            f"  Dimensions: "
            f"{src.width} x {src.height}"
        )

        print(
            f"  Resolution: "
            f"{src.res}"
        )

        print(
            f"  Valid cells: "
            f"{len(valid):,}"
        )

        print(
            f"  Min       : "
            f"{np.min(valid):.6f}"
        )

        print(
            f"  Max       : "
            f"{np.max(valid):.6f}"
        )

        print(
            f"  Mean      : "
            f"{np.mean(valid):.6f}"
        )

        print(
            f"  Median    : "
            f"{np.median(valid):.6f}"
        )


# ================================================================
# 12. VALIDATION
# ================================================================

print("\n[FINAL VALIDATION]")
print("-" * 70)


checks = {}


for file, name in outputs:

    with rasterio.open(
        file
    ) as src:

        checks[
            f"{name} exists"
        ] = file.exists()

        checks[
            f"{name} CRS"
        ] = (
            str(src.crs)
            == TARGET_CRS
        )

        checks[
            f"{name} dimensions"
        ] = (
            src.width
            ==
            target_width
            and
            src.height
            ==
            target_height
        )

        checks[
            f"{name} resolution"
        ] = (
            abs(
                src.res[0]
                -
                TARGET_RES
            )
            < 1e-6
            and
            abs(
                src.res[1]
                -
                TARGET_RES
            )
            < 1e-6
        )

        checks[
            f"{name} alignment"
        ] = (
            src.transform
            ==
            target_transform
        )


# Check rainfall values.
with rasterio.open(
    RAINFALL_OUTPUT
) as src:

    rain = src.read(1)

rain_valid = rain[
    np.isfinite(rain)
]

checks[
    "Rainfall non-negative"
] = (
    np.min(rain_valid)
    >=
    -0.001
)


# Check soil moisture.
with rasterio.open(
    SOIL_OUTPUT
) as src:

    soil = src.read(1)

soil_valid = soil[
    np.isfinite(soil)
]

checks[
    "Soil moisture range"
] = (
    np.min(soil_valid)
    >=
    -0.001
    and
    np.max(soil_valid)
    <=
    1.001
)


# Check NDVI.
with rasterio.open(
    NDVI_OUTPUT
) as src:

    ndvi = src.read(1)

ndvi_valid = ndvi[
    np.isfinite(ndvi)
]

checks[
    "NDVI range"
] = (
    np.min(ndvi_valid)
    >=
    -1.001
    and
    np.max(ndvi_valid)
    <=
    1.001
)


for name, result in checks.items():

    status = (
        "PASSED"
        if result
        else
        "FAILED"
    )

    print(
        f"  {name:<38}: "
        f"{status}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "One or more Step 64C validation "
        "checks failed."
    )


# ================================================================
# 13. COMPLETION
# ================================================================

print("\n" + "=" * 70)
print("STEP 64C COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCreated:")

print(
    "  processed\\predictors\\rainfall_3day_250m.tif"
)

print(
    "  processed\\predictors\\soil_moisture_3day_250m.tif"
)

print(
    "  processed\\predictors\\ndvi_250m.tif"
)

print("\nCommon spatial framework:")

print(
    "  CRS        : EPSG:6933"
)

print(
    "  Resolution : 250 m x 250 m"
)

print(
    f"  Dimensions : "
    f"{target_width:,} x "
    f"{target_height:,}"
)

print("\nTemporal interpretation:")

print(
    "  Rainfall     : 2023-06-01 to 2023-06-03"
)

print(
    "  Soil moisture: 2023-06-01 to 2023-06-03"
)

print(
    "  NDVI         : MOD13Q1 May 25–June 9 composite"
)

print("\nImportant:")

print(
    "  Resampling does not create finer native "
    "information in coarse datasets."
)

print("=" * 70)