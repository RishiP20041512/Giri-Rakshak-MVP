"""
STEP 64C COORDINATE DIAGNOSTIC
==============================

Diagnoses the spatial coordinate ordering of:
    - IMERG rainfall
    - SMAP soil moisture

No files are modified.
"""

from pathlib import Path

import numpy as np
import netCDF4
import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[1]

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


print("=" * 70)
print("STEP 64C — COORDINATE DIAGNOSTIC")
print("=" * 70)


# ================================================================
# 1. RAINFALL COORDINATES
# ================================================================

print("\n[1] IMERG RAINFALL COORDINATE CHECK")
print("-" * 70)

rainfall_file = sorted(
    RAINFALL_DIR.glob("*.nc4")
)[0]

print(
    f"File: {rainfall_file.name}"
)

with netCDF4.Dataset(
    rainfall_file,
    "r"
) as ds:

    lat = np.asarray(
        ds.variables["lat"][:]
    )

    lon = np.asarray(
        ds.variables["lon"][:]
    )

    precip = np.asarray(
        ds.variables["precipitation"][0]
    )

print(
    f"\nLatitude:"
)

print(
    f"  Number: {len(lat)}"
)

print(
    f"  First 5: {lat[:5]}"
)

print(
    f"  Last 5 : {lat[-5:]}"
)

print(
    f"  Minimum: {lat.min()}"
)

print(
    f"  Maximum: {lat.max()}"
)

print(
    f"  First difference: "
    f"{lat[1] - lat[0]}"
)

print(
    f"\nLongitude:"
)

print(
    f"  Number: {len(lon)}"
)

print(
    f"  First 5: {lon[:5]}"
)

print(
    f"  Last 5 : {lon[-5:]}"
)

print(
    f"  Minimum: {lon.min()}"
)

print(
    f"  Maximum: {lon.max()}"
)

print(
    f"  First difference: "
    f"{lon[1] - lon[0]}"
)

print(
    f"\nPrecipitation array:"
)

print(
    f"  Shape: {precip.shape}"
)

print(
    f"  Expected lat x lon: "
    f"({len(lat)}, {len(lon)})"
)

print(
    f"  Shape matches lat x lon: "
    f"{precip.shape == (len(lat), len(lon))}"
)

print(
    "\nIMERG coordinate orientation:"
)

if lat[1] > lat[0]:
    print(
        "  Latitude increases SOUTH -> NORTH"
    )
elif lat[1] < lat[0]:
    print(
        "  Latitude decreases NORTH -> SOUTH"
    )
else:
    print(
        "  WARNING: latitude coordinates "
        "are not changing."
    )

if lon[1] > lon[0]:
    print(
        "  Longitude increases WEST -> EAST"
    )
elif lon[1] < lon[0]:
    print(
        "  Longitude decreases EAST -> WEST"
    )
else:
    print(
        "  WARNING: longitude coordinates "
        "are not changing."
    )


# ================================================================
# 2. CHECK NER RAINFALL REGION
# ================================================================

print("\n[2] IMERG NER REGION CHECK")
print("-" * 70)

NER_WEST = 88.0
NER_EAST = 97.5
NER_SOUTH = 21.5
NER_NORTH = 29.5

lon_inside = (
    (lon >= NER_WEST)
    &
    (lon <= NER_EAST)
)

lat_inside = (
    (lat >= NER_SOUTH)
    &
    (lat <= NER_NORTH)
)

print(
    f"  NER longitude points: "
    f"{np.sum(lon_inside):,}"
)

print(
    f"  NER latitude points : "
    f"{np.sum(lat_inside):,}"
)

if np.any(lon_inside):

    print(
        f"  NER longitude range: "
        f"{lon[lon_inside].min():.6f} "
        f"to "
        f"{lon[lon_inside].max():.6f}"
    )

if np.any(lat_inside):

    print(
        f"  NER latitude range: "
        f"{lat[lat_inside].min():.6f} "
        f"to "
        f"{lat[lat_inside].max():.6f}"
    )


# ================================================================
# 3. SOIL MOISTURE COORDINATES
# ================================================================

print("\n[3] SMAP SOIL-MOISTURE COORDINATE CHECK")
print("-" * 70)

soil_file = sorted(
    list(
        SOIL_DIR.glob("*.h5")
    )
)[0]

print(
    f"File: {soil_file.name}"
)

with h5py.File(
    soil_file,
    "r"
) as h5:

    group = h5[
        "Soil_Moisture_Retrieval_Data_AM"
    ]

    sm = np.asarray(
        group[
            "soil_moisture"
        ][:]
    )

    sm_lat = np.asarray(
        group[
            "latitude"
        ][:]
    )

    sm_lon = np.asarray(
        group[
            "longitude"
        ][:]
    )

print(
    f"\nSoil moisture:"
)

print(
    f"  Shape: {sm.shape}"
)

print(
    f"  Min: {np.nanmin(sm):.6f}"
)

print(
    f"  Max: {np.nanmax(sm):.6f}"
)

print(
    f"\nSMAP latitude:"
)

print(
    f"  Shape: {sm_lat.shape}"
)

print(
    f"  Minimum: "
    f"{np.nanmin(sm_lat):.6f}"
)

print(
    f"  Maximum: "
    f"{np.nanmax(sm_lat):.6f}"
)

print(
    f"  Top-left: "
    f"{sm_lat[0, 0]:.6f}"
)

print(
    f"  Bottom-left: "
    f"{sm_lat[-1, 0]:.6f}"
)

print(
    f"  Top-right: "
    f"{sm_lat[0, -1]:.6f}"
)

print(
    f"  Bottom-right: "
    f"{sm_lat[-1, -1]:.6f}"
)

print(
    f"\nSMAP longitude:"
)

print(
    f"  Shape: {sm_lon.shape}"
)

print(
    f"  Minimum: "
    f"{np.nanmin(sm_lon):.6f}"
)

print(
    f"  Maximum: "
    f"{np.nanmax(sm_lon):.6f}"
)

print(
    f"  Top-left: "
    f"{sm_lon[0, 0]:.6f}"
)

print(
    f"  Bottom-left: "
    f"{sm_lon[-1, 0]:.6f}"
)

print(
    f"  Top-right: "
    f"{sm_lon[0, -1]:.6f}"
)

print(
    f"  Bottom-right: "
    f"{sm_lon[-1, -1]:.6f}"
)


# ================================================================
# 4. SMAP NER COVERAGE
# ================================================================

print("\n[4] SMAP NER COVERAGE")
print("-" * 70)

valid_sm = (
    np.isfinite(sm)
    &
    np.isfinite(sm_lat)
    &
    np.isfinite(sm_lon)
)

ner_sm = (
    valid_sm
    &
    (sm_lon >= NER_WEST)
    &
    (sm_lon <= NER_EAST)
    &
    (sm_lat >= NER_SOUTH)
    &
    (sm_lat <= NER_NORTH)
)

print(
    f"  Total valid SMAP observations: "
    f"{np.sum(valid_sm):,}"
)

print(
    f"  Valid observations inside NER: "
    f"{np.sum(ner_sm):,}"
)

if np.any(ner_sm):

    print(
        f"  NER SMAP latitude range: "
        f"{sm_lat[ner_sm].min():.6f} "
        f"to "
        f"{sm_lat[ner_sm].max():.6f}"
    )

    print(
        f"  NER SMAP longitude range: "
        f"{sm_lon[ner_sm].min():.6f} "
        f"to "
        f"{sm_lon[ner_sm].max():.6f}"
    )


# ================================================================
# 5. SAMPLE SMAP GRID SPACING
# ================================================================

print("\n[5] SMAP GRID SPACING DIAGNOSTIC")
print("-" * 70)

# Examine differences along rows and columns.

lat_row_diff = (
    sm_lat[:, 1:]
    -
    sm_lat[:, :-1]
)

lon_row_diff = (
    sm_lon[:, 1:]
    -
    sm_lon[:, :-1]
)

lat_col_diff = (
    sm_lat[1:, :]
    -
    sm_lat[:-1, :]
)

lon_col_diff = (
    sm_lon[1:, :]
    -
    sm_lon[:-1, :]
)


def finite_stats(
    array,
    name
):

    values = array[
        np.isfinite(array)
    ]

    if len(values) == 0:

        print(
            f"  {name}: no finite values"
        )

        return

    print(
        f"  {name}:"
    )

    print(
        f"    min    = {values.min():.8f}"
    )

    print(
        f"    max    = {values.max():.8f}"
    )

    print(
        f"    median = {np.median(values):.8f}"
    )


finite_stats(
    lat_row_diff,
    "Latitude difference across columns"
)

finite_stats(
    lon_row_diff,
    "Longitude difference across columns"
)

finite_stats(
    lat_col_diff,
    "Latitude difference down rows"
)

finite_stats(
    lon_col_diff,
    "Longitude difference down rows"
)


# ================================================================
# 6. FINAL
# ================================================================

print("\n" + "=" * 70)
print("STEP 64C COORDINATE DIAGNOSTIC COMPLETED")
print("=" * 70)

print(
    "\nNo source files were modified."
)

print(
    "Use this output to determine the exact coordinate "
    "orientation before rebuilding rainfall and soil moisture."
)

print("=" * 70)