"""
STEP 64A — CREATE FINAL NER RASTER GRID
=======================================

Purpose
-------
Establish the common CRS, extent, pixel size and grid alignment
for the final Northeast India landslide susceptibility mapping.

Study area
----------
Northeast India:
    Arunachal Pradesh
    Assam
    Manipur
    Meghalaya
    Mizoram
    Nagaland
    Sikkim
    Tripura

Input
-----
raw_data\boundaries\ner_8states.geojson

Final common CRS
----------------
EPSG:6933
WGS 84 / NSIDC EASE-Grid 2.0 Global

Reason
------
The NER study area crosses multiple UTM zones. A single UTM zone
such as EPSG:32645 is therefore not appropriate as the final
study-wide mapping CRS.

EPSG:6933 is a projected metre-based CRS suitable for environmental
gridded analysis.

Final common pixel size
-----------------------
250 m x 250 m

Reason
------
The conditioning factors have substantially different native
resolutions. In particular:
    - Copernicus DEM GLO-30: approximately 30 m
    - MODIS MOD13Q1 NDVI: 250 m
    - SMAP soil moisture: substantially coarser than 250 m
    - Bhuvan thematic datasets: map-scale dependent

A 250 m analytical grid avoids forcing all predictors onto an
artificial 30 m grid.

IMPORTANT:
A 250 m grid does NOT create 250 m information from coarser
datasets. Native-resolution limitations remain and will be
documented.

Outputs
-------
processed\step64a_final_grid_info.csv
processed\step64a_final_ner_grid.tif
processed\step64a_final_ner_mask.tif

The template raster contains:
    1 = inside NER study area
    0 = outside NER study area

This step does NOT:
    - train a model
    - change the final Random Forest
    - create susceptibility values
    - resample predictor datasets yet
"""

from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "boundaries"
    / "ner_8states.geojson"
)

PROCESSED_DIR = (
    PROJECT_ROOT / "processed"
)

GRID_FILE = (
    PROCESSED_DIR
    / "step64a_final_ner_grid.tif"
)

MASK_FILE = (
    PROCESSED_DIR
    / "step64a_final_ner_mask.tif"
)

INFO_FILE = (
    PROCESSED_DIR
    / "step64a_final_grid_info.csv"
)


# ================================================================
# 2. FINAL GRID SETTINGS
# ================================================================

TARGET_CRS = "EPSG:6933"

PIXEL_SIZE = 250.0


# ================================================================
# 3. HEADER
# ================================================================

print("=" * 70)
print("STEP 64A — CREATE FINAL NER RASTER GRID")
print("=" * 70)


# ================================================================
# 4. CHECK INPUT
# ================================================================

print("\n[1] Checking NER boundary...")
print("-" * 70)

if not BOUNDARY_FILE.exists():
    raise FileNotFoundError(
        f"NER boundary not found:\n{BOUNDARY_FILE}"
    )

print(
    "  Found:",
    BOUNDARY_FILE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 5. LOAD BOUNDARY
# ================================================================

print("\n[2] Loading NER boundary...")
print("-" * 70)

ner = gpd.read_file(
    BOUNDARY_FILE
)

print(
    f"  Boundary features: {len(ner)}"
)

print(
    f"  Original CRS      : {ner.crs}"
)

print(
    "  Boundary columns  :",
    list(ner.columns)
)


# ================================================================
# 6. CHECK REQUIRED GEOMETRY
# ================================================================

print("\n[3] Checking boundary geometry...")
print("-" * 70)

if ner.empty:
    raise ValueError(
        "NER boundary file contains no features."
    )

if ner.geometry.isna().any():
    raise ValueError(
        "Boundary contains missing geometries."
    )

if (~ner.geometry.is_valid).any():
    print(
        "  Warning: invalid geometries detected."
    )

    ner["geometry"] = (
        ner.geometry
        .make_valid()
    )

print(
    "  Geometry check completed."
)


# ================================================================
# 7. IDENTIFY NER STATES
# ================================================================

print("\n[4] Checking NER state coverage...")
print("-" * 70)

expected_states = {
    "Arunachal Pradesh",
    "Assam",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura",
}

if "shapeName" in ner.columns:

    available_states = set(
        ner["shapeName"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    print(
        "  States/features in boundary:"
    )

    for state in sorted(
        available_states
    ):
        print(
            f"    {state}"
        )

    missing_states = (
        expected_states
        - available_states
    )

    if missing_states:

        print(
            "\n  Warning: expected states "
            "not found:"
        )

        for state in sorted(
            missing_states
        ):
            print(
                f"    {state}"
            )

else:

    print(
        "  shapeName field not available."
    )

    print(
        "  Proceeding using the supplied "
        "boundary geometry."
    )


# ================================================================
# 8. DISSOLVE NER INTO ONE STUDY AREA
# ================================================================

print("\n[5] Creating unified NER study area...")
print("-" * 70)

ner_union = ner.geometry.union_all()

if ner_union.is_empty:
    raise ValueError(
        "Unified NER geometry is empty."
    )

print(
    "  Unified study-area geometry created."
)


# ================================================================
# 9. CREATE STUDY AREA GEODATAFRAME
# ================================================================

study_area = gpd.GeoDataFrame(
    {
        "study_area": ["Northeast India"]
    },
    geometry=[ner_union],
    crs=ner.crs
)


# ================================================================
# 10. REPROJECT TO FINAL CRS
# ================================================================

print("\n[6] Reprojecting NER to final CRS...")
print("-" * 70)

print(
    f"  Target CRS: {TARGET_CRS}"
)

study_area_projected = (
    study_area
    .to_crs(TARGET_CRS)
)

projected_geometry = (
    study_area_projected
    .geometry
    .iloc[0]
)


# ================================================================
# 11. GET PROJECTED BOUNDS
# ================================================================

minx, miny, maxx, maxy = (
    projected_geometry.bounds
)

print("\n[7] Projected study-area bounds...")
print("-" * 70)

print(
    f"  min X: {minx:.3f} m"
)

print(
    f"  min Y: {miny:.3f} m"
)

print(
    f"  max X: {maxx:.3f} m"
)

print(
    f"  max Y: {maxy:.3f} m"
)


# ================================================================
# 12. SNAP BOUNDS TO 250-M GRID
# ================================================================

print("\n[8] Creating aligned 250 m grid...")
print("-" * 70)

aligned_minx = (
    np.floor(
        minx / PIXEL_SIZE
    )
    * PIXEL_SIZE
)

aligned_miny = (
    np.floor(
        miny / PIXEL_SIZE
    )
    * PIXEL_SIZE
)

aligned_maxx = (
    np.ceil(
        maxx / PIXEL_SIZE
    )
    * PIXEL_SIZE
)

aligned_maxy = (
    np.ceil(
        maxy / PIXEL_SIZE
    )
    * PIXEL_SIZE
)


# ================================================================
# 13. CALCULATE GRID DIMENSIONS
# ================================================================

width = int(
    round(
        (
            aligned_maxx
            - aligned_minx
        )
        / PIXEL_SIZE
    )
)

height = int(
    round(
        (
            aligned_maxy
            - aligned_miny
        )
        / PIXEL_SIZE
    )
)

if width <= 0 or height <= 0:
    raise ValueError(
        "Invalid raster dimensions."
    )

print(
    f"  Pixel size: {PIXEL_SIZE:.1f} m"
)

print(
    f"  Width     : {width:,} pixels"
)

print(
    f"  Height    : {height:,} pixels"
)

print(
    f"  Total cells: {width * height:,}"
)


# ================================================================
# 14. CREATE TRANSFORM
# ================================================================

transform = from_origin(
    aligned_minx,
    aligned_maxy,
    PIXEL_SIZE,
    PIXEL_SIZE
)


# ================================================================
# 15. RASTERIZE NER MASK
# ================================================================

print("\n[9] Rasterizing NER study-area mask...")
print("-" * 70)

mask = rasterize(
    [
        (
            projected_geometry,
            1
        )
    ],
    out_shape=(
        height,
        width
    ),
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=False
)

inside_cells = int(
    np.sum(mask == 1)
)

outside_cells = int(
    np.sum(mask == 0)
)

print(
    f"  Inside-NER cells : "
    f"{inside_cells:,}"
)

print(
    f"  Outside cells    : "
    f"{outside_cells:,}"
)

print(
    f"  Total cells      : "
    f"{width * height:,}"
)


# ================================================================
# 16. CREATE OUTPUT DIRECTORY
# ================================================================

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 17. SAVE GRID TEMPLATE
# ================================================================

print("\n[10] Saving final grid template...")
print("-" * 70)

grid_profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "crs": TARGET_CRS,
    "transform": transform,
    "nodata": 0,
    "compress": "deflate",
}

with rasterio.open(
    GRID_FILE,
    "w",
    **grid_profile
) as dst:

    # Grid ID raster.
    # Every cell gets value 1.
    grid_array = np.ones(
        (
            height,
            width
        ),
        dtype="uint8"
    )

    dst.write(
        grid_array,
        1
    )


print(
    "  Saved:",
    GRID_FILE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 18. SAVE NER MASK
# ================================================================

print("\n[11] Saving NER mask...")
print("-" * 70)

mask_profile = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "uint8",
    "crs": TARGET_CRS,
    "transform": transform,
    "nodata": 0,
    "compress": "deflate",
}

with rasterio.open(
    MASK_FILE,
    "w",
    **mask_profile
) as dst:

    dst.write(
        mask,
        1
    )


print(
    "  Saved:",
    MASK_FILE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 19. SAVE GRID INFORMATION
# ================================================================

print("\n[12] Saving grid metadata...")
print("-" * 70)

info = {
    "parameter": [
        "target_crs",
        "pixel_size_m",
        "width_pixels",
        "height_pixels",
        "total_cells",
        "inside_ner_cells",
        "outside_cells",
        "aligned_min_x_m",
        "aligned_min_y_m",
        "aligned_max_x_m",
        "aligned_max_y_m",
    ],
    "value": [
        TARGET_CRS,
        PIXEL_SIZE,
        width,
        height,
        width * height,
        inside_cells,
        outside_cells,
        aligned_minx,
        aligned_miny,
        aligned_maxx,
        aligned_maxy,
    ]
}

info_df = pd.DataFrame(
    info
)

info_df.to_csv(
    INFO_FILE,
    index=False
)

print(
    "  Saved:",
    INFO_FILE.relative_to(PROJECT_ROOT)
)


# ================================================================
# 20. REOPEN AND VERIFY GRID
# ================================================================

print("\n[13] Reopening outputs for verification...")
print("-" * 70)

with rasterio.open(
    GRID_FILE
) as src:

    grid_crs = str(
        src.crs
    )

    grid_width = src.width
    grid_height = src.height
    grid_res = src.res
    grid_transform = src.transform

with rasterio.open(
    MASK_FILE
) as src:

    mask_crs = str(
        src.crs
    )

    mask_width = src.width
    mask_height = src.height

    mask_values = src.read(
        1
    )


print(
    f"  Grid CRS       : {grid_crs}"
)

print(
    f"  Grid dimensions: "
    f"{grid_width} x {grid_height}"
)

print(
    f"  Grid resolution: "
    f"{grid_res}"
)

print(
    f"  Mask CRS       : {mask_crs}"
)

print(
    f"  Mask dimensions: "
    f"{mask_width} x {mask_height}"
)


# ================================================================
# 21. VALIDATION CHECKS
# ================================================================

print("\n[14] Final validation...")
print("-" * 70)

checks = {}

checks[
    "Target CRS"
] = (
    grid_crs == TARGET_CRS
)

checks[
    "Grid width"
] = (
    grid_width == width
)

checks[
    "Grid height"
] = (
    grid_height == height
)

checks[
    "Pixel width"
] = (
    abs(
        grid_res[0]
        - PIXEL_SIZE
    )
    < 1e-6
)

checks[
    "Pixel height"
] = (
    abs(
        grid_res[1]
        - PIXEL_SIZE
    )
    < 1e-6
)

checks[
    "Mask CRS"
] = (
    mask_crs == TARGET_CRS
)

checks[
    "Mask dimensions"
] = (
    mask_width == width
    and
    mask_height == height
)

checks[
    "Mask binary"
] = (
    set(
        np.unique(
            mask_values
        )
    ).issubset(
        {0, 1}
    )
)

checks[
    "Grid output exists"
] = GRID_FILE.exists()

checks[
    "Mask output exists"
] = MASK_FILE.exists()

checks[
    "Metadata output exists"
] = INFO_FILE.exists()


for name, result in checks.items():

    status = (
        "PASSED"
        if result
        else
        "FAILED"
    )

    print(
        f"  {name:<30} : "
        f"{status}"
    )


if not all(checks.values()):

    raise RuntimeError(
        "One or more Step 64A validation checks failed."
    )


# ================================================================
# 22. COMPLETION
# ================================================================

print("\n" + "=" * 70)
print("STEP 64A COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal mapping CRS:")
print(
    f"  {TARGET_CRS}"
)

print("\nFinal analytical grid:")
print(
    f"  {PIXEL_SIZE:.0f} m x "
    f"{PIXEL_SIZE:.0f} m"
)

print("\nGrid dimensions:")
print(
    f"  {width:,} columns x "
    f"{height:,} rows"
)

print("\nOutputs:")
print(
    "  processed\\step64a_final_ner_grid.tif"
)

print(
    "  processed\\step64a_final_ner_mask.tif"
)

print(
    "  processed\\step64a_final_grid_info.csv"
)

print("\nImportant:")
print(
    "  This grid defines the common spatial framework."
)

print(
    "  It does not increase the native resolution "
    "of any source dataset."
)

print(
    "  Predictor preparation will occur in subsequent "
    "Step 64 stages."
)

print("=" * 70)
