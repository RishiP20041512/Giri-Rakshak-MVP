"""
STEP 64B — PREPARE ELEVATION AND SLOPE
=======================================

Purpose
-------
Reproject and resample the DEM and slope layers onto the
final common 250 m NER analytical grid.

Input
-----
raw_data\dem_ner\dem_ner.tif
raw_data\dem_ner\slope_ner.tif

Reference grid
--------------
processed\step64a_final_ner_grid.tif

Final CRS
---------
EPSG:6933

Final resolution
----------------
250 m x 250 m

Outputs
-------
processed\predictors\elevation_250m.tif
processed\predictors\slope_250m.tif

Important
---------
This step does not create new terrain information.
The original DEM is approximately 30 m, but the final
analytical grid is 250 m.
"""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEM_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "dem_ner"
    / "dem_ner.tif"
)

SLOPE_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "dem_ner"
    / "slope_ner.tif"
)

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

PREDICTOR_DIR = (
    PROJECT_ROOT
    / "processed"
    / "predictors"
)

ELEVATION_OUTPUT = (
    PREDICTOR_DIR
    / "elevation_250m.tif"
)

SLOPE_OUTPUT = (
    PREDICTOR_DIR
    / "slope_250m.tif"
)


# ================================================================
# 2. HEADER
# ================================================================

print("=" * 70)
print("STEP 64B — PREPARE ELEVATION + SLOPE")
print("=" * 70)


# ================================================================
# 3. CHECK INPUT FILES
# ================================================================

print("\n[1] Checking input files...")
print("-" * 70)

for path in [
    DEM_FILE,
    SLOPE_FILE,
    GRID_FILE,
    MASK_FILE,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(
        "  Found:",
        path.relative_to(PROJECT_ROOT)
    )


# ================================================================
# 4. READ FINAL GRID
# ================================================================

print("\n[2] Reading final analytical grid...")
print("-" * 70)

with rasterio.open(GRID_FILE) as grid_src:

    target_crs = grid_src.crs
    target_transform = grid_src.transform
    target_width = grid_src.width
    target_height = grid_src.height
    target_res = grid_src.res

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


# ================================================================
# 5. VALIDATE GRID
# ================================================================

if str(target_crs) != "EPSG:6933":

    raise ValueError(
        f"Unexpected target CRS: {target_crs}"
    )

if not (
    abs(target_res[0] - 250.0) < 1e-6
    and
    abs(target_res[1] - 250.0) < 1e-6
):

    raise ValueError(
        f"Expected 250 m grid, found {target_res}"
    )


# ================================================================
# 6. READ NER MASK
# ================================================================

print("\n[3] Reading NER mask...")
print("-" * 70)

with rasterio.open(MASK_FILE) as mask_src:

    mask = mask_src.read(1)

    mask_crs = mask_src.crs
    mask_transform = mask_src.transform

if mask.shape != (
    target_height,
    target_width
):

    raise ValueError(
        "NER mask dimensions do not match "
        "the final grid."
    )

if mask_crs != target_crs:

    raise ValueError(
        "NER mask CRS does not match "
        "the final grid."
    )

inside_count = int(
    np.sum(mask == 1)
)

print(
    f"  NER cells: {inside_count:,}"
)


# ================================================================
# 7. PREPARE OUTPUT DIRECTORY
# ================================================================

PREDICTOR_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# 8. FUNCTION FOR REPROJECTING A RASTER
# ================================================================

def prepare_raster(
    input_file,
    output_file,
    variable_name,
):
    """
    Reproject and resample one raster to the
    final 250 m EPSG:6933 grid.
    """

    print(
        f"\nPreparing {variable_name}..."
    )

    print("-" * 70)

    with rasterio.open(
        input_file
    ) as src:

        print(
            f"  Source CRS      : "
            f"{src.crs}"
        )

        print(
            f"  Source size     : "
            f"{src.width} x {src.height}"
        )

        print(
            f"  Source resolution: "
            f"{src.res}"
        )

        print(
            f"  Source nodata   : "
            f"{src.nodata}"
        )

        source = src.read(
            1
        ).astype(
            "float32"
        )

        source_nodata = src.nodata

        # --------------------------------------------------------
        # Allocate target raster
        # --------------------------------------------------------

        destination = np.full(
            (
                target_height,
                target_width
            ),
            np.nan,
            dtype="float32"
        )

        # --------------------------------------------------------
        # Reproject / resample
        # --------------------------------------------------------

        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=source_nodata,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    # ------------------------------------------------------------
    # Apply NER mask
    # ------------------------------------------------------------

    destination[
        mask != 1
    ] = np.nan

    # ------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------

    valid = destination[
        np.isfinite(destination)
    ]

    if len(valid) == 0:

        raise ValueError(
            f"No valid values produced for "
            f"{variable_name}."
        )

    print(
        f"  Valid cells      : "
        f"{len(valid):,}"
    )

    print(
        f"  Minimum          : "
        f"{np.min(valid):.6f}"
    )

    print(
        f"  Maximum          : "
        f"{np.max(valid):.6f}"
    )

    print(
        f"  Mean             : "
        f"{np.mean(valid):.6f}"
    )

    print(
        f"  Median           : "
        f"{np.median(valid):.6f}"
    )

    # ------------------------------------------------------------
    # Output profile
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------

    with rasterio.open(
        output_file,
        "w",
        **profile
    ) as dst:

        dst.write(
            destination,
            1
        )

        dst.set_band_description(
            1,
            variable_name
        )

    print(
        f"  Saved: "
        f"{output_file.relative_to(PROJECT_ROOT)}"
    )


# ================================================================
# 9. PREPARE ELEVATION
# ================================================================

prepare_raster(
    input_file=DEM_FILE,
    output_file=ELEVATION_OUTPUT,
    variable_name="elevation_m",
)


# ================================================================
# 10. PREPARE SLOPE
# ================================================================

prepare_raster(
    input_file=SLOPE_FILE,
    output_file=SLOPE_OUTPUT,
    variable_name="slope_deg",
)


# ================================================================
# 11. VERIFY OUTPUTS
# ================================================================

print("\n[4] Verifying outputs...")
print("-" * 70)

for output_file, variable_name in [
    (
        ELEVATION_OUTPUT,
        "elevation_m"
    ),
    (
        SLOPE_OUTPUT,
        "slope_deg"
    ),
]:

    if not output_file.exists():

        raise RuntimeError(
            f"Output was not created:\n"
            f"{output_file}"
        )

    with rasterio.open(
        output_file
    ) as src:

        print(
            f"\n  {variable_name}"
        )

        print(
            f"    CRS       : "
            f"{src.crs}"
        )

        print(
            f"    Dimensions: "
            f"{src.width} x {src.height}"
        )

        print(
            f"    Resolution: "
            f"{src.res}"
        )

        print(
            f"    Nodata    : "
            f"{src.nodata}"
        )

        data = src.read(
            1
        )

        valid = data[
            np.isfinite(data)
        ]

        print(
            f"    Valid cells: "
            f"{len(valid):,}"
        )

        # --------------------------------------------------------
        # Checks
        # --------------------------------------------------------

        if src.crs != target_crs:

            raise RuntimeError(
                f"{variable_name}: CRS mismatch."
            )

        if (
            src.width != target_width
            or
            src.height != target_height
        ):

            raise RuntimeError(
                f"{variable_name}: "
                f"dimension mismatch."
            )

        if not (
            abs(src.res[0] - 250.0) < 1e-6
            and
            abs(src.res[1] - 250.0) < 1e-6
        ):

            raise RuntimeError(
                f"{variable_name}: "
                f"resolution mismatch."
            )

        if src.transform != target_transform:

            raise RuntimeError(
                f"{variable_name}: "
                f"grid alignment mismatch."
            )


# ================================================================
# 12. CHECK TERRAIN VALUES
# ================================================================

print("\n[5] Checking terrain value ranges...")
print("-" * 70)

with rasterio.open(
    ELEVATION_OUTPUT
) as src:

    elevation = src.read(
        1
    )

with rasterio.open(
    SLOPE_OUTPUT
) as src:

    slope = src.read(
        1
    )

elev_valid = elevation[
    np.isfinite(elevation)
]

slope_valid = slope[
    np.isfinite(slope)
]


# Elevation should not be negative for this NER
# study area based on the source DEM.

if np.min(elev_valid) < -100:

    raise ValueError(
        "Unexpectedly low elevation value detected."
    )


# Slope should be between 0 and 90 degrees.

if (
    np.min(slope_valid) < -0.01
    or
    np.max(slope_valid) > 90.01
):

    raise ValueError(
        "Slope values fall outside "
        "the expected 0–90 degree range."
    )

print(
    f"  Elevation range: "
    f"{np.min(elev_valid):.3f} – "
    f"{np.max(elev_valid):.3f} m"
)

print(
    f"  Slope range: "
    f"{np.min(slope_valid):.3f} – "
    f"{np.max(slope_valid):.3f} degrees"
)


# ================================================================
# 13. CHECK GRID ALIGNMENT
# ================================================================

print("\n[6] Checking exact grid alignment...")
print("-" * 70)

with rasterio.open(
    ELEVATION_OUTPUT
) as src:

    elevation_transform = (
        src.transform
    )

with rasterio.open(
    SLOPE_OUTPUT
) as src:

    slope_transform = (
        src.transform
    )

if elevation_transform != target_transform:

    raise RuntimeError(
        "Elevation grid is not aligned "
        "with final grid."
    )

if slope_transform != target_transform:

    raise RuntimeError(
        "Slope grid is not aligned "
        "with final grid."
    )

print(
    "  Elevation alignment: PASSED"
)

print(
    "  Slope alignment    : PASSED"
)


# ================================================================
# 14. FINAL VALIDATION
# ================================================================

print("\n[7] Final validation...")
print("-" * 70)

checks = {
    "Elevation exists":
        ELEVATION_OUTPUT.exists(),

    "Slope exists":
        SLOPE_OUTPUT.exists(),

    "Elevation CRS":
        str(
            rasterio.open(
                ELEVATION_OUTPUT
            ).crs
        ) == "EPSG:6933",

    "Slope CRS":
        str(
            rasterio.open(
                SLOPE_OUTPUT
            ).crs
        ) == "EPSG:6933",

    "Elevation dimensions":
        elevation.shape ==
        (
            target_height,
            target_width
        ),

    "Slope dimensions":
        slope.shape ==
        (
            target_height,
            target_width
        ),

    "Elevation finite values":
        np.isfinite(
            elev_valid
        ).all(),

    "Slope finite values":
        np.isfinite(
            slope_valid
        ).all(),

    "Slope range":
        (
            np.min(slope_valid) >= -0.01
            and
            np.max(slope_valid) <= 90.01
        ),

    "Elevation grid alignment":
        elevation_transform ==
        target_transform,

    "Slope grid alignment":
        slope_transform ==
        target_transform,
}


for name, result in checks.items():

    status = (
        "PASSED"
        if result
        else
        "FAILED"
    )

    print(
        f"  {name:<32}: "
        f"{status}"
    )


if not all(
    checks.values()
):

    raise RuntimeError(
        "One or more Step 64B validation "
        "checks failed."
    )


# ================================================================
# 15. COMPLETION
# ================================================================

print("\n" + "=" * 70)
print("STEP 64B COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCreated:")

print(
    "  processed\\predictors\\elevation_250m.tif"
)

print(
    "  processed\\predictors\\slope_250m.tif"
)

print("\nCommon spatial properties:")

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

print("\nImportant:")
print(
    "  The 250 m grid does not increase "
    "the native information content of "
    "the DEM or other predictors."
)

print("=" * 70)