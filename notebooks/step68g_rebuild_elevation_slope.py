from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import array_bounds
from scipy.ndimage import sobel


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

DEM = (
    PROJECT
    / "raw_data"
    / "dem_ner"
    / "dem_ner.tif"
)

FINAL_GRID = (
    PROJECT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

NER_MASK = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

OUTPUT_DIR = (
    PROJECT
    / "processed"
    / "predictors"
)

ELEVATION_OUT = (
    OUTPUT_DIR
    / "elevation_250m.tif"
)

SLOPE_OUT = (
    OUTPUT_DIR
    / "slope_250m.tif"
)


print("=" * 75)
print("STEP 68G — REBUILD ELEVATION AND SLOPE")
print("=" * 75)


# ============================================================
# CHECK INPUTS
# ============================================================

for path in [
    DEM,
    FINAL_GRID,
    NER_MASK
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )


# ============================================================
# LOAD FINAL GRID
# ============================================================

print("\nLoading final grid...")

with rasterio.open(FINAL_GRID) as grid:

    dst_crs = grid.crs
    dst_transform = grid.transform
    dst_width = grid.width
    dst_height = grid.height

    print(f"CRS: {dst_crs}")
    print(
        f"Dimensions: "
        f"{dst_width} x {dst_height}"
    )

    print(
        f"Resolution: "
        f"{grid.res}"
    )


# ============================================================
# LOAD NER MASK
# ============================================================

print("\nLoading NER mask...")

with rasterio.open(NER_MASK) as mask_src:

    ner_mask = (
        mask_src.read(1) > 0
    )

    if (
        mask_src.width != dst_width
        or mask_src.height != dst_height
    ):

        raise ValueError(
            "NER mask does not match final grid."
        )

print(
    f"NER valid cells: "
    f"{ner_mask.sum():,}"
)


# ============================================================
# REPROJECT DEM DIRECTLY TO FINAL GRID
# ============================================================

print("\n" + "=" * 75)
print("REPROJECTING DEM DIRECTLY TO FINAL GRID")
print("=" * 75)

with rasterio.open(DEM) as src:

    print(
        f"Source CRS: {src.crs}"
    )

    print(
        f"Source size: "
        f"{src.width} x {src.height}"
    )

    print(
        f"Source resolution: "
        f"{src.res}"
    )

    print(
        f"Source NoData: "
        f"{src.nodata}"
    )

    destination = np.full(
        (dst_height, dst_width),
        np.nan,
        dtype=np.float32
    )

    reproject(
        source=rasterio.band(src, 1),
        destination=destination,

        src_transform=src.transform,
        src_crs=src.crs,

        dst_transform=dst_transform,
        dst_crs=dst_crs,

        src_nodata=None,
        dst_nodata=np.nan,

        resampling=Resampling.bilinear
    )


# ============================================================
# APPLY NER MASK
# ============================================================

destination[
    ~ner_mask
] = np.nan


valid = np.isfinite(destination)

print(
    f"\nElevation valid cells: "
    f"{valid.sum():,}"
)

print(
    f"Elevation NER coverage: "
    f"{valid.sum() / ner_mask.sum() * 100:.2f}%"
)

print(
    f"Elevation min: "
    f"{np.nanmin(destination):.3f}"
)

print(
    f"Elevation max: "
    f"{np.nanmax(destination):.3f}"
)

print(
    f"Elevation mean: "
    f"{np.nanmean(destination):.3f}"
)

print(
    f"Elevation median: "
    f"{np.nanmedian(destination):.3f}"
)


# ============================================================
# WRITE ELEVATION
# ============================================================

print("\nWriting elevation raster...")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

elevation_profile = {
    "driver": "GTiff",
    "height": dst_height,
    "width": dst_width,
    "count": 1,
    "dtype": "float32",
    "crs": dst_crs,
    "transform": dst_transform,
    "nodata": -9999.0,
    "compress": "deflate",
    "predictor": 3,
    "tiled": False,
    "BIGTIFF": "IF_SAFER",
}


elevation_output = np.where(
    np.isfinite(destination),
    destination,
    -9999.0
).astype(np.float32)


with rasterio.open(
    ELEVATION_OUT,
    "w",
    **elevation_profile
) as dst:

    dst.write(
        elevation_output,
        1
    )


print(
    f"Saved:\n{ELEVATION_OUT}"
)


# ============================================================
# CALCULATE SLOPE
# ============================================================

print("\n" + "=" * 75)
print("CALCULATING SLOPE")
print("=" * 75)

elevation = destination.copy()

# ------------------------------------------------------------
# Fill temporary NoData areas only for gradient calculation.
# Since the DEM is complete, this should normally be unnecessary,
# but protects against edge effects.
# ------------------------------------------------------------

valid_elev = np.isfinite(
    elevation
)

if valid_elev.sum() == 0:

    raise ValueError(
        "No valid elevation cells available."
    )


# ------------------------------------------------------------
# Calculate gradients
#
# Final grid is projected and has 250 m cells.
# ------------------------------------------------------------

dx = 250.0
dy = 250.0

# Replace NaN temporarily.
# These should only occur outside the NER mask.
work = elevation.copy()

work[
    ~np.isfinite(work)
] = 0.0


# Sobel gives the sum over a 3x3 neighborhood.
# Divide by 8 to approximate central difference.
grad_x = (
    sobel(
        work,
        axis=1,
        mode="nearest"
    )
    / (8.0 * dx)
)

grad_y = (
    sobel(
        work,
        axis=0,
        mode="nearest"
    )
    / (8.0 * dy)
)


slope = np.degrees(
    np.arctan(
        np.sqrt(
            grad_x ** 2
            +
            grad_y ** 2
        )
    )
)


# Restore NoData.
slope[
    ~valid_elev
] = np.nan

slope[
    ~ner_mask
] = np.nan


valid_slope = np.isfinite(
    slope
)

print(
    f"Slope valid cells: "
    f"{valid_slope.sum():,}"
)

print(
    f"Slope NER coverage: "
    f"{valid_slope.sum() / ner_mask.sum() * 100:.2f}%"
)

print(
    f"Slope min: "
    f"{np.nanmin(slope):.3f}"
)

print(
    f"Slope max: "
    f"{np.nanmax(slope):.3f}"
)

print(
    f"Slope mean: "
    f"{np.nanmean(slope):.3f}"
)

print(
    f"Slope median: "
    f"{np.nanmedian(slope):.3f}"
)


# ============================================================
# WRITE SLOPE
# ============================================================

print("\nWriting slope raster...")

slope_output = np.where(
    np.isfinite(slope),
    slope,
    -9999.0
).astype(np.float32)


with rasterio.open(
    SLOPE_OUT,
    "w",
    **elevation_profile
) as dst:

    dst.write(
        slope_output,
        1
    )


print(
    f"Saved:\n{SLOPE_OUT}"
)


# ============================================================
# FINAL QA
# ============================================================

print("\n" + "=" * 75)
print("FINAL QA")
print("=" * 75)

for name, path in [
    ("Elevation", ELEVATION_OUT),
    ("Slope", SLOPE_OUT)
]:

    with rasterio.open(path) as src:

        arr = src.read(1)

        valid = (
            np.isfinite(arr)
            & (arr != src.nodata)
        )

        valid_ner = (
            valid
            & ner_mask
        )

        print(
            f"\n{name}:"
        )

        print(
            f"  CRS: {src.crs}"
        )

        print(
            f"  Size: "
            f"{src.width} x {src.height}"
        )

        print(
            f"  Resolution: "
            f"{src.res}"
        )

        print(
            f"  Valid NER cells: "
            f"{valid_ner.sum():,}"
        )

        print(
            f"  Coverage: "
            f"{valid_ner.sum() / ner_mask.sum() * 100:.2f}%"
        )


print("\n" + "=" * 75)
print("STEP 68G COMPLETE")
print("=" * 75)