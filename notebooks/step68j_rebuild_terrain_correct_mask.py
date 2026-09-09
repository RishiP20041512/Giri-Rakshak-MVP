from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.features import geometry_mask
import geopandas as gpd
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

MASK_FILE = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

ELEVATION_OUT = (
    PROJECT
    / "processed"
    / "predictors"
    / "elevation_250m.tif"
)

SLOPE_OUT = (
    PROJECT
    / "processed"
    / "predictors"
    / "slope_250m.tif"
)


print("=" * 75)
print("STEP 68J — REBUILD ELEVATION AND SLOPE")
print("USING CORRECTED NER MASK")
print("=" * 75)


# ============================================================
# LOAD FINAL GRID + MASK
# ============================================================

print("\nLoading corrected NER grid...")

with rasterio.open(MASK_FILE) as src:

    mask_arr = src.read(1)

    dst_crs = src.crs
    dst_transform = src.transform
    dst_width = src.width
    dst_height = src.height

    dst_profile = src.profile.copy()

print(
    f"Grid CRS: {dst_crs}"
)

print(
    f"Grid size: "
    f"{dst_width} x {dst_height}"
)

print(
    f"Resolution: "
    f"{dst_transform.a} x "
    f"{abs(dst_transform.e)} m"
)

ner_cells = int(
    np.sum(mask_arr == 1)
)

print(
    f"NER cells: "
    f"{ner_cells:,}"
)


# ============================================================
# READ SOURCE DEM
# ============================================================

print("\nReading source DEM...")

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
        f"Source bounds: "
        f"{src.bounds}"
    )

    dem = src.read(
        1,
        masked=True
    )

    src_transform = src.transform
    src_crs = src.crs

    source_nodata = src.nodata

    print(
        f"Source NoData: "
        f"{source_nodata}"
    )


# ============================================================
# PREPARE DESTINATION
# ============================================================

print("\nReprojecting DEM to final 250 m grid...")

elevation = np.full(
    (
        dst_height,
        dst_width
    ),
    np.nan,
    dtype=np.float32
)


# ============================================================
# REPROJECT DEM
# ============================================================

reproject(
    source=dem.filled(np.nan).astype(np.float32),
    destination=elevation,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=dst_transform,
    dst_crs=dst_crs,
    resampling=Resampling.bilinear,
    src_nodata=np.nan,
    dst_nodata=np.nan
)


# ============================================================
# APPLY NER MASK
# ============================================================

elevation[
    mask_arr != 1
] = np.nan


valid_elevation = np.isfinite(
    elevation
)

valid_ner = (
    mask_arr == 1
)

valid_count = int(
    np.sum(
        valid_elevation
        & valid_ner
    )
)

coverage = (
    valid_count
    / ner_cells
    * 100
)

print("\n" + "=" * 75)
print("ELEVATION QA")
print("=" * 75)

print(
    f"Valid elevation cells: "
    f"{valid_count:,}"
)

print(
    f"NER cells: "
    f"{ner_cells:,}"
)

print(
    f"Coverage: "
    f"{coverage:.2f}%"
)

if valid_count > 0:

    vals = elevation[
        valid_elevation
    ]

    print(
        f"Min: "
        f"{np.min(vals):.3f}"
    )

    print(
        f"Max: "
        f"{np.max(vals):.3f}"
    )

    print(
        f"Mean: "
        f"{np.mean(vals):.3f}"
    )

    print(
        f"Median: "
        f"{np.median(vals):.3f}"
    )


# ============================================================
# CALCULATE SLOPE
# ============================================================

print("\nCalculating slope...")

# Pixel size in metres
pixel_size = abs(
    dst_transform.a
)

# Fill small NaN gaps temporarily
elev_for_slope = elevation.copy()

valid = np.isfinite(
    elev_for_slope
)

if not np.any(valid):

    raise RuntimeError(
        "No valid elevation cells available."
    )

# Nearest-style fill using mean only as a temporary
# fallback for gradient calculation.
# The final slope is masked back to the valid DEM.
mean_elev = np.nanmean(
    elev_for_slope
)

elev_for_slope[
    ~valid
] = mean_elev


dzdx = sobel(
    elev_for_slope,
    axis=1,
    mode="nearest"
) / (
    8.0 * pixel_size
)

dzdy = sobel(
    elev_for_slope,
    axis=0,
    mode="nearest"
) / (
    8.0 * pixel_size
)

slope = np.degrees(
    np.arctan(
        np.sqrt(
            dzdx ** 2
            + dzdy ** 2
        )
    )
).astype(
    np.float32
)

slope[
    ~valid_elevation
] = np.nan

slope[
    mask_arr != 1
] = np.nan


# ============================================================
# SLOPE QA
# ============================================================

valid_slope = np.isfinite(
    slope
)

slope_count = int(
    np.sum(
        valid_slope
        & valid_ner
    )
)

slope_coverage = (
    slope_count
    / ner_cells
    * 100
)

print("\n" + "=" * 75)
print("SLOPE QA")
print("=" * 75)

print(
    f"Valid slope cells: "
    f"{slope_count:,}"
)

print(
    f"NER cells: "
    f"{ner_cells:,}"
)

print(
    f"Coverage: "
    f"{slope_coverage:.2f}%"
)

if slope_count > 0:

    vals = slope[
        valid_slope
    ]

    print(
        f"Min: "
        f"{np.min(vals):.3f}°"
    )

    print(
        f"Max: "
        f"{np.max(vals):.3f}°"
    )

    print(
        f"Mean: "
        f"{np.mean(vals):.3f}°"
    )

    print(
        f"Median: "
        f"{np.median(vals):.3f}°"
    )


# ============================================================
# WRITE ELEVATION
# ============================================================

print("\nWriting elevation raster...")

profile = dst_profile.copy()

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=-9999.0,
    compress="deflate",
    BIGTIFF="IF_SAFER"
)

profile.pop(
    "blockxsize",
    None
)

profile.pop(
    "blockysize",
    None
)

elevation_write = np.where(
    np.isfinite(elevation),
    elevation,
    -9999.0
).astype(
    np.float32
)

with rasterio.open(
    ELEVATION_OUT,
    "w",
    **profile
) as dst:

    dst.write(
        elevation_write,
        1
    )


# ============================================================
# WRITE SLOPE
# ============================================================

print(
    "\nWriting slope raster..."
)

slope_write = np.where(
    np.isfinite(slope),
    slope,
    -9999.0
).astype(
    np.float32
)

with rasterio.open(
    SLOPE_OUT,
    "w",
    **profile
) as dst:

    dst.write(
        slope_write,
        1
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n" + "=" * 75)
print("FINAL VERIFICATION")
print("=" * 75)

for filepath in [
    ELEVATION_OUT,
    SLOPE_OUT
]:

    with rasterio.open(
        filepath
    ) as src:

        arr = src.read(
            1
        )

        valid = (
            arr != src.nodata
        )

        print(
            f"\n{filepath.name}"
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
            f"  NoData: "
            f"{src.nodata}"
        )

        print(
            f"  Valid cells: "
            f"{valid.sum():,}"
        )


print("\n" + "=" * 75)
print("STEP 68J COMPLETE")
print("=" * 75)