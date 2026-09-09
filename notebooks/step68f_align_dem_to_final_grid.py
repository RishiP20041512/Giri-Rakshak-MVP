from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEM_INPUT = ROOT / "raw_data" / "dem_ner" / "dem_ner.tif"
GRID_FILE = ROOT / "processed" / "step64a_final_ner_grid.tif"

DEM_OUTPUT = ROOT / "raw_data" / "dem_ner" / "dem_ner_250m.tif"


print("=" * 70)
print("ALIGNING REAL NER DEM TO FINAL 250 m GRID")
print("=" * 70)

print(f"Input DEM : {DEM_INPUT}")
print(f"Grid      : {GRID_FILE}")
print(f"Output    : {DEM_OUTPUT}")


# ============================================================
# CHECK INPUTS
# ============================================================

if not DEM_INPUT.exists():
    raise FileNotFoundError(f"DEM not found: {DEM_INPUT}")

if not GRID_FILE.exists():
    raise FileNotFoundError(f"Final grid not found: {GRID_FILE}")


# ============================================================
# READ FINAL GRID
# ============================================================

with rasterio.open(GRID_FILE) as grid_src:

    target_crs = grid_src.crs
    target_transform = grid_src.transform
    target_width = grid_src.width
    target_height = grid_src.height

    print("\nFINAL GRID")
    print("-" * 70)
    print("CRS:", target_crs)
    print("Width:", target_width)
    print("Height:", target_height)
    print("Resolution:", target_transform.a, abs(target_transform.e))


# ============================================================
# READ REAL COPERNICUS DEM
# ============================================================

with rasterio.open(DEM_INPUT) as src:

    print("\nSOURCE DEM")
    print("-" * 70)
    print("CRS:", src.crs)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Resolution:", src.res)
    print("Bounds:", src.bounds)
    print("Nodata:", src.nodata)

    source_nodata = src.nodata

    if source_nodata is None:
        source_nodata = -9999

    # --------------------------------------------------------
    # Allocate exact final-grid array
    # --------------------------------------------------------

    destination = np.full(
        (target_height, target_width),
        -9999,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Reproject / resample
    # --------------------------------------------------------

    print("\nResampling real DEM to exact 250 m grid...")

    reproject(
        source=rasterio.band(src, 1),
        destination=destination,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=source_nodata,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=-9999,
        resampling=Resampling.average
    )


# ============================================================
# CLEAN INVALID VALUES
# ============================================================

valid = np.isfinite(destination) & (destination != -9999)

# Elevation cannot be negative in this study area.
destination[(destination < 0) & valid] = -9999

valid = np.isfinite(destination) & (destination != -9999)


if valid.sum() == 0:
    raise RuntimeError("No valid elevation pixels after resampling.")


# ============================================================
# SAVE
# ============================================================

DEM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

profile = {
    "driver": "GTiff",
    "height": target_height,
    "width": target_width,
    "count": 1,
    "dtype": "float32",
    "crs": target_crs,
    "transform": target_transform,
    "nodata": -9999,
    "compress": "deflate",
    "predictor": 3,
    "tiled": True,
    "BIGTIFF": "IF_SAFER"
}


with rasterio.open(DEM_OUTPUT, "w", **profile) as dst:
    dst.write(destination, 1)


# ============================================================
# FINAL VALIDATION
# ============================================================

with rasterio.open(DEM_OUTPUT) as check:

    data = check.read(1)
    valid_check = np.isfinite(data) & (data != -9999)

    print("\n" + "=" * 70)
    print("250 m NER DEM CREATED")
    print("=" * 70)

    print("File:", DEM_OUTPUT)
    print("CRS:", check.crs)
    print("Width:", check.width)
    print("Height:", check.height)
    print("Resolution:", check.res)
    print("Valid pixels:", int(valid_check.sum()))

    print("Minimum elevation:",
          float(data[valid_check].min()))

    print("Maximum elevation:",
          float(data[valid_check].max()))

    print("=" * 70)

    # Exact grid checks
    assert check.crs == target_crs
    assert check.width == target_width
    assert check.height == target_height
    assert abs(check.res[0] - 250) < 0.01
    assert abs(check.res[1] - 250) < 0.01

print("\nSUCCESS: DEM is exactly aligned to the final 250 m NER grid.")