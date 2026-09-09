from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_RAINFALL = (
    ROOT
    / "processed"
    / "predictors"
    / "rainfall_3day_250m_raw.tif"
)

FINAL_GRID = (
    ROOT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

OUTPUT = (
    ROOT
    / "processed"
    / "predictors"
    / "rainfall_3day_250m.tif"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 68I — ALIGN REAL CHIRPS RAINFALL TO FINAL NER GRID")
print("=" * 70)

print("\nChecking required files...")

if not RAW_RAINFALL.exists():
    raise FileNotFoundError(
        f"Raw rainfall file not found:\n{RAW_RAINFALL}"
    )

if not FINAL_GRID.exists():
    raise FileNotFoundError(
        f"Final grid not found:\n{FINAL_GRID}"
    )

print("OK: Raw CHIRPS rainfall exists.")
print("OK: Final NER grid exists.")


# ============================================================
# READ FINAL GRID
# ============================================================

print("\nReading final NER grid...")

with rasterio.open(FINAL_GRID) as grid:

    target_crs = grid.crs
    target_transform = grid.transform
    target_width = grid.width
    target_height = grid.height

print("Target CRS:", target_crs)
print("Target width:", target_width)
print("Target height:", target_height)
print("Target resolution: 250 m")


# ============================================================
# READ RAINFALL
# ============================================================

print("\nReading raw CHIRPS rainfall...")

with rasterio.open(RAW_RAINFALL) as src:

    print("Source CRS:", src.crs)
    print("Source dimensions:", src.width, "x", src.height)
    print("Source resolution:", src.res)

    rainfall = src.read(1).astype(np.float32)

    source_transform = src.transform
    source_crs = src.crs
    source_nodata = src.nodata


# ============================================================
# PREPARE OUTPUT ARRAY
# ============================================================

print("\nReprojecting and resampling rainfall...")

aligned = np.full(
    (target_height, target_width),
    -9999.0,
    dtype=np.float32
)


# ============================================================
# REPROJECT
# ============================================================

reproject(
    source=rainfall,
    destination=aligned,

    src_transform=source_transform,
    src_crs=source_crs,

    dst_transform=target_transform,
    dst_crs=target_crs,

    src_nodata=source_nodata,
    dst_nodata=-9999.0,

    resampling=Resampling.average
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving aligned rainfall raster...")

profile = {
    "driver": "GTiff",
    "height": target_height,
    "width": target_width,
    "count": 1,
    "dtype": "float32",
    "crs": target_crs,
    "transform": target_transform,
    "nodata": -9999.0,
    "compress": "deflate",
    "predictor": 2,
}


with rasterio.open(OUTPUT, "w", **profile) as dst:
    dst.write(aligned, 1)


# ============================================================
# QA
# ============================================================

print("\nRunning final QA...")

with rasterio.open(OUTPUT) as check:

    data = check.read(1)

    valid = data[data != -9999.0]

    print("Output CRS:", check.crs)
    print("Output dimensions:", check.width, "x", check.height)
    print("Output resolution:", check.res)
    print("Valid pixels:", len(valid))

    if len(valid) > 0:
        print("Minimum rainfall:", float(valid.min()), "mm")
        print("Maximum rainfall:", float(valid.max()), "mm")
        print("Mean rainfall:", float(valid.mean()), "mm")
        print("Median rainfall:", float(np.median(valid)), "mm")


# ============================================================
# FINAL CHECKS
# ============================================================

with rasterio.open(OUTPUT) as check:

    assert check.crs == target_crs
    assert check.width == target_width
    assert check.height == target_height

    assert abs(check.res[0] - 250.0) < 0.01
    assert abs(check.res[1] - 250.0) < 0.01

print("\n" + "=" * 70)
print("SUCCESS: RAINFALL IS ALIGNED TO THE FINAL 250 m NER GRID")
print("=" * 70)

print("\nFinal output:")
print(OUTPUT)

print("\nSource:")
print("UCSB-CHG/CHIRPS/DAILY")

print("Variable:")
print("3-day accumulated precipitation")

print("Period:")
print("2026-07-29 to 2026-07-31")

print("Units:")
print("mm")

print("=" * 70)