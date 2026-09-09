from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_NDVI = (
    ROOT
    / "processed"
    / "predictors"
    / "ndvi_250m_raw.tif"
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
    / "ndvi_250m.tif"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 68L — ALIGN REAL SENTINEL-2 NDVI")
print("TO FINAL 250 m NER GRID")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

if not RAW_NDVI.exists():
    raise FileNotFoundError(
        f"Raw NDVI not found:\n{RAW_NDVI}"
    )

print("OK: Real Sentinel-2 NDVI exists.")

if not FINAL_GRID.exists():
    raise FileNotFoundError(
        f"Final NER grid not found:\n{FINAL_GRID}"
    )

print("OK: Final NER grid exists.")


# ============================================================
# READ TARGET GRID
# ============================================================

print("\nReading final NER grid...")

with rasterio.open(FINAL_GRID) as target:

    target_crs = target.crs
    target_transform = target.transform
    target_width = target.width
    target_height = target.height

    target_profile = target.profile.copy()

    print("Target CRS:", target_crs)
    print(
        "Target dimensions:",
        target_width,
        "x",
        target_height
    )

    print(
        "Target resolution:",
        target.res
    )


# ============================================================
# READ RAW NDVI
# ============================================================

print("\nReading raw NDVI...")

with rasterio.open(RAW_NDVI) as src:

    source_crs = src.crs
    source_transform = src.transform

    source_data = src.read(1).astype("float32")

    print("Source CRS:", source_crs)
    print(
        "Source dimensions:",
        src.width,
        "x",
        src.height
    )

    print(
        "Source resolution:",
        src.res
    )


# ============================================================
# PREPARE OUTPUT ARRAY
# ============================================================

destination = np.full(
    (
        target_height,
        target_width
    ),
    -9999.0,
    dtype="float32"
)


# ============================================================
# REPROJECT / RESAMPLE
# ============================================================

print("\nReprojecting and resampling NDVI...")

reproject(
    source=source_data,
    destination=destination,

    src_transform=source_transform,
    src_crs=source_crs,

    dst_transform=target_transform,
    dst_crs=target_crs,

    resampling=Resampling.average,

    src_nodata=-9999.0,
    dst_nodata=-9999.0
)


# ============================================================
# SAVE
# ============================================================

print("\nSaving aligned NDVI raster...")

target_profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    width=target_width,
    height=target_height,
    crs=target_crs,
    transform=target_transform,
    nodata=-9999.0,
    compress="deflate"
)

with rasterio.open(
    OUTPUT,
    "w",
    **target_profile
) as dst:

    dst.write(
        destination,
        1
    )


# ============================================================
# QA
# ============================================================

print("\nRunning final QA...")

with rasterio.open(OUTPUT) as check:

    data = check.read(1)

    valid = data[
        np.isfinite(data)
        & (data != -9999.0)
    ]

    print("Output CRS:", check.crs)

    print(
        "Output dimensions:",
        check.width,
        "x",
        check.height
    )

    print(
        "Output resolution:",
        check.res
    )

    print(
        "Valid pixels:",
        len(valid)
    )

    if len(valid) > 0:

        print(
            "Minimum NDVI:",
            float(valid.min())
        )

        print(
            "Maximum NDVI:",
            float(valid.max())
        )

        print(
            "Mean NDVI:",
            float(valid.mean())
        )

        print(
            "Median NDVI:",
            float(np.median(valid))
        )


# ============================================================
# VALIDATION
# ============================================================

if check.crs != target_crs:
    raise RuntimeError(
        "CRS mismatch."
    )

if check.width != target_width:
    raise RuntimeError(
        "Width mismatch."
    )

if check.height != target_height:
    raise RuntimeError(
        "Height mismatch."
    )

if abs(check.res[0] - 250.0) > 0.01:
    raise RuntimeError(
        "Resolution is not 250 m."
    )

if abs(check.res[1] - 250.0) > 0.01:
    raise RuntimeError(
        "Resolution is not 250 m."
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS: NDVI IS ALIGNED TO THE FINAL 250 m NER GRID")
print("=" * 70)

print("\nFinal output:")
print(OUTPUT)

print("\nSource:")
print("COPERNICUS/S2_SR_HARMONIZED")

print("\nVariable:")
print("NDVI")

print("=" * 70)