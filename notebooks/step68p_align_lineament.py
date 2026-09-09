from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "processed" / "predictors" / "lineament_density_250m.tif"
GRID = ROOT / "processed" / "step64a_final_ner_grid.tif"
OUTPUT = ROOT / "processed" / "predictors" / "lineament_density_250m_aligned.tif"

NODATA = -9999.0

print("=" * 75)
print("STEP 68P — ALIGN LINEAMENT DENSITY TO FINAL 250m GRID")
print("=" * 75)

with rasterio.open(GRID) as grid:
    target_crs = grid.crs
    target_transform = grid.transform
    target_width = grid.width
    target_height = grid.height

    print("\nTarget grid:")
    print(f"  CRS: {target_crs}")
    print(f"  Dimensions: {target_width} x {target_height}")
    print(f"  Resolution: {grid.res}")


with rasterio.open(SOURCE) as src:

    print("\nSource lineament:")
    print(f"  CRS: {src.crs}")
    print(f"  Dimensions: {src.width} x {src.height}")
    print(f"  Resolution: {src.res}")

    source = src.read(1).astype(np.float32)

    if src.nodata is not None:
        source[source == src.nodata] = NODATA

    destination = np.full(
        (target_height, target_width),
        NODATA,
        dtype=np.float32
    )

    reproject(
        source=source,
        destination=destination,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=NODATA,
        dst_transform=target_transform,
        dst_crs=target_crs,
        dst_nodata=NODATA,
        resampling=Resampling.average,
    )


valid = destination != NODATA

print("\n" + "=" * 75)
print("LINEAMENT QA")
print("=" * 75)

print(f"Valid pixels: {valid.sum():,}")
print(f"Coverage: {valid.sum() / valid.size * 100:.2f}%")

if np.any(valid):
    print(f"Minimum: {destination[valid].min():.6f}")
    print(f"Maximum: {destination[valid].max():.6f}")
    print(f"Mean: {destination[valid].mean():.6f}")
    print(f"Median: {np.median(destination[valid]):.6f}")

profile = {
    "driver": "GTiff",
    "height": target_height,
    "width": target_width,
    "count": 1,
    "dtype": "float32",
    "crs": target_crs,
    "transform": target_transform,
    "nodata": NODATA,
    "compress": "deflate",
    "predictor": 2,
}

with rasterio.open(OUTPUT, "w", **profile) as dst:
    dst.write(destination, 1)

print("\nOutput:")
print(OUTPUT)

print("\n" + "=" * 75)
print("STEP 68P COMPLETE")
print("=" * 75)