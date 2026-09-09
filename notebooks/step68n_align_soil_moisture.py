from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "processed" / "predictors" / "soil_moisture_3day_250m_raw.tif"
GRID = ROOT / "processed" / "step64a_final_ner_grid.tif"
OUTPUT = ROOT / "processed" / "predictors" / "soil_moisture_3day_250m.tif"

NODATA = -9999.0


print("=" * 75)
print("STEP 68N — ALIGN SOIL MOISTURE TO FINAL 250m GRID")
print("=" * 75)


if not SOURCE.exists():
    raise FileNotFoundError(f"Source not found:\n{SOURCE}")

if not GRID.exists():
    raise FileNotFoundError(f"Final grid not found:\n{GRID}")


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

    print("\nSource soil moisture:")
    print(f"  CRS: {src.crs}")
    print(f"  Dimensions: {src.width} x {src.height}")
    print(f"  Resolution: {src.res}")
    print(f"  Bounds: {src.bounds}")

    source = src.read(1).astype(np.float32)

    source_nodata = src.nodata

    if source_nodata is not None:
        source[source == source_nodata] = NODATA

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

if not np.any(valid):
    raise RuntimeError("No valid soil-moisture pixels after alignment.")


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


OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with rasterio.open(OUTPUT, "w", **profile) as dst:
    dst.write(destination, 1)


print("\n" + "=" * 75)
print("SOIL MOISTURE QA")
print("=" * 75)

print(f"Valid pixels: {valid.sum():,}")
print(f"Coverage: {valid.sum() / valid.size * 100:.2f}%")
print(f"Minimum: {destination[valid].min():.6f}")
print(f"Maximum: {destination[valid].max():.6f}")
print(f"Mean: {destination[valid].mean():.6f}")
print(f"Median: {np.median(destination[valid]):.6f}")

print("\nOutput:")
print(OUTPUT)

print("\n" + "=" * 75)
print("STEP 68N COMPLETE")
print("=" * 75)
