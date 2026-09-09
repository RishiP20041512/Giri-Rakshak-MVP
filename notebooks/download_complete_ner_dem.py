import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
import geopandas as gpd

ROOT = Path(".")
TILE_DIR = ROOT / "raw_data" / "dem_ner" / "tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)

# Exact corrected NER extent: 88-98E, 21-30N
names = [
    f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
    for lat in range(21, 30)
    for lon in range(88, 98)
]

def download(name):
    path = TILE_DIR / f"{name}.tif"

    if path.exists() and path.stat().st_size > 100000:
        try:
            with rasterio.open(path):
                return name, "already valid"
        except Exception:
            path.unlink()

    url = (
        f"https://copernicus-dem-30m.s3.amazonaws.com/"
        f"{name}/{name}.tif"
    )

    r = requests.get(url, timeout=180)
    r.raise_for_status()

    if not r.content[:4] in (b"II*\x00", b"MM\x00*"):
        raise RuntimeError(f"Not a TIFF: {name}")

    path.write_bytes(r.content)

    with rasterio.open(path) as src:
        if src.width == 0 or src.height == 0:
            raise RuntimeError(f"Invalid raster: {name}")

    return name, "downloaded"

print("=" * 70)
print("DOWNLOADING REAL COPERNICUS GLO-30 DEM")
print("=" * 70)
print(f"Tiles required: {len(names)}")

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(download, n): n for n in names}

    done = 0
    for future in as_completed(futures):
        name, status = future.result()
        done += 1
        print(f"[{done}/{len(names)}] {status}: {name}")

print("\nAll DEM tiles are valid.")

# ------------------------------------------------------------
# Build a 250 m mosaic directly.
# This avoids creating a huge 30 m NER mosaic.
# ------------------------------------------------------------

print("\nCreating 250 m DEM mosaic...")

tile_paths = [TILE_DIR / f"{n}.tif" for n in names]

datasets = []
vrts = []

try:
    for path in tile_paths:
        src = rasterio.open(path)
        datasets.append(src)

        vrt = WarpedVRT(
            src,
            crs="EPSG:6933",
            resolution=250,
            resampling=rasterio.enums.Resampling.bilinear,
            nodata=-9999
        )
        vrts.append(vrt)

    mosaic, transform = merge(
        vrts,
        nodata=-9999
    )

finally:
    for vrt in vrts:
        vrt.close()
    for src in datasets:
        src.close()

# ------------------------------------------------------------
# Clip to the EXACT 8-state NER boundary
# ------------------------------------------------------------

boundary = (
    ROOT
    / "raw_data"
    / "boundaries"
    / "ner_8states.geojson"
)

ner = gpd.read_file(boundary).to_crs("EPSG:6933")

from rasterio.features import geometry_mask

inside = geometry_mask(
    ner.geometry,
    out_shape=(mosaic.shape[1], mosaic.shape[2]),
    transform=transform,
    invert=True
)

mosaic = np.array(mosaic, copy=True)
mosaic[0, ~inside] = -9999

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out = ROOT / "raw_data" / "dem_ner" / "dem_ner.tif"

profile = {
    "driver": "GTiff",
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "count": 1,
    "dtype": "float32",
    "crs": "EPSG:6933",
    "transform": transform,
    "nodata": -9999,
    "compress": "deflate",
    "predictor": 3,
    "BIGTIFF": "IF_SAFER"
}

with rasterio.open(out, "w", **profile) as dst:
    dst.write(mosaic)

valid = mosaic[mosaic != -9999]

print("\n" + "=" * 70)
print("REAL NER DEM CREATED SUCCESSFULLY")
print("=" * 70)
print("File:", out)
print("CRS:", profile["crs"])
print("Resolution:", profile["transform"].a, "m")
print("Dimensions:", mosaic.shape[2], "x", mosaic.shape[1])
print("Valid pixels:", len(valid))
print("Elevation minimum:", float(valid.min()))
print("Elevation maximum:", float(valid.max()))
print("=" * 70)
