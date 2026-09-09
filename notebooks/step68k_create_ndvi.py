from pathlib import Path
import math
import shutil

import ee
import geemap
import rasterio
from rasterio.merge import merge


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BOUNDARY = ROOT / "raw_data" / "boundaries" / "ner_8states.geojson"

OUTPUT_DIR = ROOT / "processed" / "predictors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TILE_DIR = OUTPUT_DIR / "ndvi_tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "ndvi_250m_raw.tif"


# ============================================================
# SETTINGS
# ============================================================

START_DATE = "2026-07-01"
END_DATE = "2026-08-01"

DATASET = "COPERNICUS/S2_SR_HARMONIZED"

EXPORT_SCALE = 250


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 68K — REAL SENTINEL-2 NDVI — 4 TILE EXPORT")
print("=" * 70)

print("Dataset:", DATASET)
print("Period:", START_DATE, "to", "2026-07-31")
print("Resolution:", EXPORT_SCALE, "m")


# ============================================================
# EARTH ENGINE
# ============================================================

print("\nInitializing Google Earth Engine...")

try:
    ee.Initialize(project="land-slide-research")
except Exception:
    print("Starting authentication...")
    ee.Authenticate()
    ee.Initialize(project="land-slide-research")

print("Earth Engine initialized successfully.")


# ============================================================
# LOAD NER
# ============================================================

print("\nLoading NER boundary...")

if not BOUNDARY.exists():
    raise FileNotFoundError(
        f"NER boundary not found:\n{BOUNDARY}"
    )

ner = geemap.geojson_to_ee(str(BOUNDARY))

print("NER boundary loaded.")


# ============================================================
# SENTINEL-2 CLOUD MASK
# ============================================================

def mask_s2(image):

    scl = image.select("SCL")

    mask = (
        scl.eq(4)
        .Or(scl.eq(5))
        .Or(scl.eq(6))
        .Or(scl.eq(7))
    )

    return image.updateMask(mask)


# ============================================================
# LOAD SENTINEL-2
# ============================================================

print("\nLoading Sentinel-2 images...")

s2 = (
    ee.ImageCollection(DATASET)
    .filterDate(START_DATE, END_DATE)
    .filterBounds(ner)
    .filter(
        ee.Filter.lt(
            "CLOUDY_PIXEL_PERCENTAGE",
            50
        )
    )
    .map(mask_s2)
)

count = s2.size().getInfo()

print("Sentinel-2 images found:", count)

if count == 0:
    raise RuntimeError(
        "No Sentinel-2 images found."
    )


# ============================================================
# CALCULATE REAL NDVI
# ============================================================

print("\nCalculating real Sentinel-2 NDVI...")

def calculate_ndvi(image):

    return (
        image
        .normalizedDifference(["B8", "B4"])
        .rename("ndvi")
        .copyProperties(
            image,
            ["system:time_start"]
        )
    )


ndvi = (
    s2
    .map(calculate_ndvi)
    .median()
    .rename("ndvi")
    .clip(ner)
)

print("NDVI image created.")


# ============================================================
# GET NER BOUNDING BOX
# ============================================================

print("\nCreating four export tiles...")

bounds = ner.geometry().bounds()

coords = ee.List(
    bounds.coordinates().get(0)
).getInfo()

xs = [p[0] for p in coords]
ys = [p[1] for p in coords]

min_lon = min(xs)
max_lon = max(xs)
min_lat = min(ys)
max_lat = max(ys)

mid_lon = (min_lon + max_lon) / 2
mid_lat = (min_lat + max_lat) / 2


# ============================================================
# FOUR QUADRANTS
# ============================================================

tiles = [
    (
        "tile_1",
        min_lon,
        min_lat,
        mid_lon,
        mid_lat
    ),
    (
        "tile_2",
        mid_lon,
        min_lat,
        max_lon,
        mid_lat
    ),
    (
        "tile_3",
        min_lon,
        mid_lat,
        mid_lon,
        max_lat
    ),
    (
        "tile_4",
        mid_lon,
        mid_lat,
        max_lon,
        max_lat
    )
]


# ============================================================
# REMOVE OLD TILES
# ============================================================

for old_file in TILE_DIR.glob("*.tif"):
    old_file.unlink()


# ============================================================
# EXPORT EACH TILE
# ============================================================

for name, xmin, ymin, xmax, ymax in tiles:

    print()
    print("-" * 70)
    print("Exporting:", name)

    tile_file = TILE_DIR / f"{name}.tif"

    tile_geometry = ee.Geometry.Rectangle(
        [
            xmin,
            ymin,
            xmax,
            ymax
        ]
    )

    tile_ndvi = ndvi.clip(tile_geometry)

    geemap.ee_export_image(
        tile_ndvi,
        filename=str(tile_file),
        scale=EXPORT_SCALE,
        region=tile_geometry,
        file_per_band=False
    )

    if not tile_file.exists():

        raise RuntimeError(
            f"{name} failed. File was not created:\n"
            f"{tile_file}"
        )

    size_mb = (
        tile_file.stat().st_size
        / (1024 * 1024)
    )

    print(
        name,
        "SUCCESS —",
        round(size_mb, 2),
        "MB"
    )


# ============================================================
# MERGE FOUR TILES
# ============================================================

print()
print("=" * 70)
print("Merging four NDVI tiles...")
print("=" * 70)

tile_files = sorted(
    TILE_DIR.glob("tile_*.tif")
)

if len(tile_files) != 4:

    raise RuntimeError(
        f"Expected 4 NDVI tiles, found {len(tile_files)}."
    )

src_files = [
    rasterio.open(str(f))
    for f in tile_files
]

try:

    mosaic, transform = merge(
        src_files
    )

    profile = src_files[0].profile.copy()

    profile.update(
        driver="GTiff",
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
        count=1,
        dtype="float32",
        compress="deflate",
        nodata=-9999.0
    )

    mosaic = mosaic.astype("float32")

    mosaic[mosaic != mosaic] = -9999.0

    with rasterio.open(
        OUTPUT,
        "w",
        **profile
    ) as dst:

        dst.write(mosaic[0], 1)

finally:

    for src in src_files:
        src.close()


# ============================================================
# FINAL QA
# ============================================================

print()
print("=" * 70)
print("Running final QA...")
print("=" * 70)

with rasterio.open(OUTPUT) as src:

    data = src.read(1)

    valid = data[
        data != -9999.0
    ]

    print("CRS:", src.crs)
    print(
        "Dimensions:",
        src.width,
        "x",
        src.height
    )

    print(
        "Resolution:",
        src.res
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


# ============================================================
# CHECK OUTPUT
# ============================================================

if not OUTPUT.exists():

    raise RuntimeError(
        "Final NDVI file was not created."
    )

if OUTPUT.stat().st_size == 0:

    raise RuntimeError(
        "Final NDVI file is empty."
    )


# ============================================================
# CLEANUP
# ============================================================

print("\nCleaning temporary NDVI tiles...")

shutil.rmtree(
    TILE_DIR,
    ignore_errors=True
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("SUCCESS: REAL SENTINEL-2 NDVI CREATED")
print("=" * 70)

print()
print("Output:")
print(OUTPUT)

print()
print("Source:")
print("COPERNICUS/S2_SR_HARMONIZED")

print()
print("Period:")
print(START_DATE, "to", "2026-07-31")

print()
print("NDVI:")
print("(B8 - B4) / (B8 + B4)")

print()
print("=" * 70)