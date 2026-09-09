from pathlib import Path
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

TILE_DIR = OUTPUT_DIR / "road_tiles"
TILE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTPUT_DIR / "distance_to_road_250m_raw.tif"


# ============================================================
# SETTINGS
# ============================================================

ROAD_ASSET = "projects/sat-io/open-datasets/GRIP4/South-East-Asia"

EXPORT_SCALE = 250


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 68M — REAL ROAD DISTANCE — 4 TILE EXPORT")
print("=" * 70)

print("Road source:", ROAD_ASSET)
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
# LOAD REAL GRIP4 ROADS
# ============================================================

print("\nLoading real GRIP4 roads...")

roads = ee.FeatureCollection(ROAD_ASSET)

road_count = roads.size().getInfo()

print("Road features found:", road_count)

if road_count == 0:
    raise RuntimeError("No GRIP4 road features found.")


# ============================================================
# CLIP ROADS
# ============================================================

print("\nClipping roads to NER...")

roads_ner = roads.filterBounds(ner)

road_count_ner = roads_ner.size().getInfo()

print(
    "Road features inside/near NER:",
    road_count_ner
)

if road_count_ner == 0:
    raise RuntimeError(
        "No GRIP4 roads intersect the NER."
    )


# ============================================================
# CREATE ROAD RASTER
# ============================================================

print("\nCreating road raster...")

road_image = (
    roads_ner
    .reduceToImage(
        properties=[],
        reducer=ee.Reducer.countEvery()
    )
    .gt(0)
)


# ============================================================
# DISTANCE TO ROAD
# ============================================================

print("\nCalculating distance to nearest road...")

distance = (
    road_image
    .distance(
        ee.Kernel.euclidean(
            radius=50000,
            units="meters"
        )
    )
    .rename("distance_to_road_m")
    .clip(ner)
)

print("Road-distance image created.")


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
# FOUR TILES
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
# EXPORT FOUR TILES
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

    tile_distance = distance.clip(
        tile_geometry
    )

    geemap.ee_export_image(
        tile_distance,
        filename=str(tile_file),
        scale=EXPORT_SCALE,
        region=tile_geometry,
        file_per_band=False
    )

    if not tile_file.exists():

        raise RuntimeError(
            f"{name} export failed:\n{tile_file}"
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
# MERGE TILES
# ============================================================

print()
print("=" * 70)
print("Merging four road-distance tiles...")
print("=" * 70)

tile_files = sorted(
    TILE_DIR.glob("tile_*.tif")
)

if len(tile_files) != 4:

    raise RuntimeError(
        f"Expected 4 tiles, found {len(tile_files)}."
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

    with rasterio.open(
        OUTPUT,
        "w",
        **profile
    ) as dst:

        dst.write(
            mosaic[0],
            1
        )

finally:

    for src in src_files:
        src.close()


# ============================================================
# QA
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
            "Minimum distance:",
            float(valid.min()),
            "m"
        )

        print(
            "Maximum distance:",
            float(valid.max()),
            "m"
        )

        print(
            "Mean distance:",
            float(valid.mean()),
            "m"
        )

        print(
            "Median distance:",
            float(
                __import__("numpy").median(valid)
            ),
            "m"
        )


# ============================================================
# VERIFY
# ============================================================

if not OUTPUT.exists():

    raise RuntimeError(
        "Final road-distance file was not created."
    )

if OUTPUT.stat().st_size == 0:

    raise RuntimeError(
        "Final road-distance file is empty."
    )


# ============================================================
# CLEAN TEMPORARY FILES
# ============================================================

print("\nCleaning temporary tiles...")

shutil.rmtree(
    TILE_DIR,
    ignore_errors=True
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("SUCCESS: REAL ROAD DISTANCE CREATED")
print("=" * 70)

print()
print("Source:")
print("GRIP4 South-East Asia")

print()
print("Variable:")
print("Distance to nearest road")

print()
print("Units:")
print("meters")

print()
print("Output:")
print(OUTPUT)

print()
print("=" * 70)