from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box


ROOT = Path(__file__).resolve().parents[1]

GRID = ROOT / "processed" / "step64a_final_ner_grid.tif"
BOUNDARY = ROOT / "raw_data" / "boundaries" / "geoBoundaries-IND-ADM1.geojson"
ROADS = ROOT / "raw_data" / "roads" / "north-eastern-zone.gpkg"

OUT = ROOT / "processed" / "predictors" / "distance_to_road_250m.tif"


print("=" * 70)
print("STEP 68A — FINAL ROAD DISTANCE RASTER")
print("=" * 70)


# ------------------------------------------------------------
# GRID
# ------------------------------------------------------------

with rasterio.open(GRID) as src:
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height
    bounds = src.bounds

print(f"Grid: {width} x {height}")
print(f"CRS : {crs}")


# ------------------------------------------------------------
# NER BOUNDARY
# ------------------------------------------------------------

print("\nLoading NER boundary...")

boundary = gpd.read_file(BOUNDARY)

states = [
    "Arunachal Pradesh",
    "Assam",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura",
]

boundary["shapeName"] = (
    boundary["shapeName"]
    .astype(str)
    .str.strip()
)

ner = boundary[
    boundary["shapeName"].isin(states)
].copy()

if len(ner) != 8:
    raise ValueError(
        f"Expected 8 NER states, found {len(ner)}"
    )

ner = ner.to_crs(crs)

ner_geom = ner.geometry.union_all()

print("NER boundary loaded.")


# ------------------------------------------------------------
# NER MASK
# ------------------------------------------------------------

print("\nCreating NER mask...")

ner_mask = rasterize(
    [(ner_geom, 1)],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8",
)

print(
    f"NER cells: {(ner_mask > 0).sum():,}"
)


# ------------------------------------------------------------
# ROADS
# ------------------------------------------------------------

print("\nLoading roads...")

roads = gpd.read_file(
    ROADS,
    layer="gis_osm_roads_free"
)

roads = roads[
    roads.geometry.notna()
].copy()

roads = roads[
    ~roads.geometry.is_empty
].copy()

roads = roads.to_crs(crs)

print(
    f"Road features: {len(roads):,}"
)


# ------------------------------------------------------------
# CLIP ROADS TO NER
# ------------------------------------------------------------

print("\nClipping roads to NER...")

roads = gpd.clip(
    roads,
    ner_geom
)

print(
    f"Clipped roads: {len(roads):,}"
)


# ------------------------------------------------------------
# RASTERIZE ROADS
# ------------------------------------------------------------

print("\nRasterizing roads...")

road_raster = rasterize(
    [
        (geom, 1)
        for geom in roads.geometry
    ],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="uint8",
    all_touched=True,
)

print(
    f"Road pixels: {(road_raster > 0).sum():,}"
)


# ------------------------------------------------------------
# DISTANCE
# ------------------------------------------------------------

print("\nCalculating distance to nearest road...")

distance = distance_transform_edt(
    road_raster == 0,
    sampling=250.0
).astype("float32")


# ------------------------------------------------------------
# APPLY NER MASK
# ------------------------------------------------------------

distance[
    ner_mask == 0
] = -9999


# ------------------------------------------------------------
# STATISTICS
# ------------------------------------------------------------

valid = distance[
    ner_mask > 0
]

valid = valid[
    valid >= 0
]

print("\nFINAL RASTER STATISTICS")

print(
    f"Valid cells : {len(valid):,}"
)

print(
    f"Min distance: {valid.min():.2f} m"
)

print(
    f"Max distance: {valid.max():.2f} m"
)

print(
    f"Mean distance: {valid.mean():.2f} m"
)

print(
    f"Median distance: {np.median(valid):.2f} m"
)


# ------------------------------------------------------------
# SAFETY CHECK
# ------------------------------------------------------------

if valid.max() > 200000:

    raise ValueError(
        "ERROR: Maximum distance exceeds 200 km. "
        "Raster is not suitable for final modelling."
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

profile.update(
    dtype="float32",
    count=1,
    nodata=-9999,
    compress="deflate",
)

with rasterio.open(
    OUT,
    "w",
    **profile
) as dst:

    dst.write(
        distance,
        1
    )


print("\nSaved:")
print(OUT)

print("\nDONE")