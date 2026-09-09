import os
import numpy as np
import pandas as pd
import geopandas as gpd
from pyhdf.SD import SD, SDC
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"

NDVI_DIR = os.path.join(PROJECT_ROOT, "raw_data", "ndvi")

POINTS_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "ner_training_points.geojson"
)

OUTPUT_FILE = os.path.join(
    PROJECT_ROOT,
    "processed",
    "ner_ndvi.csv"
)


# ============================================================
# FIND HDF FILES AUTOMATICALLY
# ============================================================

hdf_files = sorted([
    os.path.join(NDVI_DIR, f)
    for f in os.listdir(NDVI_DIR)
    if f.lower().endswith(".hdf")
])

print("MOD13Q1 HDF files found:", len(hdf_files))

for f in hdf_files:
    print(" ", os.path.basename(f))

if len(hdf_files) != 3:
    raise RuntimeError(
        f"Expected 3 MOD13Q1 HDF files, found {len(hdf_files)}."
    )


# ============================================================
# LOAD TRAINING POINTS
# ============================================================

print("\nLoading NER training points...")

gdf = gpd.read_file(POINTS_FILE)

print("Number of points:", len(gdf))
print("CRS:", gdf.crs)

if gdf.crs is None:
    raise RuntimeError("Training points have no CRS.")

# Make sure points are WGS84
gdf = gdf.to_crs("EPSG:4326")


# ============================================================
# MODIS SINUSOIDAL PROJECTION
# ============================================================

# MODIS MOD13Q1 uses the MODIS Sinusoidal projection.
# Sphere radius = 6371007.181 m

MODIS_SINU = CRS.from_proj4(
    "+proj=sinu +R=6371007.181 +nadgrids=@null +wktext"
)

transformer = Transformer.from_crs(
    "EPSG:4326",
    MODIS_SINU,
    always_xy=True
)


# ============================================================
# MODIS TILE SIZE
# ============================================================

PIXEL_SIZE = 231.65635826395

TILE_SIZE = 4800

# MODIS Sinusoidal global grid origin
GLOBAL_X_MIN = -20015109.354
GLOBAL_Y_MAX = 10007554.677


# ============================================================
# CONVERT MODIS TILE H/V TO TILE EXTENT
# ============================================================

def tile_extent(h, v):

    x_min = GLOBAL_X_MIN + h * TILE_SIZE * PIXEL_SIZE
    x_max = GLOBAL_X_MIN + (h + 1) * TILE_SIZE * PIXEL_SIZE

    y_max = GLOBAL_Y_MAX - v * TILE_SIZE * PIXEL_SIZE
    y_min = GLOBAL_Y_MAX - (v + 1) * TILE_SIZE * PIXEL_SIZE

    return x_min, y_min, x_max, y_max


# ============================================================
# READ NDVI FROM HDF4
# ============================================================

def read_ndvi_hdf(hdf_file):

    print("\nReading:", os.path.basename(hdf_file))

    hdf = SD(hdf_file, SDC.READ)

    datasets = hdf.datasets()

    ndvi_name = None

    for name in datasets.keys():

        if "250m 16 days NDVI" in name:
            ndvi_name = name
            break

    if ndvi_name is None:
        print("Available datasets:")
        for name in datasets.keys():
            print(" ", name)

        raise RuntimeError(
            "Could not find '250m 16 days NDVI' dataset."
        )

    print("NDVI dataset:", ndvi_name)

    sds = hdf.select(ndvi_name)

    data = sds.get().astype(np.float32)

    attrs = sds.attributes()

    print("Raster shape:", data.shape)

    # MOD13Q1 NDVI scale factor
    scale_factor = attrs.get("scale_factor", 0.0001)

    print("Scale factor:", scale_factor)

    # Fill value
    fill_value = attrs.get("_FillValue", -3000)

    print("Fill value:", fill_value)

    data[data == fill_value] = np.nan

    # Apply scale factor
    data = data / scale_factor

    # Valid NDVI range
    data[(data < -1) | (data > 1)] = np.nan

    hdf.end()

    return data


# ============================================================
# EXTRACT TILE NUMBER FROM FILENAME
# ============================================================

def get_tile_from_filename(filename):

    base = os.path.basename(filename)

    parts = base.split(".")

    for part in parts:

        if (
            len(part) == 6
            and part[0] == "h"
            and "v" in part
        ):
            h = int(part.split("v")[0][1:])
            v = int(part.split("v")[1])

            return h, v

    raise RuntimeError(
        f"Could not determine MODIS h/v tile from {filename}"
    )


# ============================================================
# PREPARE POINT COORDINATES
# ============================================================

lons = gdf.geometry.x.values
lats = gdf.geometry.y.values

xs, ys = transformer.transform(lons, lats)

print("\nConverted training points to MODIS Sinusoidal CRS.")


# ============================================================
# EXTRACT NDVI
# ============================================================

point_ndvi = np.full(
    len(gdf),
    np.nan,
    dtype=np.float32
)

point_tile = np.array(
    [""] * len(gdf),
    dtype=object
)


for hdf_file in hdf_files:

    h, v = get_tile_from_filename(hdf_file)

    tile_name = f"h{h:02d}v{v:02d}"

    print("\n====================================")
    print("Processing tile:", tile_name)
    print("====================================")

    ndvi = read_ndvi_hdf(hdf_file)

    x_min, y_min, x_max, y_max = tile_extent(h, v)

    print("Tile extent:")
    print("X:", x_min, "to", x_max)
    print("Y:", y_min, "to", y_max)

    # Find training points inside this MODIS tile
    inside = (
        (xs >= x_min)
        & (xs < x_max)
        & (ys >= y_min)
        & (ys < y_max)
    )

    indices = np.where(inside)[0]

    print("Training points inside tile:", len(indices))

    if len(indices) == 0:
        continue

    # Convert coordinates to pixel row/column
    cols = ((xs[indices] - x_min) / PIXEL_SIZE).astype(int)

    rows = ((y_max - ys[indices]) / PIXEL_SIZE).astype(int)

    # Protect against edge cases
    valid = (
        (rows >= 0)
        & (rows < ndvi.shape[0])
        & (cols >= 0)
        & (cols < ndvi.shape[1])
    )

    valid_indices = indices[valid]

    rows_valid = rows[valid]
    cols_valid = cols[valid]

    point_ndvi[valid_indices] = ndvi[
        rows_valid,
        cols_valid
    ]

    point_tile[valid_indices] = tile_name


# ============================================================
# SAVE RESULTS
# ============================================================

result = gdf.drop(columns="geometry").copy()

result["ndvi"] = point_ndvi
result["ndvi_tile"] = point_tile

result.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n====================================")
print("NDVI EXTRACTION COMPLETE")
print("====================================")

print("Output:", OUTPUT_FILE)
print("Total points:", len(result))

print(
    "Valid NDVI:",
    result["ndvi"].notna().sum()
)

print(
    "Missing NDVI:",
    result["ndvi"].isna().sum()
)

if result["ndvi"].notna().any():

    print(
        "NDVI minimum:",
        result["ndvi"].min()
    )

    print(
        "NDVI maximum:",
        result["ndvi"].max()
    )

    print(
        "NDVI mean:",
        result["ndvi"].mean()
    )