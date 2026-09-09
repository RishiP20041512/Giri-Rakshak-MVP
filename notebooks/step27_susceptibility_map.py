import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.warp import reproject, Resampling
from scipy.spatial import cKDTree


# ============================================================
# FILES
# ============================================================

dem_file = "raw_data/dem/dem.tif"
slope_file = "raw_data/dem/slope.tif"
aspect_file = "raw_data/dem/aspect.tif"
ndvi_file = "processed/ndvi.tif"

training_file = (
    "processed/training_features_complete.geojson"
)

model_file = (
    "processed/random_forest_model.joblib"
)


# ============================================================
# OUTPUTS
# ============================================================

probability_file = (
    "processed/susceptibility_probability.tif"
)

class_file = (
    "processed/susceptibility_class.tif"
)

csv_file = (
    "processed/susceptibility_summary.csv"
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "elevation",
    "slope",
    "aspect",
    "ndvi",
    "rainfall_3day",
    "soil_moisture",
    "dist_to_road",
    "dist_to_river"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("========================================")
print("STEP 27 — SUSCEPTIBILITY MAP")
print("========================================")

print("\nLoading Random Forest...")

model = joblib.load(model_file)

print("Model loaded.")


# ============================================================
# LOAD TRAINING FEATURES
# ============================================================

print("\nLoading training features...")

points = gpd.read_file(training_file)

print(
    "Training points:",
    len(points)
)

print(
    "CRS:",
    points.crs
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [
    c for c in features
    if c not in points.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns: "
        + str(missing_columns)
    )


# ============================================================
# LOAD DEM AS MASTER GRID
# ============================================================

print("\nLoading DEM...")

with rasterio.open(dem_file) as src:

    dem = src.read(1)

    profile = src.profile.copy()

    transform = src.transform

    crs = src.crs

    width = src.width
    height = src.height

    nodata = src.nodata

print(
    "DEM size:",
    width,
    "x",
    height
)

print(
    "DEM CRS:",
    crs
)


# ============================================================
# LOAD SLOPE
# ============================================================

print("\nLoading slope...")

with rasterio.open(slope_file) as src:

    slope = src.read(1)


# ============================================================
# LOAD ASPECT
# ============================================================

print("Loading aspect...")

with rasterio.open(aspect_file) as src:

    aspect = src.read(1)


# ============================================================
# CHECK TERRAIN SHAPES
# ============================================================

if slope.shape != dem.shape:

    raise ValueError(
        "Slope dimensions do not match DEM."
    )

if aspect.shape != dem.shape:

    raise ValueError(
        "Aspect dimensions do not match DEM."
    )


# ============================================================
# RESAMPLE NDVI TO DEM GRID
# ============================================================

print("\nResampling NDVI to DEM grid...")

ndvi = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)

with rasterio.open(ndvi_file) as src:

    reproject(
        source=rasterio.band(src, 1),
        destination=ndvi,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=Resampling.bilinear,
        src_nodata=src.nodata,
        dst_nodata=np.nan
    )

print("NDVI aligned.")


# ============================================================
# PREPARE POINT COORDINATES
# ============================================================

print("\nPreparing point-based features...")

points = points.to_crs(crs)

point_x = points.geometry.x.to_numpy()
point_y = points.geometry.y.to_numpy()


# ============================================================
# CREATE GRID COORDINATES
# ============================================================

rows, cols = np.indices(
    (height, width)
)

xs, ys = rasterio.transform.xy(
    transform,
    rows,
    cols
)

xs = np.asarray(xs)
ys = np.asarray(ys)


# ============================================================
# BUILD KD TREE
# ============================================================

tree = cKDTree(
    np.column_stack(
        (point_x, point_y)
    )
)


# ============================================================
# FUNCTION FOR POINT-BASED SURFACES
# ============================================================

def interpolate_feature(
    feature_name
):

    print(
        "Creating surface:",
        feature_name
    )

    values = (
        pd.to_numeric(
            points[feature_name],
            errors="coerce"
        )
        .to_numpy(
            dtype=float
        )
    )

    valid = np.isfinite(values)

    if valid.sum() == 0:

        raise ValueError(
            f"No valid values for {feature_name}"
        )

    local_tree = cKDTree(
        np.column_stack(
            (
                point_x[valid],
                point_y[valid]
            )
        )
    )

    local_values = values[valid]

    result = np.empty(
        (height, width),
        dtype=np.float32
    )

    # Process in chunks so memory stays reasonable
    chunk_size = 100000

    flat_x = xs.ravel()
    flat_y = ys.ravel()

    flat_result = result.ravel()

    for start in range(
        0,
        len(flat_x),
        chunk_size
    ):

        end = min(
            start + chunk_size,
            len(flat_x)
        )

        query_points = np.column_stack(
            (
                flat_x[start:end],
                flat_y[start:end]
            )
        )

        _, nearest = local_tree.query(
            query_points,
            k=1
        )

        flat_result[start:end] = (
            local_values[nearest]
        )

    return result


# ============================================================
# CREATE WEATHER / DISTANCE SURFACES
# ============================================================

rainfall = interpolate_feature(
    "rainfall_3day"
)

soil_moisture = interpolate_feature(
    "soil_moisture"
)

dist_to_road = interpolate_feature(
    "dist_to_road"
)

dist_to_river = interpolate_feature(
    "dist_to_river"
)


# ============================================================
# CREATE VALID MASK
# ============================================================

valid_mask = np.isfinite(dem)

valid_mask &= np.isfinite(slope)
valid_mask &= np.isfinite(aspect)
valid_mask &= np.isfinite(ndvi)
valid_mask &= np.isfinite(rainfall)
valid_mask &= np.isfinite(soil_moisture)
valid_mask &= np.isfinite(dist_to_road)
valid_mask &= np.isfinite(dist_to_river)


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

print("\nPreparing model grid...")

total_pixels = height * width

print(
    "Total pixels:",
    f"{total_pixels:,}"
)

print(
    "Valid pixels:",
    f"{valid_mask.sum():,}"
)


# ============================================================
# PREDICT IN CHUNKS
# ============================================================

probability = np.full(
    (height, width),
    np.nan,
    dtype=np.float32
)

valid_indices = np.where(
    valid_mask.ravel()
)[0]


print("\nRunning Random Forest over map...")

chunk_size = 100000

for start in range(
    0,
    len(valid_indices),
    chunk_size
):

    end = min(
        start + chunk_size,
        len(valid_indices)
    )

    idx = valid_indices[start:end]

    X = np.column_stack(
        [
            dem.ravel()[idx],
            slope.ravel()[idx],
            aspect.ravel()[idx],
            ndvi.ravel()[idx],
            rainfall.ravel()[idx],
            soil_moisture.ravel()[idx],
            dist_to_road.ravel()[idx],
            dist_to_river.ravel()[idx]
        ]
    )

    probabilities = model.predict_proba(
        X
    )[:, 1]

    probability.ravel()[idx] = (
        probabilities
    )

    print(
        f"Processed {end:,} / "
        f"{len(valid_indices):,} pixels"
    )


# ============================================================
# SAVE PROBABILITY RASTER
# ============================================================

print("\nSaving probability map...")

prob_profile = profile.copy()

prob_profile.update(
    dtype="float32",
    count=1,
    compress="lzw",
    nodata=-9999
)

probability_to_save = np.where(
    np.isfinite(probability),
    probability,
    -9999
).astype(np.float32)


with rasterio.open(
    probability_file,
    "w",
    **prob_profile
) as dst:

    dst.write(
        probability_to_save,
        1
    )


# ============================================================
# CLASSIFY SUSCEPTIBILITY
# ============================================================

print("\nClassifying susceptibility...")

susceptibility_class = np.full(
    (height, width),
    255,
    dtype=np.uint8
)


# LOW = 1
# MODERATE = 2
# HIGH = 3

low = (
    valid_mask
    & (probability < 0.33)
)

moderate = (
    valid_mask
    & (probability >= 0.33)
    & (probability < 0.66)
)

high = (
    valid_mask
    & (probability >= 0.66)
)


susceptibility_class[low] = 1
susceptibility_class[moderate] = 2
susceptibility_class[high] = 3


# ============================================================
# SAVE CLASS MAP
# ============================================================

class_profile = profile.copy()

class_profile.update(
    dtype="uint8",
    count=1,
    compress="lzw",
    nodata=255
)


with rasterio.open(
    class_file,
    "w",
    **class_profile
) as dst:

    dst.write(
        susceptibility_class,
        1
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("SUSCEPTIBILITY SUMMARY")
print("========================================")

valid_probability = probability[
    np.isfinite(probability)
]

print(
    "\nProbability statistics:"
)

print(
    pd.Series(
        valid_probability
    ).describe()
)


print("\nPixel classes:")

print(
    "LOW:",
    int(np.sum(susceptibility_class == 1))
)

print(
    "MODERATE:",
    int(np.sum(susceptibility_class == 2))
)

print(
    "HIGH:",
    int(np.sum(susceptibility_class == 3))
)


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary = pd.DataFrame({
    "class": [
        "Low",
        "Moderate",
        "High"
    ],
    "pixel_count": [
        int(np.sum(susceptibility_class == 1)),
        int(np.sum(susceptibility_class == 2)),
        int(np.sum(susceptibility_class == 3))
    ]
})

summary.to_csv(
    csv_file,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("STEP 27 COMPLETE")
print("========================================")

print(
    "Probability map:",
    probability_file
)

print(
    "Class map:",
    class_file
)

print(
    "Summary:",
    csv_file
)