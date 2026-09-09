from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]

POINT_FILE = ROOT / "processed" / "ner_distance_to_road.csv"
RASTER = ROOT / "processed" / "predictors" / "distance_to_road_250m.tif"

OUTPUT = ROOT / "processed" / "step68a_road_raster_validation.csv"


print("=" * 70)
print("STEP 68A — ROAD DISTANCE RASTER VALIDATION")
print("=" * 70)


# ============================================================
# LOAD POINT DATA
# ============================================================

df = pd.read_csv(POINT_FILE)

print(f"\nPoint rows: {len(df)}")
print("Columns:", df.columns.tolist())


# Detect coordinate columns
lon_col = "longitude" if "longitude" in df.columns else "lon"
lat_col = "latitude" if "latitude" in df.columns else "lat"

if "distance_to_road_m" in df.columns:
    distance_col = "distance_to_road_m"
else:
    raise ValueError(
        "distance_to_road_m column not found."
    )


# ============================================================
# LOAD RASTER
# ============================================================

with rasterio.open(RASTER) as src:

    raster = src.read(1)
    transform = src.transform
    raster_crs = src.crs
    nodata = src.nodata

    xs, ys = rio_transform(
        "EPSG:4326",
        raster_crs,
        df[lon_col].values,
        df[lat_col].values,
    )

    rows, cols = rasterio.transform.rowcol(
        transform,
        xs,
        ys,
    )

rows = np.asarray(rows)
cols = np.asarray(cols)


# ============================================================
# EXTRACT RASTER VALUES
# ============================================================

inside = (
    (rows >= 0)
    & (rows < raster.shape[0])
    & (cols >= 0)
    & (cols < raster.shape[1])
)

raster_values = np.full(
    len(df),
    np.nan,
    dtype=float,
)

raster_values[inside] = raster[
    rows[inside],
    cols[inside]
]

if nodata is not None:

    raster_values[
        raster_values == nodata
    ] = np.nan


# ============================================================
# VALID COMPARISON
# ============================================================

original = df[
    distance_col
].astype(float).values

valid = (
    np.isfinite(original)
    & np.isfinite(raster_values)
)

a = original[valid]
b = raster_values[valid]

print("\n" + "=" * 70)
print("POINT-TO-RASTER COMPARISON")
print("=" * 70)

print(f"Original valid : {len(a)}")
print(f"Raster valid   : {len(b)}")
print(
    f"Coverage       : "
    f"{len(a) / len(df) * 100:.2f}%"
)


# ============================================================
# METRICS
# ============================================================

difference = b - a

mae = np.mean(
    np.abs(difference)
)

rmse = np.sqrt(
    np.mean(
        difference ** 2
    )
)

mean_difference = np.mean(
    difference
)

median_difference = np.median(
    difference
)

pearson = pearsonr(
    a,
    b
).statistic

spearman = spearmanr(
    a,
    b
).statistic


print(
    f"\nOriginal mean : {a.mean():.2f} m"
)

print(
    f"Raster mean   : {b.mean():.2f} m"
)

print(
    f"Original max  : {a.max():.2f} m"
)

print(
    f"Raster max    : {b.max():.2f} m"
)

print(
    f"\nMAE           : {mae:.2f} m"
)

print(
    f"RMSE          : {rmse:.2f} m"
)

print(
    f"Mean diff     : {mean_difference:.2f} m"
)

print(
    f"Median diff   : {median_difference:.2f} m"
)

print(
    f"Pearson       : {pearson:.4f}"
)

print(
    f"Spearman      : {spearman:.4f}"
)


# ============================================================
# SAVE
# ============================================================

comparison = df[
    [
        lat_col,
        lon_col,
        distance_col,
    ]
].copy()

comparison[
    "raster_distance_m"
] = raster_values

comparison[
    "difference_m"
] = (
    raster_values
    - comparison[
        distance_col
    ]
)

comparison.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# CONCLUSION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULT")
print("=" * 70)

if (
    len(a) / len(df) >= 0.99
    and pearson >= 0.90
    and spearman >= 0.90
):

    print(
        "ROAD RASTER VALIDATION: PASSED"
    )

elif (
    len(a) / len(df) >= 0.99
    and pearson >= 0.80
    and spearman >= 0.80
):

    print(
        "ROAD RASTER VALIDATION: "
        "ACCEPTABLE WITH CAUTION"
    )

else:

    print(
        "ROAD RASTER VALIDATION: FAILED"
    )


print(
    f"\nSaved: {OUTPUT}"
)

print("\nDONE")