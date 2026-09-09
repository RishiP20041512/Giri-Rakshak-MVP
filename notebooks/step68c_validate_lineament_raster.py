from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform
from scipy.stats import pearsonr, spearmanr


# ============================================================
# STEP 68C — LINEAMENT RASTER VALIDATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

POINT_FILE = (
    ROOT
    / "processed"
    / "ner_lineament_density.csv"
)

RASTER_FILE = (
    ROOT
    / "processed"
    / "predictors"
    / "lineament_density_250m.tif"
)

OUTPUT_FILE = (
    ROOT
    / "processed"
    / "step68c_lineament_raster_validation.csv"
)


print("=" * 70)
print("STEP 68C — LINEAMENT DENSITY RASTER VALIDATION")
print("=" * 70)


# ============================================================
# LOAD POINT DATA
# ============================================================

df = pd.read_csv(
    POINT_FILE
)

print(
    "\nPoint rows:",
    len(df)
)

print(
    "Columns:",
    df.columns.tolist()
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required = [
    "lat",
    "lon",
    "lineament_density"
]

for column in required:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# LOAD RASTER
# ============================================================

print(
    "\nLoading raster..."
)

with rasterio.open(
    RASTER_FILE
) as src:

    raster = src.read(1)

    raster_crs = src.crs

    raster_transform = src.transform

    raster_nodata = src.nodata

    raster_height = src.height

    raster_width = src.width

    raster_resolution = src.res


print(
    "Raster CRS:",
    raster_crs
)

print(
    "Raster size:",
    raster_width,
    "x",
    raster_height
)

print(
    "Raster resolution:",
    raster_resolution
)

print(
    "Raster NoData:",
    raster_nodata
)


# ============================================================
# CRS CHECK
# ============================================================

if raster_crs.to_epsg() != 6933:

    raise ValueError(
        "Raster CRS is not EPSG:6933."
    )


# ============================================================
# TRANSFORM POINT COORDINATES
# ============================================================

print(
    "\nTransforming training points..."
)

xs, ys = rio_transform(
    "EPSG:4326",
    raster_crs,
    df["lon"].values,
    df["lat"].values
)


# ============================================================
# GET RASTER ROW/COLUMN
# ============================================================

rows, cols = rasterio.transform.rowcol(
    raster_transform,
    xs,
    ys
)

rows = np.asarray(
    rows
)

cols = np.asarray(
    cols
)


# ============================================================
# CHECK POINTS INSIDE RASTER
# ============================================================

inside = (

    (rows >= 0)

    & (rows < raster_height)

    & (cols >= 0)

    & (cols < raster_width)
)


print(
    "\nPoints inside raster:",
    int(inside.sum()),
    "/",
    len(df)
)


# ============================================================
# EXTRACT RASTER VALUES
# ============================================================

raster_values = np.full(
    len(df),
    np.nan,
    dtype=float
)

raster_values[inside] = (
    raster[
        rows[inside],
        cols[inside]
    ]
)


# ============================================================
# HANDLE NODATA
# ============================================================

if raster_nodata is not None:

    raster_values[
        raster_values == raster_nodata
    ] = np.nan


# ============================================================
# ORIGINAL VALUES
# ============================================================

original_values = (
    df[
        "lineament_density"
    ]
    .astype(float)
    .values
)


# ============================================================
# VALID COMPARISON
# ============================================================

valid = (

    np.isfinite(
        original_values
    )

    & np.isfinite(
        raster_values
    )
)


original = (
    original_values[
        valid
    ]
)

raster_extracted = (
    raster_values[
        valid
    ]
)


print(
    "\n" + "=" * 70
)

print(
    "POINT-TO-RASTER COMPARISON"
)

print(
    "=" * 70
)


print(
    "Original valid:",
    len(original)
)

print(
    "Raster valid:",
    len(raster_extracted)
)

print(
    "Coverage:",
    f"{len(original) / len(df) * 100:.2f}%"
)


# ============================================================
# METRICS
# ============================================================

if len(original) < 2:

    raise RuntimeError(
        "Not enough valid points for validation."
    )


difference = (
    raster_extracted
    - original
)


mae = np.mean(
    np.abs(
        difference
    )
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
    original,
    raster_extracted
).statistic


spearman = spearmanr(
    original,
    raster_extracted
).statistic


# ============================================================
# PRINT STATISTICS
# ============================================================

print(
    "\nOriginal mean:",
    f"{original.mean():.6f}"
)

print(
    "Raster mean:",
    f"{raster_extracted.mean():.6f}"
)

print(
    "Original max:",
    f"{original.max():.6f}"
)

print(
    "Raster max:",
    f"{raster_extracted.max():.6f}"
)

print(
    "\nMAE:",
    f"{mae:.6f}"
)

print(
    "RMSE:",
    f"{rmse:.6f}"
)

print(
    "Mean difference:",
    f"{mean_difference:.6f}"
)

print(
    "Median difference:",
    f"{median_difference:.6f}"
)

print(
    "Pearson:",
    f"{pearson:.4f}"
)

print(
    "Spearman:",
    f"{spearman:.4f}"
)


# ============================================================
# SAVE POINT COMPARISON
# ============================================================

comparison = df[
    [
        "lat",
        "lon",
        "lineament_density"
    ]
].copy()


comparison[
    "raster_lineament_density"
] = raster_values


comparison[
    "difference"
] = (
    raster_values
    - comparison[
        "lineament_density"
    ]
)


comparison.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION DECISION
# ============================================================

coverage = (
    len(original)
    / len(df)
)


print(
    "\n" + "=" * 70
)

print(
    "VALIDATION RESULT"
)

print(
    "=" * 70
)


if (

    coverage >= 0.99

    and pearson >= 0.90

    and spearman >= 0.90

):

    print(
        "LINEAMENT RASTER VALIDATION: PASSED"
    )


elif (

    coverage >= 0.90

    and pearson >= 0.80

    and spearman >= 0.80

):

    print(
        "LINEAMENT RASTER VALIDATION: "
        "ACCEPTABLE WITH CAUTION"
    )


else:

    print(
        "LINEAMENT RASTER VALIDATION: FAILED"
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print(
    "\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    "\nDONE"
)

print(
    "=" * 70
)