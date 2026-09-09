from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform
from scipy.stats import pearsonr, spearmanr


PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

POINT_FILE = (
    PROJECT
    / "processed"
    / "ner_lineament_density.csv"
)

RASTER_FILE = (
    PROJECT
    / "processed"
    / "predictors"
    / "lineament_density_250m.tif"
)

OUTPUT_FILE = (
    PROJECT
    / "processed"
    / "step68e_lineament_raster_validation.csv"
)


print("=" * 70)
print("STEP 68E — LINEAMENT DENSITY RASTER VALIDATION")
print("=" * 70)


# ============================================================
# LOAD POINT DATA
# ============================================================

df = pd.read_csv(POINT_FILE)

print(f"\nPoint rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

required = [
    "lat",
    "lon",
    "lineament_density"
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# LOAD RASTER
# ============================================================

print("\nLoading raster...")

with rasterio.open(RASTER_FILE) as src:

    print(f"Raster CRS: {src.crs}")
    print(
        f"Raster size: "
        f"{src.width} x {src.height}"
    )
    print(
        f"Raster resolution: "
        f"{src.res}"
    )
    print(
        f"Raster NoData: "
        f"{src.nodata}"
    )

    raster = src.read(1)

    raster_crs = src.crs
    nodata = src.nodata
    transform_affine = src.transform

    height = src.height
    width = src.width


# ============================================================
# TRANSFORM POINTS
# ============================================================

print("\nTransforming training points...")

xs = df["lon"].to_numpy(dtype=float)
ys = df["lat"].to_numpy(dtype=float)

rx, ry = transform(
    "EPSG:4326",
    raster_crs,
    xs.tolist(),
    ys.tolist()
)

rx = np.asarray(rx)
ry = np.asarray(ry)


# ============================================================
# CHECK POINTS INSIDE RASTER
# ============================================================

inside = (
    (rx >= transform_affine.c)
    &
    (rx <= transform_affine.c
     + width * transform_affine.a)
)

# More robust row/column check
rows, cols = rowcol(
    transform_affine,
    rx,
    ry
)

rows = np.asarray(rows)
cols = np.asarray(cols)

inside = (
    (rows >= 0)
    & (rows < height)
    & (cols >= 0)
    & (cols < width)
)

print(
    f"\nPoints inside raster: "
    f"{inside.sum()} / {len(df)}"
)


# ============================================================
# EXTRACT RASTER VALUES
# ============================================================

raster_values = np.full(
    len(df),
    np.nan,
    dtype=float
)

valid_inside = np.where(inside)[0]

for i in valid_inside:

    value = raster[
        rows[i],
        cols[i]
    ]

    if nodata is not None:

        if np.isclose(
            value,
            nodata
        ):
            continue

    if np.isfinite(value):

        raster_values[i] = float(value)


df["raster_lineament_density"] = (
    raster_values
)


# ============================================================
# VALIDATION DATASET
# ============================================================

original = (
    df["lineament_density"]
    .to_numpy(dtype=float)
)

predicted = (
    df["raster_lineament_density"]
    .to_numpy(dtype=float)
)

original_valid = np.isfinite(original)
raster_valid = np.isfinite(predicted)

comparable = (
    original_valid
    & raster_valid
)

n_original = int(
    original_valid.sum()
)

n_raster = int(
    raster_valid.sum()
)

n_comparable = int(
    comparable.sum()
)

coverage = (
    n_comparable
    / n_original
    * 100
    if n_original > 0
    else 0
)


print("\n" + "=" * 70)
print("POINT-TO-RASTER COMPARISON")
print("=" * 70)

print(
    f"Original valid: "
    f"{n_original}"
)

print(
    f"Raster valid: "
    f"{n_raster}"
)

print(
    f"Comparable: "
    f"{n_comparable}"
)

print(
    f"Coverage: "
    f"{coverage:.2f}%"
)


if n_comparable < 2:

    raise ValueError(
        "Not enough comparable points."
    )


x = original[comparable]
y = predicted[comparable]


# ============================================================
# METRICS
# ============================================================

difference = y - x

mae = float(
    np.mean(np.abs(difference))
)

rmse = float(
    np.sqrt(
        np.mean(difference ** 2)
    )
)

mean_difference = float(
    np.mean(difference)
)

median_difference = float(
    np.median(difference)
)

pearson = float(
    pearsonr(x, y).statistic
)

spearman = float(
    spearmanr(x, y).statistic
)


print(
    f"\nOriginal mean: "
    f"{np.mean(x):.6f}"
)

print(
    f"Raster mean: "
    f"{np.mean(y):.6f}"
)

print(
    f"Original max: "
    f"{np.max(x):.6f}"
)

print(
    f"Raster max: "
    f"{np.max(y):.6f}"
)

print(
    f"\nMAE: "
    f"{mae:.6f}"
)

print(
    f"RMSE: "
    f"{rmse:.6f}"
)

print(
    f"Mean difference: "
    f"{mean_difference:.6f}"
)

print(
    f"Median difference: "
    f"{median_difference:.6f}"
)

print(
    f"Pearson: "
    f"{pearson:.4f}"
)

print(
    f"Spearman: "
    f"{spearman:.4f}"
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION RESULT")
print("=" * 70)


if (
    coverage >= 99
    and pearson >= 0.90
    and spearman >= 0.90
):

    result = "PASSED"

elif (
    coverage >= 90
    and pearson >= 0.80
    and spearman >= 0.80
):

    result = "ACCEPTABLE"

else:

    result = "FAILED"


print(
    f"LINEAMENT RASTER VALIDATION: "
    f"{result}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "original_valid": n_original,
    "raster_valid": n_raster,
    "comparable": n_comparable,
    "coverage_percent": coverage,
    "original_mean": float(np.mean(x)),
    "raster_mean": float(np.mean(y)),
    "original_max": float(np.max(x)),
    "raster_max": float(np.max(y)),
    "MAE": mae,
    "RMSE": rmse,
    "mean_difference": mean_difference,
    "median_difference": median_difference,
    "Pearson": pearson,
    "Spearman": spearman,
    "validation_result": result
}])


results.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved:\n{OUTPUT_FILE}"
)

print("\nDONE")
print("=" * 70)