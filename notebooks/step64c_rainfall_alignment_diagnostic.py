"""
STEP 64C — RAINFALL SPATIAL ALIGNMENT DIAGNOSTIC

Purpose
-------
Determine why the correctly extracted IMERG rainfall values do not
match the reprojected 250 m raster.

This script DOES NOT modify any raster or CSV.

It compares:
1. Exact IMERG source value
2. Existing rainfall CSV value
3. Final raster value at the training point
4. Raster values around the training point
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from netCDF4 import Dataset
from rasterio.transform import rowcol


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

RAIN_DIR = (
    PROJECT
    / "raw_data"
    / "rainfall"
)

CSV_FILE = (
    PROJECT
    / "processed"
    / "ner_rainfall.csv"
)

RASTER_FILE = (
    PROJECT
    / "processed"
    / "predictors"
    / "rainfall_3day_250m.tif"
)


# ============================================================
# LOAD POINTS
# ============================================================

print("=" * 70)
print("STEP 64C — RAINFALL SPATIAL ALIGNMENT DIAGNOSTIC")
print("=" * 70)

points = pd.read_csv(CSV_FILE)

print("\n[1] Training points")
print("-" * 70)

print(f"Points: {len(points):,}")


# ============================================================
# READ IMERG SOURCE
# ============================================================

files = sorted(
    RAIN_DIR.glob("*.nc4")
)

if len(files) != 3:

    raise RuntimeError(
        f"Expected 3 IMERG files, found {len(files)}"
    )


daily = []

for file in files:

    print(f"\nReading {file.name}")

    with Dataset(file, "r") as ds:

        lat = np.asarray(
            ds.variables["lat"][:],
            dtype=np.float64
        )

        lon = np.asarray(
            ds.variables["lon"][:],
            dtype=np.float64
        )

        var = ds.variables["precipitation"]

        arr = np.asarray(
            var[:],
            dtype=np.float64
        )

        fill = getattr(
            var,
            "_FillValue",
            -9999.9
        )

    arr = np.squeeze(arr)

    # Correct dimension order
    if arr.shape == (
        len(lon),
        len(lat)
    ):

        arr = arr.T

    elif arr.shape != (
        len(lat),
        len(lon)
    ):

        raise RuntimeError(
            f"Unexpected array shape: {arr.shape}"
        )

    # Remove fill values
    arr[
        (~np.isfinite(arr))
        | np.isclose(arr, fill, atol=1.0)
        | (arr < -1000)
    ] = np.nan

    daily.append(arr)


# ============================================================
# EXACT SOURCE EXTRACTION
# ============================================================

print("\n[2] Exact IMERG extraction")
print("-" * 70)

source_values = []

for lat_value, lon_value in zip(
    points["lat"],
    points["lon"]
):

    lat_idx = np.abs(
        lat - lat_value
    ).argmin()

    lon_idx = np.abs(
        lon - lon_value
    ).argmin()

    vals = [
        arr[lat_idx, lon_idx]
        for arr in daily
    ]

    if all(np.isfinite(vals)):

        source_values.append(
            np.sum(vals)
        )

    else:

        source_values.append(
            np.nan
        )

source_values = np.asarray(
    source_values
)


# ============================================================
# READ FINAL RASTER
# ============================================================

print("\n[3] Final rainfall raster")
print("-" * 70)

with rasterio.open(RASTER_FILE) as src:

    raster = src.read(1)

    transform = src.transform
    crs = src.crs

    raster_values = []

    raster_rows = []
    raster_cols = []

    for lat_value, lon_value in zip(
        points["lat"],
        points["lon"]
    ):

        # Transform WGS84 point to raster CRS
        from pyproj import Transformer

        transformer = Transformer.from_crs(
            "EPSG:4326",
            crs,
            always_xy=True
        )

        x, y = transformer.transform(
            lon_value,
            lat_value
        )

        row, col = rowcol(
            transform,
            x,
            y
        )

        raster_rows.append(row)
        raster_cols.append(col)

        if (
            row < 0
            or row >= src.height
            or col < 0
            or col >= src.width
        ):

            raster_values.append(
                np.nan
            )

        else:

            raster_values.append(
                raster[row, col]
            )

raster_values = np.asarray(
    raster_values
)


# ============================================================
# CSV VALUES
# ============================================================

csv_values = points[
    "rainfall_3day"
].to_numpy(
    dtype=float
)


# ============================================================
# COMPARISON FUNCTION
# ============================================================

def statistics(a, b):

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
    )

    if np.sum(valid) < 2:

        return None

    aa = a[valid]
    bb = b[valid]

    diff = bb - aa

    mae = np.mean(
        np.abs(diff)
    )

    rmse = np.sqrt(
        np.mean(diff ** 2)
    )

    pearson = np.corrcoef(
        aa,
        bb
    )[0, 1]

    rank_a = pd.Series(
        aa
    ).rank().to_numpy()

    rank_b = pd.Series(
        bb
    ).rank().to_numpy()

    spearman = np.corrcoef(
        rank_a,
        rank_b
    )[0, 1]

    return {
        "n": np.sum(valid),
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman
    }


# ============================================================
# MAIN COMPARISONS
# ============================================================

print("\n[4] Comparison statistics")
print("-" * 70)

print("\nExact source vs CSV")

s1 = statistics(
    source_values,
    csv_values
)

print(s1)


print("\nExact source vs final raster")

s2 = statistics(
    source_values,
    raster_values
)

print(s2)


print("\nCSV vs final raster")

s3 = statistics(
    csv_values,
    raster_values
)

print(s3)


# ============================================================
# TEST SHIFTED RASTER VALUES
# ============================================================

print("\n[5] Testing neighboring raster cells")
print("-" * 70)

# Read raster again and test offsets.
with rasterio.open(RASTER_FILE) as src:

    arr = src.read(1)

    transformer = Transformer.from_crs(
        "EPSG:4326",
        src.crs,
        always_xy=True
    )

    base_rows = []
    base_cols = []

    for lat_value, lon_value in zip(
        points["lat"],
        points["lon"]
    ):

        x, y = transformer.transform(
            lon_value,
            lat_value
        )

        r, c = rowcol(
            src.transform,
            x,
            y
        )

        base_rows.append(r)
        base_cols.append(c)


offset_results = []

for dr in range(-3, 4):

    for dc in range(-3, 4):

        values = []

        for r, c in zip(
            base_rows,
            base_cols
        ):

            rr = r + dr
            cc = c + dc

            if (
                0 <= rr < arr.shape[0]
                and 0 <= cc < arr.shape[1]
            ):

                values.append(
                    arr[rr, cc]
                )

            else:

                values.append(
                    np.nan
                )

        values = np.asarray(
            values
        )

        stats = statistics(
            source_values,
            values
        )

        if stats is not None:

            offset_results.append({
                "row_offset": dr,
                "col_offset": dc,
                **stats
            })


offset_df = pd.DataFrame(
    offset_results
)

offset_df = offset_df.sort_values(
    "pearson",
    ascending=False
)

print(
    "\nBest raster offsets by Pearson correlation:"
)

print(
    offset_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

output = (
    PROJECT
    / "processed"
    / "step64c_rainfall_alignment_diagnostic.csv"
)

offset_df.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)


# ============================================================
# FIRST 20 POINTS
# ============================================================

print("\n[6] First 20 points")
print("-" * 70)

preview = pd.DataFrame({
    "lat": points["lat"],
    "lon": points["lon"],
    "source": source_values,
    "csv": csv_values,
    "raster": raster_values,
    "difference_source_raster":
        raster_values - source_values
})

print(
    preview.head(20).to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("RAINFALL ALIGNMENT DIAGNOSTIC COMPLETED")
print("=" * 70)

print("\nNO FILES WERE MODIFIED.")