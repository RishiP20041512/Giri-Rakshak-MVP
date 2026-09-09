"""
STEP 64C — VALIDATE IMERG RAINFALL SOURCE EXTRACTION

Purpose
-------
Determine whether the existing point-level rainfall CSV agrees
with the original IMERG NetCDF files after correcting the
longitude/latitude array orientation and handling fill values.

This script DOES NOT modify any existing files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from netCDF4 import Dataset


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main")

RAIN_DIR = PROJECT / "raw_data" / "rainfall"
POINT_FILE = PROJECT / "processed" / "ner_rainfall.csv"


# ============================================================
# LOAD POINT DATA
# ============================================================

print("=" * 70)
print("STEP 64C — IMERG SOURCE VALIDATION")
print("=" * 70)

print("\n[1] Loading existing rainfall point data")
print("-" * 70)

points = pd.read_csv(POINT_FILE)

print(f"Rows: {len(points):,}")
print(f"Columns: {list(points.columns)}")

required = {"lat", "lon", "rainfall_3day"}

missing = required - set(points.columns)

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

print(f"Latitude range : {points['lat'].min():.6f} "
      f"to {points['lat'].max():.6f}")

print(f"Longitude range: {points['lon'].min():.6f} "
      f"to {points['lon'].max():.6f}")

print(f"CSV rainfall min : {points['rainfall_3day'].min():.6f}")
print(f"CSV rainfall max : {points['rainfall_3day'].max():.6f}")
print(f"CSV rainfall mean: {points['rainfall_3day'].mean():.6f}")


# ============================================================
# READ THREE IMERG FILES
# ============================================================

rain_files = sorted(RAIN_DIR.glob("*.nc4"))

if len(rain_files) != 3:
    raise RuntimeError(
        f"Expected 3 IMERG files, found {len(rain_files)}"
    )

daily_values = []

for file in rain_files:

    print("\n" + "=" * 70)
    print(f"Reading: {file.name}")
    print("=" * 70)

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

        print("\nVariable metadata:")

        print(f"  dtype      : {var.dtype}")
        print(f"  dimensions : {var.dimensions}")

        for attr in [
            "units",
            "_FillValue",
            "missing_value",
            "long_name",
            "standard_name"
        ]:
            if hasattr(var, attr):
                print(
                    f"  {attr:13s}: "
                    f"{getattr(var, attr)}"
                )

        precip = np.asarray(
            var[:],
            dtype=np.float64
        )

    precip = np.squeeze(precip)

    print(f"\nRaw precipitation shape: {precip.shape}")
    print(f"Expected lat x lon    : {(len(lat), len(lon))}")

    # --------------------------------------------------------
    # Correct orientation
    # --------------------------------------------------------

    if precip.shape == (len(lon), len(lat)):

        print(
            "Orientation: lon x lat detected -> TRANSPOSE"
        )

        precip = precip.T

    elif precip.shape == (len(lat), len(lon)):

        print(
            "Orientation: already lat x lon"
        )

    else:

        raise RuntimeError(
            f"Unexpected shape: {precip.shape}"
        )

    # --------------------------------------------------------
    # Identify fill values robustly
    # --------------------------------------------------------

    invalid = ~np.isfinite(precip)

    # Known IMERG fill value
    invalid |= np.isclose(
        precip,
        -9999.9,
        atol=1.0
    )

    invalid |= precip < -1000

    precip[invalid] = np.nan

    print(
        f"Valid pixels after fill masking: "
        f"{np.sum(np.isfinite(precip)):,}"
    )

    print(
        f"Daily valid min: "
        f"{np.nanmin(precip):.6f}"
    )

    print(
        f"Daily valid max: "
        f"{np.nanmax(precip):.6f}"
    )

    daily_values.append(
        precip
    )


# ============================================================
# EXACT POINT EXTRACTION
# ============================================================

print("\n\n[2] EXACT IMERG POINT EXTRACTION")
print("-" * 70)

# Coordinates are regular 0.1-degree grids.
# Find nearest coordinate independently for each point.

corrected_daily = []

for day_index, file in enumerate(rain_files):

    with Dataset(file, "r") as ds:

        lat = np.asarray(
            ds.variables["lat"][:],
            dtype=np.float64
        )

        lon = np.asarray(
            ds.variables["lon"][:],
            dtype=np.float64
        )

    precip = daily_values[day_index]

    extracted = []

    for lat_value, lon_value in zip(
        points["lat"],
        points["lon"]
    ):

        lat_idx = np.abs(lat - lat_value).argmin()
        lon_idx = np.abs(lon - lon_value).argmin()

        value = precip[lat_idx, lon_idx]

        extracted.append(value)

    corrected_daily.append(
        np.asarray(extracted)
    )


corrected_daily = np.vstack(corrected_daily)

corrected_3day = np.nansum(
    corrected_daily,
    axis=0
)

# If any day is missing, make the 3-day value missing
missing_day = np.any(
    ~np.isfinite(corrected_daily),
    axis=0
)

corrected_3day[missing_day] = np.nan


# ============================================================
# COMPARISON
# ============================================================

csv_values = points[
    "rainfall_3day"
].to_numpy(dtype=float)

valid = (
    np.isfinite(csv_values)
    & np.isfinite(corrected_3day)
)

print(f"\nComparable points: {np.sum(valid):,}")

if np.sum(valid) == 0:
    raise RuntimeError(
        "No comparable rainfall observations."
    )

csv_valid = csv_values[valid]
corrected_valid = corrected_3day[valid]

difference = corrected_valid - csv_valid


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

mae = np.mean(
    np.abs(difference)
)

rmse = np.sqrt(
    np.mean(difference ** 2)
)

mean_diff = np.mean(difference)

median_diff = np.median(difference)

pearson = np.corrcoef(
    csv_valid,
    corrected_valid
)[0, 1]

csv_rank = pd.Series(csv_valid).rank().to_numpy()
corrected_rank = pd.Series(
    corrected_valid
).rank().to_numpy()

spearman = np.corrcoef(
    csv_rank,
    corrected_rank
)[0, 1]


print("\n" + "=" * 70)
print("RAINfall SOURCE COMPARISON")
print("=" * 70)

print(f"\nExisting CSV:")
print(f"  Mean: {np.mean(csv_valid):.6f}")
print(f"  Min : {np.min(csv_valid):.6f}")
print(f"  Max : {np.max(csv_valid):.6f}")

print("\nCorrected direct IMERG extraction:")
print(f"  Mean: {np.mean(corrected_valid):.6f}")
print(f"  Min : {np.min(corrected_valid):.6f}")
print(f"  Max : {np.max(corrected_valid):.6f}")

print("\nDifference:")
print(f"  MAE             : {mae:.6f}")
print(f"  RMSE            : {rmse:.6f}")
print(f"  Mean difference : {mean_diff:.6f}")
print(f"  Median difference: {median_diff:.6f}")
print(f"  Pearson r       : {pearson:.6f}")
print(f"  Spearman r      : {spearman:.6f}")


# ============================================================
# DAILY SOURCE STATISTICS
# ============================================================

print("\n\n[3] DAILY VALUES AT TRAINING POINTS")
print("-" * 70)

for i in range(3):

    values = corrected_daily[i]

    valid_day = np.isfinite(values)

    print(
        f"Day {i + 1}: "
        f"valid={np.sum(valid_day):,}, "
        f"min={np.nanmin(values):.6f}, "
        f"max={np.nanmax(values):.6f}, "
        f"mean={np.nanmean(values):.6f}"
    )


# ============================================================
# SHOW FIRST 20 POINTS
# ============================================================

print("\n\n[4] FIRST 20 POINT COMPARISONS")
print("-" * 70)

comparison = pd.DataFrame({
    "lat": points["lat"],
    "lon": points["lon"],
    "csv_rainfall_3day": csv_values,
    "corrected_source_rainfall_3day": corrected_3day,
    "difference": corrected_3day - csv_values
})

print(
    comparison.head(20).to_string(
        index=False
    )
)


# ============================================================
# SAVE DIAGNOSTIC TABLE
# ============================================================

output = (
    PROJECT
    / "processed"
    / "step64c_rainfall_source_validation.csv"
)

comparison.to_csv(
    output,
    index=False
)

print("\nSaved:")
print(output)


print("\n" + "=" * 70)
print("STEP 64C RAINFALL SOURCE VALIDATION COMPLETED")
print("=" * 70)

print("\nNO EXISTING RASTER OR CSV WAS MODIFIED.")