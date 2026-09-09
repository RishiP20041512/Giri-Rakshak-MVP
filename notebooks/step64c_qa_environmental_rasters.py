"""
STEP 64C-QA — ENVIRONMENTAL RASTER QUALITY CHECK
================================================

Purpose
-------
Validate the newly created rainfall, soil-moisture and NDVI
250 m rasters against the original point-level predictor
values used during model development.

This step does NOT modify any raster or CSV.

Comparisons
-----------
1. Rainfall raster vs ner_rainfall.csv
2. Soil moisture raster vs ner_soil_moisture.csv
3. NDVI raster vs ner_ndvi.csv

For each predictor:
    - extract raster value at each training point
    - compare against original CSV value
    - calculate valid coverage
    - calculate MAE
    - calculate RMSE
    - calculate mean difference
    - calculate median difference
    - calculate Pearson correlation
    - calculate Spearman correlation

Important
---------
The raster and point values are expected to be similar but
may not be numerically identical because the final raster is
on a 250 m projected grid and source datasets have different
native resolutions.

The purpose is to detect major processing errors, not demand
bit-for-bit equality.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

from scipy.stats import pearsonr, spearmanr


# ================================================================
# 1. PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTOR_DIR = (
    PROJECT_ROOT
    / "processed"
    / "predictors"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "processed"
)

RAINFALL_CSV = (
    PROCESSED_DIR
    / "ner_rainfall.csv"
)

SOIL_CSV = (
    PROCESSED_DIR
    / "ner_soil_moisture.csv"
)

NDVI_CSV = (
    PROCESSED_DIR
    / "ner_ndvi.csv"
)

RAINFALL_RASTER = (
    PREDICTOR_DIR
    / "rainfall_3day_250m.tif"
)

SOIL_RASTER = (
    PREDICTOR_DIR
    / "soil_moisture_3day_250m.tif"
)

NDVI_RASTER = (
    PREDICTOR_DIR
    / "ndvi_250m.tif"
)

GRID_FILE = (
    PROCESSED_DIR
    / "step64a_final_ner_grid.tif"
)

OUTPUT_CSV = (
    PROCESSED_DIR
    / "step64c_environmental_raster_qa.csv"
)

POINT_OUTPUT_CSV = (
    PROCESSED_DIR
    / "step64c_environmental_point_comparison.csv"
)


# ================================================================
# 2. HEADER
# ================================================================

print("=" * 70)
print("STEP 64C-QA — ENVIRONMENTAL RASTER QUALITY CHECK")
print("=" * 70)


# ================================================================
# 3. CHECK FILES
# ================================================================

print("\n[1] Checking required files...")
print("-" * 70)

required_files = [
    RAINFALL_CSV,
    SOIL_CSV,
    NDVI_CSV,
    RAINFALL_RASTER,
    SOIL_RASTER,
    NDVI_RASTER,
    GRID_FILE,
]

for file in required_files:

    if not file.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )

    print(
        "  Found:",
        file.relative_to(PROJECT_ROOT)
    )


# ================================================================
# 4. LOAD CSV FILES
# ================================================================

print("\n[2] Loading original point-level predictors...")
print("-" * 70)

rain_df = pd.read_csv(
    RAINFALL_CSV
)

soil_df = pd.read_csv(
    SOIL_CSV
)

ndvi_df = pd.read_csv(
    NDVI_CSV
)

print(
    f"  Rainfall rows      : {len(rain_df):,}"
)

print(
    f"  Soil-moisture rows : {len(soil_df):,}"
)

print(
    f"  NDVI rows          : {len(ndvi_df):,}"
)


# ================================================================
# 5. DISPLAY COLUMNS
# ================================================================

print("\n[3] Checking CSV columns...")
print("-" * 70)

print(
    "  Rainfall columns:",
    list(rain_df.columns)
)

print(
    "  Soil columns:",
    list(soil_df.columns)
)

print(
    "  NDVI columns:",
    list(ndvi_df.columns)
)


# ================================================================
# 6. IDENTIFY VALUE COLUMNS
# ================================================================

def find_value_column(
    df,
    candidates,
    dataset_name
):

    for column in candidates:

        if column in df.columns:

            return column

    raise ValueError(
        f"Could not find predictor column for "
        f"{dataset_name}.\n"
        f"Available columns: {list(df.columns)}"
    )


rain_value_col = find_value_column(
    rain_df,
    [
        "rainfall_3day",
        "rainfall",
        "precipitation",
    ],
    "rainfall"
)

soil_value_col = find_value_column(
    soil_df,
    [
        "soil_moisture",
        "soil_moisture_3day",
    ],
    "soil moisture"
)

ndvi_value_col = find_value_column(
    ndvi_df,
    [
        "ndvi",
        "NDVI",
    ],
    "NDVI"
)

print(
    f"\n  Rainfall value column: "
    f"{rain_value_col}"
)

print(
    f"  Soil value column    : "
    f"{soil_value_col}"
)

print(
    f"  NDVI value column    : "
    f"{ndvi_value_col}"
)


# ================================================================
# 7. NORMALIZE COORDINATE COLUMN NAMES
# ================================================================

def normalize_coordinates(
    df,
    dataset_name
):

    df = df.copy()

    # ------------------------------------------------------------
    # Latitude
    # ------------------------------------------------------------

    if "lat" in df.columns:

        lat_col = "lat"

    elif "latitude" in df.columns:

        lat_col = "latitude"

    else:

        raise ValueError(
            f"No latitude column found in "
            f"{dataset_name}."
        )

    # ------------------------------------------------------------
    # Longitude
    # ------------------------------------------------------------

    if "lon" in df.columns:

        lon_col = "lon"

    elif "longitude" in df.columns:

        lon_col = "longitude"

    else:

        raise ValueError(
            f"No longitude column found in "
            f"{dataset_name}."
        )

    df[
        "latitude"
    ] = pd.to_numeric(
        df[lat_col],
        errors="coerce"
    )

    df[
        "longitude"
    ] = pd.to_numeric(
        df[lon_col],
        errors="coerce"
    )

    return df


rain_df = normalize_coordinates(
    rain_df,
    "rainfall"
)

soil_df = normalize_coordinates(
    soil_df,
    "soil moisture"
)

ndvi_df = normalize_coordinates(
    ndvi_df,
    "NDVI"
)


# ================================================================
# 8. LOAD FINAL GRID
# ================================================================

print("\n[4] Reading final grid...")
print("-" * 70)

with rasterio.open(
    GRID_FILE
) as src:

    grid_crs = src.crs
    grid_transform = src.transform
    grid_width = src.width
    grid_height = src.height

print(
    f"  CRS        : {grid_crs}"
)

print(
    f"  Dimensions : "
    f"{grid_width} x {grid_height}"
)


# ================================================================
# 9. RASTER EXTRACTION FUNCTION
# ================================================================

def extract_raster_values(
    dataframe,
    raster_file,
    name
):

    print(
        f"\n  Extracting {name} raster values..."
    )

    df = dataframe.copy()

    coordinates = list(
        zip(
            df["longitude"],
            df["latitude"]
        )
    )

    with rasterio.open(
        raster_file
    ) as src:

        # --------------------------------------------------------
        # Transform geographic WGS84 coordinates into raster
        # coordinate system.
        # --------------------------------------------------------

        if src.crs is None:

            raise ValueError(
                f"{name} raster has no CRS."
            )

        from rasterio.warp import transform

        xs, ys = transform(
            "EPSG:4326",
            src.crs,
            df["longitude"].to_numpy(),
            df["latitude"].to_numpy()
        )

        rows, cols = rowcol(
            src.transform,
            xs,
            ys
        )

        rows = np.asarray(
            rows
        )

        cols = np.asarray(
            cols
        )

        inside = (
            (rows >= 0)
            &
            (rows < src.height)
            &
            (cols >= 0)
            &
            (cols < src.width)
        )

        extracted = np.full(
            len(df),
            np.nan,
            dtype="float64"
        )

        data = src.read(
            1
        )

        valid_positions = np.where(
            inside
        )[0]

        for idx in valid_positions:

            value = data[
                rows[idx],
                cols[idx]
            ]

            if np.isfinite(value):

                extracted[
                    idx
                ] = float(value)

    df[
        f"{name}_raster"
    ] = extracted

    return df


# ================================================================
# 10. EXTRACT RASTER VALUES
# ================================================================

rain_df = extract_raster_values(
    rain_df,
    RAINFALL_RASTER,
    "rainfall"
)

soil_df = extract_raster_values(
    soil_df,
    SOIL_RASTER,
    "soil_moisture"
)

ndvi_df = extract_raster_values(
    ndvi_df,
    NDVI_RASTER,
    "ndvi"
)


# ================================================================
# 11. COMPARISON FUNCTION
# ================================================================

def calculate_comparison(
    df,
    original_column,
    raster_column,
    name
):

    original = pd.to_numeric(
        df[original_column],
        errors="coerce"
    ).to_numpy(
        dtype="float64"
    )

    raster = pd.to_numeric(
        df[raster_column],
        errors="coerce"
    ).to_numpy(
        dtype="float64"
    )

    valid = (
        np.isfinite(original)
        &
        np.isfinite(raster)
    )

    original_valid = (
        original[valid]
    )

    raster_valid = (
        raster[valid]
    )

    n_original = int(
        np.sum(
            np.isfinite(original)
        )
    )

    n_raster = int(
        np.sum(
            np.isfinite(raster)
        )
    )

    n_comparable = len(
        original_valid
    )

    print(
        f"\n{name}"
    )

    print("-" * 70)

    print(
        f"  Original valid : "
        f"{n_original:,}"
    )

    print(
        f"  Raster valid   : "
        f"{n_raster:,}"
    )

    print(
        f"  Comparable     : "
        f"{n_comparable:,}"
    )

    if n_comparable == 0:

        raise ValueError(
            f"No comparable values for {name}."
        )

    difference = (
        raster_valid
        -
        original_valid
    )

    abs_difference = np.abs(
        difference
    )

    squared_difference = (
        difference ** 2
    )

    mae = float(
        np.mean(
            abs_difference
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                squared_difference
            )
        )
    )

    mean_difference = float(
        np.mean(
            difference
        )
    )

    median_difference = float(
        np.median(
            difference
        )
    )

    if (
        len(original_valid) >= 2
        and
        np.std(original_valid) > 0
        and
        np.std(raster_valid) > 0
    ):

        pearson_r = float(
            pearsonr(
                original_valid,
                raster_valid
            )[0]
        )

        spearman_r = float(
            spearmanr(
                original_valid,
                raster_valid
            )[0]
        )

    else:

        pearson_r = np.nan
        spearman_r = np.nan

    print(
        f"  MAE            : "
        f"{mae:.6f}"
    )

    print(
        f"  RMSE           : "
        f"{rmse:.6f}"
    )

    print(
        f"  Mean difference: "
        f"{mean_difference:.6f}"
    )

    print(
        f"  Median difference: "
        f"{median_difference:.6f}"
    )

    print(
        f"  Pearson r      : "
        f"{pearson_r:.6f}"
    )

    print(
        f"  Spearman r     : "
        f"{spearman_r:.6f}"
    )

    result = {
        "predictor": name,
        "original_valid": n_original,
        "raster_valid": n_raster,
        "comparable": n_comparable,
        "coverage_percent": (
            100.0
            * n_comparable
            / n_original
            if n_original > 0
            else np.nan
        ),
        "original_min": float(
            np.min(original_valid)
        ),
        "original_max": float(
            np.max(original_valid)
        ),
        "raster_min": float(
            np.min(raster_valid)
        ),
        "raster_max": float(
            np.max(raster_valid)
        ),
        "mae": mae,
        "rmse": rmse,
        "mean_difference": mean_difference,
        "median_difference": median_difference,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
    }

    return result


# ================================================================
# 12. CALCULATE COMPARISONS
# ================================================================

print("\n" + "=" * 70)
print("POINT-LEVEL RASTER COMPARISON")
print("=" * 70)

results = []

results.append(
    calculate_comparison(
        rain_df,
        rain_value_col,
        "rainfall_raster",
        "rainfall_3day"
    )
)

results.append(
    calculate_comparison(
        soil_df,
        soil_value_col,
        "soil_moisture_raster",
        "soil_moisture"
    )
)

results.append(
    calculate_comparison(
        ndvi_df,
        ndvi_value_col,
        "ndvi_raster",
        "ndvi"
    )
)


# ================================================================
# 13. CREATE POINT COMPARISON TABLE
# ================================================================

print("\n[5] Creating point comparison table...")
print("-" * 70)

# ---------------------------------------------------------------
# Rainfall comparison
# ---------------------------------------------------------------

rain_compare = rain_df[
    [
        "latitude",
        "longitude",
        rain_value_col,
        "rainfall_raster",
    ]
].copy()

rain_compare = rain_compare.rename(
    columns={
        rain_value_col:
            "rainfall_original"
    }
)

rain_compare[
    "predictor"
] = "rainfall_3day"

# ---------------------------------------------------------------
# Soil comparison
# ---------------------------------------------------------------

soil_compare = soil_df[
    [
        "latitude",
        "longitude",
        soil_value_col,
        "soil_moisture_raster",
    ]
].copy()

soil_compare = soil_compare.rename(
    columns={
        soil_value_col:
            "soil_moisture_original"
    }
)

soil_compare[
    "predictor"
] = "soil_moisture"

soil_compare = soil_compare.rename(
    columns={
        "soil_moisture_raster":
            "raster_value"
    }
)

soil_compare[
    "original_value"
] = soil_compare[
    "soil_moisture_original"
]

soil_compare = soil_compare[
    [
        "latitude",
        "longitude",
        "predictor",
        "original_value",
        "raster_value",
    ]
]


# ---------------------------------------------------------------
# Rainfall standardized table
# ---------------------------------------------------------------

rain_compare = rain_compare.rename(
    columns={
        "rainfall_original":
            "original_value",
        "rainfall_raster":
            "raster_value",
    }
)

rain_compare = rain_compare[
    [
        "latitude",
        "longitude",
        "predictor",
        "original_value",
        "raster_value",
    ]
]


# ---------------------------------------------------------------
# NDVI comparison
# ---------------------------------------------------------------

ndvi_compare = ndvi_df[
    [
        "latitude",
        "longitude",
        ndvi_value_col,
        "ndvi_raster",
    ]
].copy()

ndvi_compare = ndvi_compare.rename(
    columns={
        ndvi_value_col:
            "original_value",
        "ndvi_raster":
            "raster_value",
    }
)

ndvi_compare[
    "predictor"
] = "ndvi"

ndvi_compare = ndvi_compare[
    [
        "latitude",
        "longitude",
        "predictor",
        "original_value",
        "raster_value",
    ]
]


# ---------------------------------------------------------------
# Combine
# ---------------------------------------------------------------

point_comparison = pd.concat(
    [
        rain_compare,
        soil_compare,
        ndvi_compare,
    ],
    ignore_index=True
)

point_comparison[
    "difference"
] = (
    point_comparison[
        "raster_value"
    ]
    -
    point_comparison[
        "original_value"
    ]
)

point_comparison.to_csv(
    POINT_OUTPUT_CSV,
    index=False
)

print(
    "  Saved:",
    POINT_OUTPUT_CSV.relative_to(
        PROJECT_ROOT
    )
)


# ================================================================
# 14. SAVE SUMMARY
# ================================================================

summary = pd.DataFrame(
    results
)

summary.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    "  Saved:",
    OUTPUT_CSV.relative_to(
        PROJECT_ROOT
    )
)


# ================================================================
# 15. DISPLAY SUMMARY
# ================================================================

print("\n[6] QA SUMMARY")
print("-" * 70)

display_columns = [
    "predictor",
    "original_valid",
    "raster_valid",
    "comparable",
    "coverage_percent",
    "mae",
    "rmse",
    "pearson_r",
    "spearman_r",
]

print(
    summary[
        display_columns
    ].to_string(
        index=False
    )
)


# ================================================================
# 16. BASIC SANITY CHECKS
# ================================================================

print("\n[7] Sanity checks...")
print("-" * 70)

# ---------------------------------------------------------------
# Rainfall
# ---------------------------------------------------------------

rain_raster_valid = rain_df[
    "rainfall_raster"
].dropna()

if len(rain_raster_valid) == 0:

    raise RuntimeError(
        "Rainfall raster has no valid point values."
    )

if (
    rain_raster_valid.min()
    < -0.001
):

    raise RuntimeError(
        "Rainfall raster contains negative values."
    )

print(
    "  Rainfall non-negative: PASSED"
)


# ---------------------------------------------------------------
# Soil moisture
# ---------------------------------------------------------------

soil_raster_valid = soil_df[
    "soil_moisture_raster"
].dropna()

if len(soil_raster_valid) == 0:

    raise RuntimeError(
        "Soil-moisture raster has no valid point values."
    )

if (
    soil_raster_valid.min()
    < -0.001
    or
    soil_raster_valid.max()
    > 1.001
):

    raise RuntimeError(
        "Soil-moisture raster contains "
        "values outside 0–1."
    )

print(
    "  Soil moisture 0–1 range: PASSED"
)


# ---------------------------------------------------------------
# NDVI
# ---------------------------------------------------------------

ndvi_raster_valid = ndvi_df[
    "ndvi_raster"
].dropna()

if len(ndvi_raster_valid) == 0:

    raise RuntimeError(
        "NDVI raster has no valid point values."
    )

if (
    ndvi_raster_valid.min()
    < -1.001
    or
    ndvi_raster_valid.max()
    > 1.001
):

    raise RuntimeError(
        "NDVI raster contains values "
        "outside -1 to 1."
    )

print(
    "  NDVI -1 to 1 range: PASSED"
)


# ================================================================
# 17. COVERAGE CHECK
# ================================================================

print("\n[8] Point coverage...")
print("-" * 70)

for row in results:

    print(
        f"  {row['predictor']:<18} "
        f"{row['coverage_percent']:.2f}%"
    )


# ================================================================
# 18. FINAL STATUS
# ================================================================

print("\n" + "=" * 70)
print("STEP 64C-QA COMPLETED")
print("=" * 70)

print("\nOutputs:")

print(
    "  processed\\step64c_environmental_raster_qa.csv"
)

print(
    "  processed\\step64c_environmental_point_comparison.csv"
)

print(
    "\nIMPORTANT:"
)

print(
    "  This QA does not automatically declare the "
    "rasters scientifically acceptable."
)

print(
    "  The comparison statistics must be reviewed "
    "before proceeding to the categorical predictors."
)

print("=" * 70)