import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.mask import mask


# ============================================================
# STEP 72 — STATE-WISE SUSCEPTIBILITY ANALYSIS
# ============================================================

print("=" * 75)
print("STEP 72 — STATE-WISE 8-FACTOR SUSCEPTIBILITY")
print("=" * 75)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RASTER = os.path.join(
    BASE,
    "processed",
    "step71_final_8factor_susceptibility.tif"
)

BOUNDARY = os.path.join(
    BASE,
    "raw_data",
    "boundaries",
    "ner_8states.geojson"
)

OUTPUT = os.path.join(
    BASE,
    "processed",
    "step72_statewise_susceptibility.csv"
)

STATES = [
    "Arunāchal Pradesh",
    "Assam",
    "Manipur",
    "Meghālaya",
    "Mizoram",
    "Nāgāland",
    "Sikkim",
    "Tripura"
]


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(RASTER):
    raise FileNotFoundError(RASTER)

if not os.path.exists(BOUNDARY):
    raise FileNotFoundError(BOUNDARY)

print("\nInput files:")
print("OK:", RASTER)
print("OK:", BOUNDARY)


# ============================================================
# LOAD BOUNDARIES
# ============================================================

states = gpd.read_file(BOUNDARY)

print(f"\nBoundary features: {len(states)}")

# Find state-name column
name_column = None

for col in ["shapeName", "NAME_1", "NAME", "name", "State"]:
    if col in states.columns:
        name_column = col
        break

if name_column is None:
    raise ValueError(
        f"Could not find state name column. Columns: {list(states.columns)}"
    )

print("State name column:", name_column)


# ============================================================
# OPEN SUSCEPTIBILITY RASTER
# ============================================================

with rasterio.open(RASTER) as src:

    raster_crs = src.crs
    nodata = src.nodata

    states = states.to_crs(raster_crs)

    print("\nRaster:")
    print("CRS:", raster_crs)
    print("Size:", src.width, "x", src.height)
    print("NoData:", nodata)


    results = []


    # ========================================================
    # PROCESS EACH STATE
    # ========================================================

    for state_name in STATES:

        print("\n" + "-" * 60)
        print(state_name)

        matches = states[
            states[name_column].astype(str).str.strip() == state_name
        ]

        # Fallback: normalized comparison
        if len(matches) == 0:

            normalized_target = (
                state_name
                .lower()
                .replace("ā", "a")
                .replace("ī", "i")
                .replace("ṅ", "n")
                .replace("á", "a")
            )

            for idx, row in states.iterrows():

                candidate = str(row[name_column]).strip()

                normalized_candidate = (
                    candidate
                    .lower()
                    .replace("ā", "a")
                    .replace("ī", "i")
                    .replace("ṅ", "n")
                    .replace("á", "a")
                )

                if normalized_candidate == normalized_target:
                    matches = states.iloc[[idx]]
                    break

        if len(matches) == 0:

            print("WARNING: State boundary not found.")

            results.append({
                "state": state_name,
                "valid_cells": 0,
                "coverage_percent": 0,
                "mean_susceptibility": np.nan,
                "median_susceptibility": np.nan,
                "low_percent": np.nan,
                "moderate_percent": np.nan,
                "high_percent": np.nan,
                "very_high_percent": np.nan
            })

            continue


        geometry = [
            geom for geom in matches.geometry
            if geom is not None and not geom.is_empty
        ]

        try:

            data, transform = mask(
                src,
                geometry,
                crop=False,
                filled=False
            )

            arr = data[0]

            # Mask NoData and invalid probabilities
            valid = (
                (~np.ma.getmaskarray(arr)) &
                np.isfinite(arr) &
                (arr >= 0) &
                (arr <= 1)
            )

            values = np.asarray(arr[valid], dtype=np.float64)

        except Exception as e:

            print("ERROR:", e)

            values = np.array([])


        # ====================================================
        # STATISTICS
        # ====================================================

        if len(values) == 0:

            print("No valid susceptibility predictions.")

            results.append({
                "state": state_name,
                "valid_cells": 0,
                "coverage_percent": 0,
                "mean_susceptibility": np.nan,
                "median_susceptibility": np.nan,
                "low_percent": np.nan,
                "moderate_percent": np.nan,
                "high_percent": np.nan,
                "very_high_percent": np.nan
            })

            continue


        # Class percentages
        low = np.sum(values < 0.20)
        moderate = np.sum(
            (values >= 0.20) & (values < 0.40)
        )
        high = np.sum(
            (values >= 0.40) & (values < 0.60)
        )
        very_high = np.sum(values >= 0.60)


        # ====================================================
        # COVERAGE
        # ====================================================
        #
        # Estimate state area pixels from geometry mask.
        # ====================================================

        try:

            state_mask, _ = mask(
                src,
                geometry,
                crop=False,
                filled=False
            )

            total_state_cells = np.sum(
                ~np.ma.getmaskarray(state_mask[0])
            )

            coverage = (
                len(values) / total_state_cells * 100
                if total_state_cells > 0 else 0
            )

        except Exception:

            coverage = np.nan


        results.append({
            "state": state_name,
            "valid_cells": len(values),
            "coverage_percent": coverage,
            "mean_susceptibility": np.mean(values),
            "median_susceptibility": np.median(values),
            "low_percent": low / len(values) * 100,
            "moderate_percent": moderate / len(values) * 100,
            "high_percent": high / len(values) * 100,
            "very_high_percent": very_high / len(values) * 100
        })


        print(f"Valid cells : {len(values):,}")
        print(f"Coverage    : {coverage:.2f}%")
        print(f"Mean        : {np.mean(values):.4f}")
        print(f"Median      : {np.median(values):.4f}")
        print(f"Low         : {low / len(values) * 100:.2f}%")
        print(f"Moderate    : {moderate / len(values) * 100:.2f}%")
        print(f"High        : {high / len(values) * 100:.2f}%")
        print(f"Very High   : {very_high / len(values) * 100:.2f}%")


# ============================================================
# SAVE CSV
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT,
    index=False,
    float_format="%.4f"
)


# ============================================================
# FINAL TABLE
# ============================================================

print("\n")
print("=" * 75)
print("STATE-WISE RESULTS")
print("=" * 75)

print(
    df.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT)

print("\n" + "=" * 75)
print("STEP 72 COMPLETE")
print("=" * 75)
