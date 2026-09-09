from pathlib import Path
import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

CLASS_FILE = (
    PROCESSED /
    "step71_8factor_susceptibility_classes.tif"
)

PROBABILITY_FILE = (
    PROCESSED /
    "step70_8factor_susceptibility_probability.tif"
)

OUTPUT_REPORT = (
    PROCESSED /
    "step73_final_map_statistics.txt"
)


# ============================================================
# CONSTANTS
# ============================================================

PIXEL_SIZE_M = 250
PIXEL_AREA_M2 = PIXEL_SIZE_M ** 2
PIXEL_AREA_KM2 = PIXEL_AREA_M2 / 1_000_000

CLASS_NAMES = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 73 — FINAL SUSCEPTIBILITY MAP STATISTICS")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not CLASS_FILE.exists():
    raise FileNotFoundError(
        f"Class raster not found:\n{CLASS_FILE}"
    )

if not PROBABILITY_FILE.exists():
    raise FileNotFoundError(
        f"Probability raster not found:\n{PROBABILITY_FILE}"
    )


# ============================================================
# READ CLASS RASTER
# ============================================================

print("\nReading classified susceptibility raster...")

with rasterio.open(CLASS_FILE) as src:

    classes = src.read(1)

    width = src.width
    height = src.height
    crs = src.crs
    transform = src.transform
    nodata = src.nodata

print(f"Width      : {width}")
print(f"Height     : {height}")
print(f"CRS        : {crs}")
print(f"Pixel size : {PIXEL_SIZE_M} m")
print(f"NoData     : {nodata}")


# ============================================================
# VALID CELLS
# ============================================================

valid = np.isin(
    classes,
    [1, 2, 3, 4, 5]
)

valid_cells = int(valid.sum())

total_cells = classes.size

nodata_cells = total_cells - valid_cells


# ============================================================
# CLASS STATISTICS
# ============================================================

statistics = {}

for class_id in range(1, 6):

    count = int(
        (classes == class_id).sum()
    )

    area_km2 = (
        count *
        PIXEL_AREA_KM2
    )

    percentage = (
        count /
        valid_cells *
        100
        if valid_cells > 0
        else 0
    )

    statistics[class_id] = {
        "count": count,
        "area_km2": area_km2,
        "percentage": percentage,
    }


# ============================================================
# TOTAL AREA
# ============================================================

total_valid_area_km2 = (
    valid_cells *
    PIXEL_AREA_KM2
)

total_raster_area_km2 = (
    total_cells *
    PIXEL_AREA_KM2
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MAP AREA STATISTICS")
print("=" * 70)

print(
    f"Total raster cells : "
    f"{total_cells:,}"
)

print(
    f"Valid cells        : "
    f"{valid_cells:,}"
)

print(
    f"NoData cells       : "
    f"{nodata_cells:,}"
)

print(
    f"Mapped area        : "
    f"{total_valid_area_km2:,.2f} km²"
)

print(
    f"Total raster area  : "
    f"{total_raster_area_km2:,.2f} km²"
)

print("\nSusceptibility classes:")

for class_id in range(1, 6):

    s = statistics[class_id]

    print(
        f"{class_id} "
        f"{CLASS_NAMES[class_id]:10s} : "
        f"{s['count']:>10,} cells | "
        f"{s['area_km2']:>10,.2f} km² | "
        f"{s['percentage']:>6.2f}%"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nFinal model:")
print("Random Forest")
print("8 conditioning factors")
print("700 trees")
print("max_features = sqrt")
print("min_samples_leaf = 5")
print("max_depth = None")


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL NORTHEAST INDIA LANDSLIDE "
        "SUSCEPTIBILITY MAP\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "MODEL: 8-FACTOR RANDOM FOREST\n\n"
    )

    f.write(
        "Conditioning factors:\n"
    )

    factors = [
        "1. Elevation",
        "2. Slope",
        "3. Rainfall",
        "4. Soil moisture",
        "5. NDVI",
        "6. Distance to road",
        "7. Lineament density",
        "8. Geomorphological origin",
    ]

    for factor in factors:
        f.write(f"{factor}\n")

    f.write("\n")

    f.write(
        "Excluded/unavailable factors:\n"
    )

    f.write(
        "Level-I LUCC: excluded from final model\n"
    )

    f.write(
        "Lithology: unavailable during final modelling\n"
    )

    f.write("\n")

    f.write(
        "Raster specification:\n"
    )

    f.write(
        "CRS: EPSG:6933\n"
    )

    f.write(
        "Pixel size: 250 m x 250 m\n"
    )

    f.write(
        "Pixel area: 0.0625 km²\n"
    )

    f.write("\n")

    f.write(
        f"Total raster cells: "
        f"{total_cells}\n"
    )

    f.write(
        f"Valid susceptibility cells: "
        f"{valid_cells}\n"
    )

    f.write(
        f"NoData cells: "
        f"{nodata_cells}\n"
    )

    f.write(
        f"Mapped area: "
        f"{total_valid_area_km2:.2f} km²\n"
    )

    f.write("\n")

    f.write(
        "Susceptibility class statistics:\n"
    )

    for class_id in range(1, 6):

        s = statistics[class_id]

        f.write(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]}: "
            f"{s['count']} cells, "
            f"{s['area_km2']:.2f} km², "
            f"{s['percentage']:.2f}%\n"
        )

    f.write("\n")

    f.write(
        "Classification thresholds:\n"
    )

    f.write(
        "Very Low: RF score < 0.20\n"
    )

    f.write(
        "Low: 0.20 <= RF score < 0.40\n"
    )

    f.write(
        "Moderate: 0.40 <= RF score < 0.60\n"
    )

    f.write(
        "High: 0.60 <= RF score < 0.80\n"
    )

    f.write(
        "Very High: RF score >= 0.80\n"
    )

    f.write("\n")

    f.write(
        "Important interpretation note:\n"
    )

    f.write(
        "The five classes represent relative "
        "Random Forest susceptibility scores. "
        "They should not be interpreted as "
        "calibrated probabilities of landslide "
        "occurrence.\n"
    )

    f.write(
        "\nThe map contains NoData where one or "
        "more required conditioning factors "
        "were unavailable. No predictor values "
        "were artificially imputed for mapping.\n"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 73 COMPLETE")
print("=" * 70)

print("\nFinal classified raster:")
print(CLASS_FILE)

print("\nContinuous probability raster:")
print(PROBABILITY_FILE)

print("\nFinal statistics report:")
print(OUTPUT_REPORT)

print("\nNext step: prepare the final map in QGIS.")

print("=" * 70)