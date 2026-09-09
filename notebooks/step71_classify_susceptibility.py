from pathlib import Path
import numpy as np
import rasterio


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

INPUT_FILE = (
    PROCESSED /
    "step70_8factor_susceptibility_probability.tif"
)

OUTPUT_FILE = (
    PROCESSED /
    "step71_8factor_susceptibility_classes.tif"
)

REPORT_FILE = (
    PROCESSED /
    "step71_classification_report.txt"
)


# ============================================================
# CLASSIFICATION
# ============================================================
#
# We use fixed RF-score intervals:
#
# 1 = Very Low      : < 0.20
# 2 = Low           : 0.20 - <0.40
# 3 = Moderate      : 0.40 - <0.60
# 4 = High          : 0.60 - <0.80
# 5 = Very High     : >= 0.80
#
# These are relative susceptibility-score classes,
# NOT calibrated probabilities of landslide occurrence.
# ============================================================

NODATA_INPUT = -9999.0
NODATA_OUTPUT = 0

THRESHOLDS = [
    0.20,
    0.40,
    0.60,
    0.80,
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 71 — 8-FACTOR SUSCEPTIBILITY CLASSIFICATION")
print("=" * 70)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input raster not found:\n{INPUT_FILE}"
    )

print("\nInput:")
print(INPUT_FILE)


# ============================================================
# PROCESS RASTER
# ============================================================

with rasterio.open(INPUT_FILE) as src:

    profile = src.profile.copy()

    profile.update(
        dtype="uint8",
        count=1,
        nodata=NODATA_OUTPUT,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_SAFER",
    )

    counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
    }

    min_value = np.inf
    max_value = -np.inf

    with rasterio.open(
        OUTPUT_FILE,
        "w",
        **profile
    ) as dst:

        for _, window in src.block_windows(1):

            probability = src.read(
                1,
                window=window
            ).astype(np.float32)

            # ------------------------------------------------
            # VALID CELLS
            # ------------------------------------------------

            valid = (
                np.isfinite(probability)
                &
                (probability != NODATA_INPUT)
            )

            classes = np.zeros(
                probability.shape,
                dtype=np.uint8
            )

            # ------------------------------------------------
            # CLASSIFY
            # ------------------------------------------------

            classes[
                valid & (probability < 0.20)
            ] = 1

            classes[
                valid &
                (probability >= 0.20) &
                (probability < 0.40)
            ] = 2

            classes[
                valid &
                (probability >= 0.40) &
                (probability < 0.60)
            ] = 3

            classes[
                valid &
                (probability >= 0.60) &
                (probability < 0.80)
            ] = 4

            classes[
                valid &
                (probability >= 0.80)
            ] = 5

            # ------------------------------------------------
            # COUNTS
            # ------------------------------------------------

            unique, values = np.unique(
                classes,
                return_counts=True
            )

            for u, v in zip(unique, values):

                counts[int(u)] += int(v)

            # ------------------------------------------------
            # SCORE RANGE
            # ------------------------------------------------

            if valid.any():

                block_values = probability[valid]

                min_value = min(
                    min_value,
                    float(block_values.min())
                )

                max_value = max(
                    max_value,
                    float(block_values.max())
                )

            # ------------------------------------------------
            # WRITE
            # ------------------------------------------------

            dst.write(
                classes,
                1,
                window=window
            )


# ============================================================
# RESULTS
# ============================================================

valid_cells = (
    counts[1] +
    counts[2] +
    counts[3] +
    counts[4] +
    counts[5]
)

total_cells = (
    valid_cells +
    counts[0]
)

print("\n" + "=" * 70)
print("CLASSIFICATION RESULTS")
print("=" * 70)

print(
    f"Total cells       : {total_cells:,}"
)

print(
    f"Valid cells       : {valid_cells:,}"
)

print(
    f"NoData cells      : {counts[0]:,}"
)

if valid_cells > 0:

    coverage = (
        valid_cells /
        total_cells *
        100
    )

    print(
        f"Coverage          : {coverage:.2f}%"
    )

    print(
        f"Minimum RF score  : {min_value:.6f}"
    )

    print(
        f"Maximum RF score  : {max_value:.6f}"
    )


# ============================================================
# CLASS TABLE
# ============================================================

class_names = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}

print("\nSusceptibility classes:")

for class_id in range(1, 6):

    n = counts[class_id]

    percentage = (
        n /
        valid_cells *
        100
        if valid_cells > 0
        else 0
    )

    print(
        f"{class_id} = "
        f"{class_names[class_id]:10s} : "
        f"{n:,} cells "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 71 — 8-FACTOR SUSCEPTIBILITY "
        "CLASSIFICATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        "Input raster:\n"
    )

    f.write(
        str(INPUT_FILE) + "\n\n"
    )

    f.write(
        "Classification thresholds:\n"
    )

    f.write(
        "1 Very Low    : RF score < 0.20\n"
    )

    f.write(
        "2 Low         : 0.20 <= RF score < 0.40\n"
    )

    f.write(
        "3 Moderate    : 0.40 <= RF score < 0.60\n"
    )

    f.write(
        "4 High        : 0.60 <= RF score < 0.80\n"
    )

    f.write(
        "5 Very High   : RF score >= 0.80\n\n"
    )

    f.write(
        "Important: classes represent relative "
        "RF susceptibility scores and are not "
        "calibrated probabilities of landslide "
        "occurrence.\n\n"
    )

    f.write(
        f"Total cells: {total_cells}\n"
    )

    f.write(
        f"Valid cells: {valid_cells}\n"
    )

    f.write(
        f"NoData cells: {counts[0]}\n"
    )

    if valid_cells > 0:

        f.write(
            f"Coverage (%): {coverage:.4f}\n"
        )

        f.write(
            f"Minimum score: {min_value:.6f}\n"
        )

        f.write(
            f"Maximum score: {max_value:.6f}\n"
        )

    f.write("\nClass distribution:\n")

    for class_id in range(1, 6):

        n = counts[class_id]

        percentage = (
            n /
            valid_cells *
            100
            if valid_cells > 0
            else 0
        )

        f.write(
            f"{class_id} - "
            f"{class_names[class_id]}: "
            f"{n} cells "
            f"({percentage:.2f}%)\n"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 71 COMPLETE")
print("=" * 70)

print("\nClassified raster:")
print(OUTPUT_FILE)

print("\nClassification report:")
print(REPORT_FILE)

print("\nClass legend:")
print("1 = Very Low")
print("2 = Low")
print("3 = Moderate")
print("4 = High")
print("5 = Very High")

print("=" * 70)