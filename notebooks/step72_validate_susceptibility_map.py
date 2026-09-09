from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

POINTS_FILE = (
    PROCESSED /
    "ner_final_modeling_table_8factor.csv"
)

RASTER_FILE = (
    PROCESSED /
    "step70_8factor_susceptibility_probability.tif"
)

CLASS_FILE = (
    PROCESSED /
    "step71_8factor_susceptibility_classes.tif"
)

OUTPUT_POINTS = (
    PROCESSED /
    "step72_inventory_validation.csv"
)

OUTPUT_REPORT = (
    PROCESSED /
    "step72_susceptibility_validation_report.txt"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 72 — FINAL SUSCEPTIBILITY MAP VALIDATION")
print("=" * 70)


# ============================================================
# LOAD INVENTORY
# ============================================================

print("\nLoading inventory points...")

df = pd.read_csv(POINTS_FILE)

required = [
    "lat",
    "lon",
    "label",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

print(
    f"Inventory points: {len(df)}"
)


# ============================================================
# OPEN RASTERS
# ============================================================

print("\nOpening susceptibility raster...")

with rasterio.open(RASTER_FILE) as src:

    probability = src.read(1)

    probability_nodata = src.nodata

    probability_transform = src.transform
    probability_crs = src.crs

    print(
        "Probability raster CRS:",
        probability_crs
    )

    print(
        "Probability NoData:",
        probability_nodata
    )


print("\nOpening susceptibility classes...")

with rasterio.open(CLASS_FILE) as src:

    classes = src.read(1)

    class_transform = src.transform
    class_crs = src.crs

    print(
        "Class raster CRS:",
        class_crs
    )


# ============================================================
# VERIFY RASTERS
# ============================================================

if probability.shape != classes.shape:

    raise ValueError(
        "Probability and class raster dimensions differ."
    )

if probability_crs != class_crs:

    raise ValueError(
        "Probability and class raster CRS differ."
    )


# ============================================================
# TRANSFORM POINTS TO RASTER CRS
# ============================================================

from rasterio.warp import transform

xs, ys = transform(
    "EPSG:4326",
    probability_crs,
    df["lon"].to_numpy(),
    df["lat"].to_numpy(),
)


# ============================================================
# SAMPLE RASTER
# ============================================================

print("\nSampling susceptibility raster...")

rows, cols = rasterio.transform.rowcol(
    probability_transform,
    xs,
    ys,
)

rows = np.asarray(rows)
cols = np.asarray(cols)

height, width = probability.shape

inside = (
    (rows >= 0)
    & (rows < height)
    & (cols >= 0)
    & (cols < width)
)

scores = np.full(
    len(df),
    np.nan,
    dtype=np.float64,
)

sampled_classes = np.zeros(
    len(df),
    dtype=np.uint8,
)

scores[inside] = probability[
    rows[inside],
    cols[inside]
]

sampled_classes[inside] = classes[
    rows[inside],
    cols[inside]
]


# ============================================================
# VALID SAMPLE
# ============================================================

valid = (
    inside
    & np.isfinite(scores)
    & (
        scores != probability_nodata
    )
)

df["susceptibility_score"] = scores
df["susceptibility_class"] = sampled_classes
df["map_valid"] = valid.astype(int)


# ============================================================
# COVERAGE
# ============================================================

n_total = len(df)
n_valid = int(valid.sum())
n_missing = n_total - n_valid

print("\n" + "=" * 70)
print("POINT COVERAGE")
print("=" * 70)

print(
    f"Total inventory points : {n_total}"
)

print(
    f"Map-covered points     : {n_valid}"
)

print(
    f"Missing from map       : {n_missing}"
)

print(
    f"Point coverage         : "
    f"{100*n_valid/n_total:.2f}%"
)


# ============================================================
# VALIDATION DATA
# ============================================================

validation = df.loc[
    valid
].copy()

if validation["label"].nunique() < 2:

    raise ValueError(
        "Both landslide and background classes "
        "are required for ROC-AUC."
    )


# ============================================================
# ROC / PR AUC
# ============================================================

y_true = validation["label"].to_numpy()
y_score = validation[
    "susceptibility_score"
].to_numpy()

roc_auc = roc_auc_score(
    y_true,
    y_score
)

pr_auc = average_precision_score(
    y_true,
    y_score
)


# ============================================================
# GROUP STATISTICS
# ============================================================

background_scores = validation.loc[
    validation["label"] == 0,
    "susceptibility_score"
]

landslide_scores = validation.loc[
    validation["label"] == 1,
    "susceptibility_score"
]


print("\n" + "=" * 70)
print("SUSCEPTIBILITY VALIDATION")
print("=" * 70)

print(
    f"ROC-AUC : {roc_auc:.4f}"
)

print(
    f"PR-AUC  : {pr_auc:.4f}"
)

print("\nBackground points:")
print(
    f"n      : {len(background_scores)}"
)
print(
    f"mean   : {background_scores.mean():.6f}"
)
print(
    f"median : {background_scores.median():.6f}"
)

print("\nLandslide points:")
print(
    f"n      : {len(landslide_scores)}"
)
print(
    f"mean   : {landslide_scores.mean():.6f}"
)
print(
    f"median : {landslide_scores.median():.6f}"
)


# ============================================================
# CLASS DISTRIBUTION AT INVENTORY POINTS
# ============================================================

print("\n" + "=" * 70)
print("INVENTORY POINT CLASS DISTRIBUTION")
print("=" * 70)

class_names = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}

for class_id in range(1, 6):

    total_class = int(
        (
            validation["susceptibility_class"]
            == class_id
        ).sum()
    )

    landslide_class = int(
        (
            (validation["label"] == 1)
            &
            (
                validation[
                    "susceptibility_class"
                ]
                == class_id
            )
        ).sum()
    )

    background_class = int(
        (
            (validation["label"] == 0)
            &
            (
                validation[
                    "susceptibility_class"
                ]
                == class_id
            )
        ).sum()
    )

    print(
        f"{class_id} "
        f"{class_names[class_id]:10s}: "
        f"total={total_class:4d}, "
        f"landslide={landslide_class:4d}, "
        f"background={background_class:4d}"
    )


# ============================================================
# SAVE POINT VALIDATION
# ============================================================

df.to_csv(
    OUTPUT_POINTS,
    index=False
)


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    OUTPUT_REPORT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 72 — FINAL SUSCEPTIBILITY MAP VALIDATION\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Total inventory points: {n_total}\n"
    )

    f.write(
        f"Map-covered points: {n_valid}\n"
    )

    f.write(
        f"Missing points: {n_missing}\n"
    )

    f.write(
        f"Point coverage (%): "
        f"{100*n_valid/n_total:.4f}\n\n"
    )

    f.write(
        f"ROC-AUC: {roc_auc:.6f}\n"
    )

    f.write(
        f"PR-AUC: {pr_auc:.6f}\n\n"
    )

    f.write(
        "Background mean score: "
        f"{background_scores.mean():.6f}\n"
    )

    f.write(
        "Background median score: "
        f"{background_scores.median():.6f}\n"
    )

    f.write(
        "Landslide mean score: "
        f"{landslide_scores.mean():.6f}\n"
    )

    f.write(
        "Landslide median score: "
        f"{landslide_scores.median():.6f}\n\n"
    )

    f.write(
        "Inventory point class distribution:\n"
    )

    for class_id in range(1, 6):

        total_class = int(
            (
                validation[
                    "susceptibility_class"
                ]
                == class_id
            ).sum()
        )

        landslide_class = int(
            (
                (validation["label"] == 1)
                &
                (
                    validation[
                        "susceptibility_class"
                    ]
                    == class_id
                )
            ).sum()
        )

        background_class = int(
            (
                (validation["label"] == 0)
                &
                (
                    validation[
                        "susceptibility_class"
                    ]
                    == class_id
                )
            ).sum()
        )

        f.write(
            f"{class_id} - "
            f"{class_names[class_id]}: "
            f"total={total_class}, "
            f"landslide={landslide_class}, "
            f"background={background_class}\n"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 72 COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_POINTS)
print(OUTPUT_REPORT)

print("\nThe continuous susceptibility raster remains:")
print(RASTER_FILE)

print("\nThe classified susceptibility raster remains:")
print(CLASS_FILE)

print("=" * 70)