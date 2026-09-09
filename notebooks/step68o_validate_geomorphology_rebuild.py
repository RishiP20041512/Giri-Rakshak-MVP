from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import confusion_matrix


PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

POINTS = (
    PROJECT
    / "processed"
    / "ner_training_points.geojson"
)

ORIGINAL = (
    PROJECT
    / "processed"
    / "ner_geomorphology.csv"
)

RASTER = (
    PROJECT
    / "processed"
    / "predictors"
    / "geomorph_origin_250m.tif"
)

OUTPUT = (
    PROJECT
    / "processed"
    / "step68o_geomorphology_validation.csv"
)


print("=" * 75)
print("STEP 68O — VALIDATE REBUILT GEOMORPHOLOGY")
print("=" * 75)


# ============================================================
# ORIGIN MAPPING
# ============================================================

def origin_from_class(value):

    if pd.isna(value):
        return np.nan

    value = str(value)

    if value.startswith("Structural Origin"):
        return 5

    if value.startswith("Fluvial Origin"):
        return 2

    if value.startswith("Glacial Origin"):
        return 3

    if value.startswith("Denudational Origin"):
        return 1

    if value.startswith("Water Bodies"):
        return 6

    if value.startswith("Lacustrine Origin"):
        return 4

    return np.nan


# ============================================================
# LOAD ORIGINAL BHUVAN DATA
# ============================================================

original = pd.read_csv(
    ORIGINAL
)

print(
    f"\nOriginal observations: "
    f"{len(original)}"
)

original["expected_origin"] = (
    original["geomorphology"]
    .apply(origin_from_class)
)

expected_valid = (
    original["expected_origin"]
    .notna()
)

print(
    f"Original valid observations: "
    f"{expected_valid.sum()}"
)

print(
    f"Original missing observations: "
    f"{(~expected_valid).sum()}"
)


# ============================================================
# LOAD TRAINING POINTS
# ============================================================

points = gpd.read_file(
    POINTS
)

print(
    f"Training points: "
    f"{len(points)}"
)


# ============================================================
# MAKE SURE COORDINATES ARE WGS84
# ============================================================

points = points.to_crs(
    "EPSG:4326"
)

# Determine coordinate columns
if "longitude" in original.columns:
    lon_col = "longitude"
else:
    lon_col = "lon"

if "latitude" in original.columns:
    lat_col = "latitude"
else:
    lat_col = "lat"


# ============================================================
# MATCH ORIGINAL DATA TO TRAINING POINTS
# ============================================================

expected = []

for _, point in points.iterrows():

    lon = point.geometry.x
    lat = point.geometry.y

    distance = (
        (original[lon_col] - lon) ** 2
        +
        (original[lat_col] - lat) ** 2
    )

    idx = distance.idxmin()

    nearest = original.loc[idx]

    coordinate_match = (
        abs(
            nearest[lon_col]
            - lon
        ) < 1e-5
        and
        abs(
            nearest[lat_col]
            - lat
        ) < 1e-5
    )

    if coordinate_match:
        expected.append(
            nearest["expected_origin"]
        )
    else:
        expected.append(
            np.nan
        )


expected = np.array(
    expected,
    dtype=float
)


# ============================================================
# SAMPLE REBUILT RASTER
# ============================================================

with rasterio.open(
    RASTER
) as src:

    raster_points = points.to_crs(
        src.crs
    )

    coords = [
        (
            geom.x,
            geom.y
        )
        for geom in raster_points.geometry
    ]

    predicted = np.array(
        list(
            src.sample(coords)
        )
    )[:, 0]

    nodata = src.nodata

    raster_valid = (
        np.isfinite(predicted)
        &
        (
            predicted != nodata
        )
        &
        (
            predicted > 0
        )
    )


# ============================================================
# COMPARABLE OBSERVATIONS
# ============================================================

comparable = (
    expected_valid.values
    &
    raster_valid
)


expected_c = (
    expected[comparable]
    .astype(int)
)

predicted_c = (
    predicted[comparable]
    .astype(int)
)


# ============================================================
# ACCURACY
# ============================================================

agreement = (
    expected_c
    == predicted_c
)

accuracy = (
    agreement.mean()
    if len(agreement) > 0
    else np.nan
)


print("\n" + "=" * 75)
print("VALIDATION SUMMARY")
print("=" * 75)

print(
    f"Original valid: "
    f"{expected_valid.sum()}"
)

print(
    f"Raster valid at points: "
    f"{raster_valid.sum()}"
)

print(
    f"Comparable: "
    f"{comparable.sum()}"
)

print(
    f"Agreement: "
    f"{agreement.sum()}"
)

print(
    f"Agreement percentage: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    1, 2, 3, 4, 5, 6
]

names = {
    1: "Denudational",
    2: "Fluvial",
    3: "Glacial",
    4: "Lacustrine",
    5: "Structural",
    6: "Water Bodies",
}

if len(expected_c) > 0:

    cm = confusion_matrix(
        expected_c,
        predicted_c,
        labels=labels
    )

    print("\n" + "=" * 75)
    print("CONFUSION MATRIX")
    print("=" * 75)

    print(
        "Rows = expected"
    )

    print(
        "Columns = predicted"
    )

    header = (
        "Expected".ljust(20)
        +
        "".join(
            f"{names[x][:12]:>14}"
            for x in labels
        )
    )

    print(header)

    for i, code in enumerate(labels):

        row = (
            names[code].ljust(20)
            +
            "".join(
                f"{cm[i, j]:>14}"
                for j in range(
                    len(labels)
                )
            )
        )

        print(row)


# ============================================================
# SAVE POINT-LEVEL VALIDATION
# ============================================================

result = pd.DataFrame({
    "expected_origin": expected,
    "predicted_origin": predicted,
    "expected_valid": expected_valid.values,
    "raster_valid": raster_valid,
    "comparable": comparable,
    "correct": (
        np.where(
            comparable,
            expected == predicted,
            False
        )
    )
})

result.to_csv(
    OUTPUT,
    index=False
)


print(
    f"\nSaved:\n{OUTPUT}"
)

print("\n" + "=" * 75)
print("STEP 68O COMPLETE")
print("=" * 75)