from pathlib import Path

import geopandas as gpd
import rasterio
import numpy as np


PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

MASK = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

POINTS = (
    PROJECT
    / "processed"
    / "ner_training_points.geojson"
)


print("=" * 75)
print("STEP 68I — VALIDATE CORRECT NER MASK AGAINST TRAINING POINTS")
print("=" * 75)


points = gpd.read_file(POINTS)

print(
    f"\nTraining points: {len(points)}"
)

with rasterio.open(MASK) as src:

    print(
        f"Raster CRS: {src.crs}"
    )

    points = points.to_crs(src.crs)

    coords = [
        (geom.x, geom.y)
        for geom in points.geometry
    ]

    rows, cols = zip(
        *[src.index(x, y) for x, y in coords]
    )

    rows = np.array(rows)
    cols = np.array(cols)

    inside = (
        (rows >= 0)
        & (rows < src.height)
        & (cols >= 0)
        & (cols < src.width)
    )

    values = np.full(
        len(points),
        0,
        dtype=np.uint8
    )

    values[inside] = src.read(
        1
    )[
        rows[inside],
        cols[inside]
    ]

    valid = (
        values == 1
    )

print("\n" + "=" * 75)
print("RESULT")
print("=" * 75)

print(
    f"Points inside raster: "
    f"{inside.sum()} / {len(points)}"
)

print(
    f"Points inside NER mask: "
    f"{valid.sum()} / {len(points)}"
)

print(
    f"Points outside NER mask: "
    f"{len(points) - valid.sum()}"
)

if valid.sum() == len(points):
    print(
        "\nSTATUS: PASSED"
    )
    print(
        "All training points fall inside the corrected NER mask."
    )
else:
    print(
        "\nSTATUS: CHECK REQUIRED"
    )


print("\n" + "=" * 75)
print("STEP 68I COMPLETE")
print("=" * 75)