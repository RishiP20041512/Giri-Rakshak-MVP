from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio


PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

POINTS = (
    PROJECT
    / "processed"
    / "ner_training_points.geojson"
)

MASK = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

PREDICTORS = {
    "elevation": (
        PROJECT
        / "processed"
        / "predictors"
        / "elevation_250m.tif"
    ),
    "slope": (
        PROJECT
        / "processed"
        / "predictors"
        / "slope_250m.tif"
    ),
    "rainfall": (
        PROJECT
        / "processed"
        / "predictors"
        / "rainfall_3day_250m.tif"
    ),
    "soil_moisture": (
        PROJECT
        / "processed"
        / "predictors"
        / "soil_moisture_3day_250m.tif"
    ),
    "ndvi": (
        PROJECT
        / "processed"
        / "predictors"
        / "ndvi_250m.tif"
    ),
    "distance_to_road": (
        PROJECT
        / "processed"
        / "predictors"
        / "distance_to_road_250m.tif"
    ),
    "geomorphology": (
        PROJECT
        / "processed"
        / "predictors"
        / "geomorph_origin_250m.tif"
    ),
    "lineament_density": (
        PROJECT
        / "processed"
        / "predictors"
        / "lineament_density_250m.tif"
    ),
}


print("=" * 75)
print("STEP 68K — VALIDATE REQUIRED PREDICTORS AT TRAINING POINTS")
print("=" * 75)


# ------------------------------------------------------------
# LOAD POINTS
# ------------------------------------------------------------

points = gpd.read_file(POINTS)

print(
    f"\nTraining points: {len(points)}"
)


# ------------------------------------------------------------
# CHECK EACH RASTER
# ------------------------------------------------------------

results = []

for name, filepath in PREDICTORS.items():

    print("\n" + "-" * 75)
    print(name.upper())
    print("-" * 75)

    if not filepath.exists():

        print(
            f"FILE MISSING: {filepath}"
        )

        continue

    with rasterio.open(filepath) as src:

        pts = points.to_crs(
            src.crs
        )

        coords = [
            (geom.x, geom.y)
            for geom in pts.geometry
        ]

        values = np.array(
            list(
                src.sample(coords)
            )
        )[:, 0]

        nodata = src.nodata

        if nodata is not None:

            valid = (
                np.isfinite(values)
                & (
                    values != nodata
                )
            )

        else:

            valid = np.isfinite(
                values
            )

        print(
            f"Raster CRS: {src.crs}"
        )

        print(
            f"Raster size: "
            f"{src.width} x {src.height}"
        )

        print(
            f"Raster resolution: "
            f"{src.res}"
        )

        print(
            f"Valid at points: "
            f"{valid.sum()} / {len(values)}"
        )

        print(
            f"Missing at points: "
            f"{(~valid).sum()}"
        )

        if valid.any():

            print(
                f"Point minimum: "
                f"{np.nanmin(values[valid]):.6f}"
            )

            print(
                f"Point maximum: "
                f"{np.nanmax(values[valid]):.6f}"
            )

            print(
                f"Point mean: "
                f"{np.nanmean(values[valid]):.6f}"
            )

        results.append({
            "predictor": name,
            "valid_points": int(valid.sum()),
            "missing_points": int((~valid).sum()),
            "coverage_percent": (
                valid.sum()
                / len(values)
                * 100
            )
        })


# ------------------------------------------------------------
# SAVE AUDIT
# ------------------------------------------------------------

audit = pd.DataFrame(results)

output = (
    PROJECT
    / "processed"
    / "step68k_required_predictor_point_audit.csv"
)

audit.to_csv(
    output,
    index=False
)


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("POINT-LEVEL SUMMARY")
print("=" * 75)

print(
    audit.to_string(
        index=False
    )
)

print(
    f"\nSaved:\n{output}"
)

print("\n" + "=" * 75)
print("STEP 68K COMPLETE")
print("=" * 75)