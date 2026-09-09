from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

GRID_FILE = (
    PROJECT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

MASK_FILE = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

PREDICTOR_DIR = (
    PROJECT
    / "processed"
    / "predictors"
)

OUTPUT_FILE = (
    PROJECT
    / "processed"
    / "step68f_final_predictor_raster_audit.csv"
)


# ============================================================
# CANDIDATE FINAL PREDICTORS
# ============================================================

PREDICTORS = {
    "elevation":
        "elevation_250m.tif",

    "slope":
        "slope_250m.tif",

    "rainfall":
        "rainfall_3day_250m.tif",

    "soil_moisture":
        "soil_moisture_3day_250m.tif",

    "ndvi":
        "ndvi_250m.tif",

    "distance_to_road":
        "distance_to_road_250m.tif",

    "geomorphology":
        "geomorph_origin_250m.tif",

    # These are deliberately audited but NOT proposed
    # as final predictors because of previous validation.
    "lineament_EXCLUDED":
        "lineament_density_250m.tif",

    "lulc_EXCLUDED":
        "lulc_level1_250m.tif",
}


print("=" * 75)
print("STEP 68F — FINAL PREDICTOR RASTER AUDIT")
print("=" * 75)


# ============================================================
# LOAD FINAL GRID
# ============================================================

print("\nLoading final grid...")

with rasterio.open(GRID_FILE) as src:

    grid_crs = src.crs
    grid_width = src.width
    grid_height = src.height
    grid_transform = src.transform
    grid_res = src.res

    print(f"CRS: {grid_crs}")
    print(
        f"Dimensions: "
        f"{grid_width} x {grid_height}"
    )
    print(
        f"Resolution: {grid_res}"
    )


# ============================================================
# LOAD NER MASK
# ============================================================

print("\nLoading NER mask...")

with rasterio.open(MASK_FILE) as src:

    mask = src.read(1)

    mask_crs = src.crs

    if (
        src.width != grid_width
        or src.height != grid_height
    ):
        raise ValueError(
            "NER mask dimensions do not match final grid."
        )

    ner_mask = mask > 0

    ner_cells = int(
        ner_mask.sum()
    )

    print(
        f"NER valid cells: "
        f"{ner_cells:,}"
    )


# ============================================================
# AUDIT
# ============================================================

results = []

for name, filename in PREDICTORS.items():

    path = PREDICTOR_DIR / filename

    print("\n" + "-" * 75)
    print(name)
    print("-" * 75)

    if not path.exists():

        print(
            f"FILE MISSING: {path}"
        )

        results.append({
            "predictor": name,
            "file": filename,
            "exists": False,
            "crs": None,
            "width": None,
            "height": None,
            "resolution_x": None,
            "resolution_y": None,
            "aligned": False,
            "valid_cells_total": 0,
            "valid_cells_ner": 0,
            "ner_coverage_percent": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "nodata": None,
        })

        continue


    with rasterio.open(path) as src:

        arr = src.read(1)

        same_crs = (
            src.crs == grid_crs
        )

        same_dimensions = (
            src.width == grid_width
            and
            src.height == grid_height
        )

        same_resolution = (
            np.isclose(
                src.res[0],
                grid_res[0]
            )
            and
            np.isclose(
                src.res[1],
                grid_res[1]
            )
        )

        same_transform = (
            np.allclose(
                np.array(
                    src.transform
                ),
                np.array(
                    grid_transform
                )
            )
        )

        aligned = (
            same_crs
            and
            same_dimensions
            and
            same_resolution
            and
            same_transform
        )

        nodata = src.nodata

        valid = np.isfinite(arr)

        if nodata is not None:

            valid &= ~np.isclose(
                arr,
                nodata
            )

        valid_total = int(
            valid.sum()
        )

        # ----------------------------------------------------
        # NER coverage
        # ----------------------------------------------------

        if (
            arr.shape == ner_mask.shape
        ):

            valid_ner = (
                valid
                & ner_mask
            )

            valid_ner_cells = int(
                valid_ner.sum()
            )

        else:

            valid_ner_cells = 0

        coverage = (
            valid_ner_cells
            / ner_cells
            * 100
            if ner_cells > 0
            else 0
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if valid_ner_cells > 0:

            values = arr[
                valid_ner
            ].astype(float)

            min_value = float(
                np.min(values)
            )

            max_value = float(
                np.max(values)
            )

            mean_value = float(
                np.mean(values)
            )

            median_value = float(
                np.median(values)
            )

        else:

            min_value = np.nan
            max_value = np.nan
            mean_value = np.nan
            median_value = np.nan


        print(
            f"CRS: {src.crs}"
        )

        print(
            f"Dimensions: "
            f"{src.width} x {src.height}"
        )

        print(
            f"Resolution: {src.res}"
        )

        print(
            f"NoData: {nodata}"
        )

        print(
            f"Aligned with final grid: "
            f"{aligned}"
        )

        print(
            f"Valid cells total: "
            f"{valid_total:,}"
        )

        print(
            f"Valid cells inside NER: "
            f"{valid_ner_cells:,}"
        )

        print(
            f"NER coverage: "
            f"{coverage:.2f}%"
        )

        if valid_ner_cells > 0:

            print(
                f"Min: {min_value:.6f}"
            )

            print(
                f"Max: {max_value:.6f}"
            )

            print(
                f"Mean: {mean_value:.6f}"
            )

            print(
                f"Median: {median_value:.6f}"
            )


        results.append({
            "predictor": name,
            "file": filename,
            "exists": True,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution_x": src.res[0],
            "resolution_y": src.res[1],
            "aligned": aligned,
            "valid_cells_total": valid_total,
            "valid_cells_ner": valid_ner_cells,
            "ner_coverage_percent": coverage,
            "min": min_value,
            "max": max_value,
            "mean": mean_value,
            "median": median_value,
            "nodata": nodata,
        })


# ============================================================
# SAVE AUDIT
# ============================================================

audit = pd.DataFrame(results)

audit.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL AUDIT SUMMARY")
print("=" * 75)

print(
    audit[
        [
            "predictor",
            "exists",
            "aligned",
            "valid_cells_ner",
            "ner_coverage_percent"
        ]
    ].to_string(index=False)
)


# ============================================================
# USABLE FINAL PREDICTORS
# ============================================================

print("\n" + "=" * 75)
print("CANDIDATE FINAL PREDICTORS")
print("=" * 75)

candidate_names = [
    "elevation",
    "slope",
    "rainfall",
    "soil_moisture",
    "ndvi",
    "distance_to_road",
    "geomorphology",
]

candidate_audit = audit[
    audit["predictor"].isin(
        candidate_names
    )
].copy()

candidate_audit[
    "usable"
] = (
    candidate_audit["exists"]
    &
    candidate_audit["aligned"]
    &
    (
        candidate_audit[
            "ner_coverage_percent"
        ] >= 90
    )
)

print(
    candidate_audit[
        [
            
            "predictor",
            "aligned",
            "ner_coverage_percent",
            "usable"
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 75)
print("STEP 68F COMPLETE")
print("=" * 75)

print(
    f"\nSaved:\n{OUTPUT_FILE}"
)