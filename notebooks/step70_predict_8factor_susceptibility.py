from pathlib import Path
import warnings

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"

MODEL_FILE = (
    PROCESSED /
    "step69_production_8factor_rf.joblib"
)

MASK_FILE = (
    PROCESSED /
    "step64a_final_ner_mask.tif"
)

PREDICTOR_DIR = PROCESSED / "predictors"

OUTPUT_FILE = (
    PROCESSED /
    "step70_8factor_susceptibility_probability.tif"
)

OUTPUT_COVERAGE = (
    PROCESSED /
    "step70_prediction_coverage.txt"
)


# ============================================================
# PREDICTOR FILES
# ============================================================

PREDICTORS = {
    "elevation_m":
        PREDICTOR_DIR / "elevation_250m.tif",

    "slope_deg":
        PREDICTOR_DIR / "slope_250m.tif",

    "rainfall_3day":
        PREDICTOR_DIR / "rainfall_3day_250m.tif",

    "soil_moisture":
        PREDICTOR_DIR / "soil_moisture_3day_250m.tif",

    "ndvi":
        PREDICTOR_DIR / "ndvi_250m.tif",

    "distance_to_road_m":
        PREDICTOR_DIR / "distance_to_road_250m.tif",

    "lineament_density":
        PREDICTOR_DIR / "lineament_density_250m.tif",

    "geomorph_origin":
        PREDICTOR_DIR / "geomorph_origin_250m.tif",
}


# ============================================================
# SETTINGS
# ============================================================

NODATA = -9999.0

# Process the raster in manageable windows.
BLOCK_SIZE = 512


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 70 — 8-FACTOR SUSCEPTIBILITY RASTER")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking required files...")

required_files = [
    MODEL_FILE,
    MASK_FILE,
]

required_files.extend(PREDICTORS.values())

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print("OK:", path.name)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading production model...")

model = joblib.load(MODEL_FILE)

print("Model type:", type(model))

if not hasattr(model, "predict_proba"):

    raise TypeError(
        "Loaded model does not support predict_proba()."
    )

print("predict_proba() available.")


# ============================================================
# OPEN REFERENCE MASK
# ============================================================

print("\nOpening final NER mask...")

with rasterio.open(MASK_FILE) as mask_src:

    width = mask_src.width
    height = mask_src.height

    crs = mask_src.crs
    transform = mask_src.transform

    mask_profile = mask_src.profile.copy()

    print("Width :", width)
    print("Height:", height)
    print("CRS   :", crs)
    print("Transform:", transform)


# ============================================================
# OPEN ALL PREDICTORS
# ============================================================

print("\nOpening predictor rasters...")

sources = {}

for name, path in PREDICTORS.items():

    src = rasterio.open(path)

    sources[name] = src

    print(
        f"{name:22s} "
        f"{src.width} x {src.height} | "
        f"{src.crs}"
    )


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

print("\nChecking raster alignment...")

for name, src in sources.items():

    if src.width != width or src.height != height:

        raise ValueError(
            f"{name}: dimensions do not match final grid."
        )

    if src.crs != crs:

        raise ValueError(
            f"{name}: CRS does not match final grid."
        )

    if not np.allclose(
        tuple(src.transform),
        tuple(transform),
        rtol=0,
        atol=1e-8,
    ):

        raise ValueError(
            f"{name}: transform does not match final grid."
        )

print("All predictor rasters are aligned.")


# ============================================================
# OUTPUT PROFILE
# ============================================================

profile = mask_profile.copy()

profile.update(
    driver="GTiff",
    dtype="float32",
    count=1,
    nodata=NODATA,
    compress="deflate",
    predictor=3,
    tiled=True,
    blockxsize=512,
    blockysize=512,
    BIGTIFF="IF_SAFER",
)


# ============================================================
# COUNTERS
# ============================================================

total_ner_cells = 0
valid_prediction_cells = 0

prediction_min = np.inf
prediction_max = -np.inf
prediction_sum = 0.0

histogram = np.zeros(10, dtype=np.int64)


# ============================================================
# PREDICTOR ORDER
# ============================================================

FEATURE_ORDER = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
    "geomorph_origin",
]


# ============================================================
# GENERATE RASTER
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SUSCEPTIBILITY PROBABILITY")
print("=" * 70)

with rasterio.open(MASK_FILE) as mask_src, \
     rasterio.open(
         OUTPUT_FILE,
         "w",
         **profile
     ) as dst:

    for row_start in range(
        0,
        height,
        BLOCK_SIZE
    ):

        for col_start in range(
            0,
            width,
            BLOCK_SIZE
        ):

            row_stop = min(
                row_start + BLOCK_SIZE,
                height
            )

            col_stop = min(
                col_start + BLOCK_SIZE,
                width
            )

            h = row_stop - row_start
            w = col_stop - col_start

            window = Window(
                col_start,
                row_start,
                w,
                h
            )

            # ------------------------------------------------
            # READ NER MASK
            # ------------------------------------------------

            ner_mask = mask_src.read(
                1,
                window=window
            )

            ner_cells = ner_mask == 1

            total_ner_cells += int(
                ner_cells.sum()
            )

            # ------------------------------------------------
            # READ PREDICTORS
            # ------------------------------------------------

            arrays = {}

            valid = ner_cells.copy()

            for name in FEATURE_ORDER:

                arr = sources[name].read(
                    1,
                    window=window
                ).astype(
                    np.float32
                )

                arrays[name] = arr

                src_nodata = sources[name].nodata

                if src_nodata is not None:

                    valid &= (
                        np.isfinite(arr)
                        &
                        (arr != src_nodata)
                    )

                else:

                    valid &= np.isfinite(arr)

            # ------------------------------------------------
            # OUTPUT BLOCK
            # ------------------------------------------------

            output = np.full(
                (h, w),
                NODATA,
                dtype=np.float32
            )

            n_valid = int(valid.sum())

            if n_valid > 0:

                # --------------------------------------------
                # BUILD DATAFRAME
                # --------------------------------------------

                import pandas as pd

                data = {}

                for name in FEATURE_ORDER:

                    data[name] = arrays[name][valid]

                X = pd.DataFrame(
                    data,
                    columns=FEATURE_ORDER
                )

                # --------------------------------------------
                # GEOMORPHOLOGY
                # --------------------------------------------
                #
                # Raster classes:
                #
                # 1 = Denudational
                # 2 = Fluvial
                # 3 = Glacial
                # 4 = Lacustrine
                # 5 = Structural
                # 6 = Water Bodies
                #
                # The model was trained using the textual
                # geomorph_origin categories.
                #

                geomorph_mapping = {
                    1: "Denudational",
                    2: "Fluvial",
                    3: "Glacial",
                    4: "Lacustrine",
                    5: "Structural",
                    6: "Water Bodies",
                }

                X["geomorph_origin"] = (
                    X["geomorph_origin"]
                    .round()
                    .astype(np.int16)
                    .map(geomorph_mapping)
                )

                # Unknown geomorphology should not be
                # predicted.

                geom_valid = (
                    X["geomorph_origin"]
                    .notna()
                    .to_numpy()
                )

                if geom_valid.sum() > 0:

                    X_valid = X.loc[
                        geom_valid
                    ].copy()

                    probabilities = (
                        model
                        .predict_proba(
                            X_valid
                        )[:, 1]
                    )

                    valid_indices = np.flatnonzero(
                        valid
                    )

                    valid_indices = (
                        valid_indices[geom_valid]
                    )

                    output_flat = output.ravel()

                    output_flat[
                        valid_indices
                    ] = probabilities.astype(
                        np.float32
                    )

                    output = output_flat.reshape(
                        h,
                        w
                    )

                    n_predicted = len(
                        probabilities
                    )

                    valid_prediction_cells += (
                        n_predicted
                    )

                    prediction_min = min(
                        prediction_min,
                        float(
                            probabilities.min()
                        )
                    )

                    prediction_max = max(
                        prediction_max,
                        float(
                            probabilities.max()
                        )
                    )

                    prediction_sum += float(
                        probabilities.sum()
                    )

                    # ----------------------------------------
                    # 10-bin probability histogram
                    # ----------------------------------------

                    bins = np.linspace(
                        0.0,
                        1.0,
                        11
                    )

                    counts, _ = np.histogram(
                        probabilities,
                        bins=bins
                    )

                    histogram += counts

            # ------------------------------------------------
            # WRITE BLOCK
            # ------------------------------------------------

            dst.write(
                output,
                1,
                window=window
            )

        print(
            f"Processed rows "
            f"{row_start + 1:,} - "
            f"{row_stop:,} / "
            f"{height:,}"
        )


# ============================================================
# CLOSE INPUT RASTERS
# ============================================================

for src in sources.values():

    src.close()


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 70 RESULTS")
print("=" * 70)

print(
    f"NER cells              : "
    f"{total_ner_cells:,}"
)

print(
    f"Valid prediction cells : "
    f"{valid_prediction_cells:,}"
)

coverage = (
    100.0 * valid_prediction_cells /
    total_ner_cells
    if total_ner_cells > 0
    else 0
)

print(
    f"Prediction coverage    : "
    f"{coverage:.2f}%"
)

if valid_prediction_cells > 0:

    prediction_mean = (
        prediction_sum /
        valid_prediction_cells
    )

    print(
        f"Minimum susceptibility : "
        f"{prediction_min:.6f}"
    )

    print(
        f"Maximum susceptibility : "
        f"{prediction_max:.6f}"
    )

    print(
        f"Mean susceptibility    : "
        f"{prediction_mean:.6f}"
    )


# ============================================================
# HISTOGRAM
# ============================================================

print("\nProbability distribution:")

for i in range(10):

    lower = i / 10
    upper = (i + 1) / 10

    print(
        f"{lower:.1f} - {upper:.1f}: "
        f"{histogram[i]:,}"
    )


# ============================================================
# SAVE COVERAGE REPORT
# ============================================================

with open(
    OUTPUT_COVERAGE,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 70 — 8-FACTOR SUSCEPTIBILITY "
        "PREDICTION COVERAGE\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        f"Total NER cells: "
        f"{total_ner_cells}\n"
    )

    f.write(
        f"Valid prediction cells: "
        f"{valid_prediction_cells}\n"
    )

    f.write(
        f"Prediction coverage (%): "
        f"{coverage:.4f}\n"
    )

    if valid_prediction_cells > 0:

        f.write(
            f"Minimum susceptibility: "
            f"{prediction_min:.6f}\n"
        )

        f.write(
            f"Maximum susceptibility: "
            f"{prediction_max:.6f}\n"
        )

        f.write(
            f"Mean susceptibility: "
            f"{prediction_mean:.6f}\n"
        )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 70 COMPLETE")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nCoverage report:")
print(OUTPUT_COVERAGE)

print("\nNoData = -9999")
print("Valid cells contain RF landslide susceptibility")
print("probability estimates between 0 and 1.")

print("=" * 70)