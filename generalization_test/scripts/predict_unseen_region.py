import os
import joblib
import numpy as np
import rasterio

# ============================================================
# GIRIRAKSHAK
# Geographic Generalization Test
#
# IMPORTANT:
# This script DOES NOT retrain the model.
# It loads the existing trained Random Forest and applies it
# to a geographically unseen NER test area.
# ============================================================

MODEL_PATH = r"processed\random_forest_model.joblib"

DATA_DIR = r"generalization_test\test_data"

OUTPUT_DIR = r"generalization_test\outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. EXACT 8 FEATURES
# ============================================================

FEATURES = [
    "elevation",
    "slope",
    "aspect",
    "ndvi",
    "rainfall_3day",
    "soil_moisture",
    "dist_to_road",
    "dist_to_river"
]

FILES = [
    "elevation.tif",
    "slope.tif",
    "aspect.tif",
    "ndvi.tif",
    "rainfall_3day.tif",
    "soil_moisture.tif",
    "dist_to_road.tif",
    "dist_to_river.tif"
]


print("=" * 70)
print("GIRIRAKSHAK - GEOGRAPHIC GENERALIZATION TEST")
print("=" * 70)


# ============================================================
# 2. LOAD EXISTING MODEL
# ============================================================

print("\n[1/5] Loading existing Random Forest model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"\nModel not found:\n{os.path.abspath(MODEL_PATH)}"
    )

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print("Model type:", type(model).__name__)
print("Number of features expected:", model.n_features_in_)


if model.n_features_in_ != 8:
    raise ValueError(
        f"\nERROR: This model expects "
        f"{model.n_features_in_} features, not 8."
    )


# ============================================================
# 3. LOAD THE 8 TEST RASTERS
# ============================================================

print("\n[2/5] Loading test-area rasters...")

arrays = []

reference_profile = None
reference_shape = None
reference_transform = None
reference_crs = None


for feature, filename in zip(FEATURES, FILES):

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nMissing raster:\n{os.path.abspath(path)}"
        )

    print(f"  Loading: {feature}")

    with rasterio.open(path) as src:

        array = src.read(1).astype("float32")

        # First raster becomes the reference
        if reference_profile is None:

            reference_profile = src.profile.copy()
            reference_shape = array.shape
            reference_transform = src.transform
            reference_crs = src.crs

        else:

            # Check dimensions
            if array.shape != reference_shape:
                raise ValueError(
                    f"\nDimension mismatch in {filename}"
                )

            # Check alignment
            if src.transform != reference_transform:
                raise ValueError(
                    f"\nAlignment mismatch in {filename}"
                )

            # Check CRS
            if src.crs != reference_crs:
                raise ValueError(
                    f"\nCRS mismatch in {filename}"
                )

        arrays.append(array)


print("\nAll 8 rasters loaded successfully.")
print("All rasters have matching dimensions, CRS and alignment.")


# ============================================================
# 4. CREATE PIXEL-BASED FEATURE MATRIX
# ============================================================

print("\n[3/5] Creating feature matrix...")

stack = np.stack(arrays, axis=0)

bands, rows, cols = stack.shape

print("Features :", bands)
print("Rows     :", rows)
print("Columns  :", cols)

# Current shape:
#
#     (8, rows, columns)
#
# Required by Random Forest:
#
#     (pixels, 8)

X = stack.reshape(bands, -1).T

print("Feature matrix:", X.shape)


# ============================================================
# 5. IDENTIFY VALID PIXELS
# ============================================================

print("\n[4/5] Checking valid pixels...")

valid = np.all(np.isfinite(X), axis=1)

total_pixels = len(X)
valid_pixels = np.sum(valid)
invalid_pixels = np.sum(~valid)

print("Total pixels  :", total_pixels)
print("Valid pixels  :", valid_pixels)
print("Invalid pixels:", invalid_pixels)


if valid_pixels == 0:
    raise ValueError(
        "\nNo valid pixels were found."
    )


# ============================================================
# 6. RUN MODEL
# ============================================================

print("\n[5/5] Applying trained Random Forest...")

probability = np.full(
    X.shape[0],
    np.nan,
    dtype="float32"
)

# IMPORTANT:
# We are ONLY predicting.
# There is NO model.fit() here.

probability[valid] = model.predict_proba(
    X[valid]
)[:, 1]


susceptibility = probability.reshape(
    rows,
    cols
)


# ============================================================
# 7. SAVE PROBABILITY MAP
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "unseen_region_susceptibility_probability.tif"
)


profile = reference_profile.copy()

profile.update(
    dtype="float32",
    count=1,
    compress="lzw",
    nodata=-9999
)


output_array = np.where(
    np.isfinite(susceptibility),
    susceptibility,
    -9999
).astype("float32")


with rasterio.open(
    output_path,
    "w",
    **profile
) as dst:

    dst.write(
        output_array,
        1
    )


# ============================================================
# 8. REPORT RESULTS
# ============================================================

valid_values = susceptibility[
    np.isfinite(susceptibility)
]


print("\n" + "=" * 70)
print("SUCCESS - PREDICTION COMPLETE")
print("=" * 70)

print("\nOutput file:")
print(os.path.abspath(output_path))

print("\nSusceptibility probability statistics:")

print(
    "Minimum:",
    float(np.min(valid_values))
)

print(
    "Maximum:",
    float(np.max(valid_values))
)

print(
    "Mean:",
    float(np.mean(valid_values))
)

print("\nModel was NOT retrained.")

print("=" * 70)