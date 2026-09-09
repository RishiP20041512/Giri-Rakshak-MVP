from pathlib import Path
import json

import rasterio
from rasterio.warp import transform
import joblib
import pandas as pd

from step2_extract_predictors import extract_all_predictors


ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "processed"
    / "step69_production_8factor_rf.joblib"
)


GEOMORPH_MAP = {
    1: "Denudational Origin",
    2: "Fluvial Origin",
    3: "Glacial Origin",
    4: "Lacustrine Origin",
    5: "Structural Origin",
    6: "Water Bodies",
}


FEATURE_COLUMNS = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
    "geomorph_origin",
]


def predict_any_location(lat, lon):

    print("=" * 70)
    print("GIRI-RAKSHAK — ANY LOCATION PREDICTION")
    print("=" * 70)

    print(f"\nLatitude : {lat}")
    print(f"Longitude: {lon}")

    # ========================================================
    # 1. EXTRACT REAL PREDICTORS
    # ========================================================

    print("\nExtracting REAL predictor values...")
    print("-" * 60)

    predictors = extract_all_predictors(
        lat,
        lon
    )

    for name, value in predictors.items():

        if value is None:
            print(
                f"{name:<25} ❌ MISSING"
            )
        else:
            print(
                f"{name:<25} ✅ {value}"
            )

    # ========================================================
    # 2. CHECK MISSING DATA
    # ========================================================

    missing = [
        name
        for name, value in predictors.items()
        if value is None
    ]

    if missing:

        print("\n" + "!" * 70)

        print(
            "PREDICTION UNAVAILABLE"
        )

        print(
            "Missing real predictor(s):"
        )

        for name in missing:
            print(
                f"  - {name}"
            )

        print(
            "\nNo synthetic value will be substituted."
        )

        print("!" * 70)

        return None

    # ========================================================
    # 3. CONVERT GEOMORPHOLOGY CODE
    # ========================================================

    geomorph_code = int(
        predictors["geomorph_origin"]
    )

    if geomorph_code not in GEOMORPH_MAP:

        print(
            "\nUnknown geomorphology code:",
            geomorph_code
        )

        return None

    predictors["geomorph_origin"] = (
        GEOMORPH_MAP[geomorph_code]
    )

    # ========================================================
    # 4. LOAD TRAINED RF
    # ========================================================

    print("\nLoading trained RF model...")

    model = joblib.load(
        MODEL_PATH
    )

    print("✅ Model loaded")

    # ========================================================
    # 5. BUILD MODEL INPUT
    # ========================================================

    X = pd.DataFrame(
        [predictors],
        columns=FEATURE_COLUMNS
    )

    print("\nModel input:")
    print(X.to_string(index=False))

    # ========================================================
    # 6. RF PREDICTION
    # ========================================================

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        model.predict(X)[0]
    )

    # ========================================================
    # 7. CLASSIFICATION
    # ========================================================

    if probability < 0.20:
        level = "VERY LOW"

    elif probability < 0.40:
        level = "LOW"

    elif probability < 0.60:
        level = "MODERATE"

    elif probability < 0.80:
        level = "HIGH"

    else:
        level = "VERY HIGH"

    # ========================================================
    # 8. RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("LANDSLIDE SUSCEPTIBILITY")
    print("=" * 70)

    print(
        f"\nRF score       : {probability:.4f}"
    )

    print(
        f"RF score       : {probability * 100:.2f}%"
    )

    print(
        f"RF class       : {prediction}"
    )

    print(
        f"Susceptibility: {level}"
    )

    print("\n" + "=" * 70)

    return {
        "latitude": lat,
        "longitude": lon,
        "susceptibility": probability,
        "prediction": prediction,
        "level": level,
        "predictors": predictors,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    # Change ONLY these two numbers
    # to test any location in NER.

    LAT = 27.4728
    LON = 94.9120

    result = predict_any_location(
        LAT,
        LON
    )
