from pathlib import Path

import joblib
import pandas as pd

from step2_extract_predictors import extract_all_predictors


ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    ROOT
    / "processed"
    / "step69_production_8factor_rf.joblib"
)


# =========================================================
# GEOMORPHOLOGY CODE → MODEL CATEGORY
# =========================================================

GEOMORPH_MAP = {
    1: "Denudational Origin",
    2: "Fluvial Origin",
    3: "Glacial Origin",
    4: "Lacustrine Origin",
    5: "Structural Origin",
    6: "Water Bodies",
}


# =========================================================
# PREDICT AT COORDINATE
# =========================================================

def predict_at_coordinate(lat, lon):

    print("=" * 60)
    print("GIRI-RAKSHAK — COORDINATE PREDICTION")
    print("=" * 60)

    print(f"\nLatitude  : {lat}")
    print(f"Longitude : {lon}")

    # -----------------------------------------------------
    # STEP 1: Extract REAL 8 predictors
    # -----------------------------------------------------

    predictors = extract_all_predictors(
        lat,
        lon
    )

    # -----------------------------------------------------
    # STEP 2: Check for missing data
    # -----------------------------------------------------

    missing = [
        name
        for name, value in predictors.items()
        if value is None
    ]

    if missing:

        print("\n❌ PREDICTION NOT AVAILABLE")

        print("\nMissing predictors:")

        for name in missing:
            print(f"  - {name}")

        print(
            "\nNo synthetic or fake values will be used."
        )

        return None

    # -----------------------------------------------------
    # STEP 3: Load existing trained RF model
    # -----------------------------------------------------

    print("\nLoading trained RF model...")

    if not MODEL_PATH.exists():

        print("\n❌ Model file not found:")
        print(MODEL_PATH)

        return None

    model = joblib.load(MODEL_PATH)

    print("✅ Model loaded")

    # -----------------------------------------------------
    # STEP 4: Convert geomorphology code
    #         to exact training category
    # -----------------------------------------------------

    geomorph_code = int(
        predictors["geomorph_origin"]
    )

    if geomorph_code not in GEOMORPH_MAP:

        print(
            f"\n❌ Unknown geomorphology code: "
            f"{geomorph_code}"
        )

        print(
            "\nPrediction stopped."
        )

        return None

    predictors["geomorph_origin"] = GEOMORPH_MAP[
        geomorph_code
    ]

    # -----------------------------------------------------
    # STEP 5: Create dataframe
    #         using exact trained feature names
    # -----------------------------------------------------

    X = pd.DataFrame(
        [predictors],
        columns=[
            "elevation_m",
            "slope_deg",
            "rainfall_3day",
            "soil_moisture",
            "ndvi",
            "distance_to_road_m",
            "lineament_density",
            "geomorph_origin",
        ],
    )

    print("\nModel input:")
    print(X.to_string(index=False))

    # -----------------------------------------------------
    # STEP 6: RF probability prediction
    # -----------------------------------------------------

    probability = model.predict_proba(X)[0, 1]

    prediction = model.predict(X)[0]

    # -----------------------------------------------------
    # STEP 7: Convert probability to susceptibility level
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # STEP 8: OUTPUT
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("LANDSLIDE SUSCEPTIBILITY RESULT")
    print("=" * 60)

    print(
        f"\nRF probability : {probability:.4f}"
    )

    print(
        f"RF probability : {probability * 100:.2f}%"
    )

    print(
        f"RF class       : {prediction}"
    )

    print(
        f"Susceptibility : {level}"
    )

    print("\n" + "=" * 60)

    return {
        "latitude": lat,
        "longitude": lon,
        "susceptibility": float(probability),
        "prediction": int(prediction),
        "level": level,
        "predictors": predictors,
    }


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # GUWAHATI TEST COORDINATE
    # -----------------------------------------------------

    lat = 26.1445
    lon = 91.7362

    result = predict_at_coordinate(
        lat,
        lon
    )

    if result is None:

        print("\n❌ Prediction failed.")

    else:

        print(
            "\n✅ COORDINATE PREDICTION SUCCESSFUL"
        )

        print(
            "\nPrediction pipeline:"
        )

        print(
            "LAT/LON"
            " → 8 REAL FACTORS"
            " → TRAINED RF"
            " → SUSCEPTIBILITY"
        )


if __name__ == "__main__":
    main()