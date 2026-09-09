from pathlib import Path
import requests
import pandas as pd

from dynamic.config import EAST_KHASI_HILLS
from dynamic.rainfall_trigger import (
    antecedent_wetness_index,
    simplified_two_window_trigger,
    combined_trigger_score,
    alert_state,
)
from risk_fusion import fuse_risk

from step3_coordinate_prediction import predict_at_coordinate


# ============================================================
# LOCATION
# ============================================================

LAT = 26.1445
LON = 91.7362

LOCATION_NAME = "Guwahati, Assam"


# ============================================================
# OPEN-METEO
# ============================================================

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather_rainfall(lat, lon):

    params = {
        "latitude": lat,
        "longitude": lon,

        # We need recent rainfall for antecedent wetness
        "past_days": 15,

        # Future rainfall
        "forecast_days": 7,

        "hourly": (
            "precipitation"
        ),

        "timezone": "auto",
    }

    print("\nFetching real rainfall from Open-Meteo...")

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": pd.to_datetime(hourly["time"]),
        "rainfall_mm": hourly["precipitation"],
    })

    df = df.set_index("time")

    df["rainfall_mm"] = (
        pd.to_numeric(
            df["rainfall_mm"],
            errors="coerce"
        )
        .fillna(0.0)
    )

    print("Rainfall records:", len(df))
    print(
        "Rainfall period:",
        df.index.min(),
        "to",
        df.index.max()
    )

    return df


# ============================================================
# FORECAST RAINFALL CALCULATION
# ============================================================

def calculate_forecast_windows(rainfall_df):

    now = rainfall_df.index[-1]

    # The API data includes past + future.
    # Find the approximate current time.
    current_time = pd.Timestamp.now(
        tz=rainfall_df.index.tz
    ).floor("h")

    # Use the closest available timestamp
    closest_idx = rainfall_df.index.get_indexer(
        [current_time],
        method="nearest"
    )[0]

    current_time = rainfall_df.index[closest_idx]

    # --------------------------------------------------------
    # Future rainfall
    # --------------------------------------------------------

    next_24 = rainfall_df.loc[
        (rainfall_df.index > current_time)
        &
        (
            rainfall_df.index
            <= current_time + pd.Timedelta(hours=24)
        )
    ]

    next_72 = rainfall_df.loc[
        (rainfall_df.index > current_time)
        &
        (
            rainfall_df.index
            <= current_time + pd.Timedelta(hours=72)
        )
    ]

    rain24 = float(next_24["rainfall_mm"].sum())
    rain72 = float(next_72["rainfall_mm"].sum())

    return current_time, rain24, rain72


# ============================================================
# FORECAST TRIGGER
# ============================================================

def calculate_forecast_trigger(
    rainfall_df,
    current_time,
    cfg,
):

    # --------------------------------------------------------
    # Build daily rainfall series
    # --------------------------------------------------------

    daily_rainfall = (
        rainfall_df["rainfall_mm"]
        .resample("D")
        .sum()
    )

    # Existing Giri-Rakshak API calculation
    api_series = antecedent_wetness_index(
        daily_rainfall,
        cfg
    )

    # --------------------------------------------------------
    # Future timestamps
    # --------------------------------------------------------

    future_times = [
        current_time + pd.Timedelta(hours=24),
        current_time + pd.Timedelta(hours=72),
    ]

    results = []

    for future_time in future_times:

        # ----------------------------------------------------
        # 24h rainfall before future time
        # ----------------------------------------------------

        start_24 = (
            future_time
            - pd.Timedelta(hours=24)
        )

        rainfall_24 = rainfall_df.loc[
            (rainfall_df.index > start_24)
            &
            (rainfall_df.index <= future_time),
            "rainfall_mm",
        ].sum()

        # ----------------------------------------------------
        # 72h rainfall before future time
        # ----------------------------------------------------

        start_72 = (
            future_time
            - pd.Timedelta(hours=72)
        )

        rainfall_72 = rainfall_df.loc[
            (rainfall_df.index > start_72)
            &
            (rainfall_df.index <= future_time),
            "rainfall_mm",
        ].sum()

        rainfall_24 = float(rainfall_24)
        rainfall_72 = float(rainfall_72)

        # ----------------------------------------------------
        # EXISTING ID LOGIC
        # ----------------------------------------------------

        id_ratio = simplified_two_window_trigger(
            rainfall_24,
            rainfall_72,
            cfg
        )

        # ----------------------------------------------------
        # EXISTING API LOGIC
        # ----------------------------------------------------

        future_day = pd.Timestamp(
            future_time
        ).normalize()

        if future_day in api_series.index:
            api_value = float(
                api_series.loc[future_day]
            )
        else:
            api_value = float(
                api_series.iloc[-1]
            )

        # ----------------------------------------------------
        # EXISTING TRIGGER SCORE
        # ----------------------------------------------------

        trigger_score = combined_trigger_score(
            id_ratio,
            api_value,
            cfg
        )

        state = alert_state(
            trigger_score,
            cfg
        )

        results.append({
            "forecast_time": future_time,
            "rainfall_24h_mm": rainfall_24,
            "rainfall_72h_mm": rainfall_72,
            "id_ratio": id_ratio,
            "api_value": api_value,
            "trigger_score": trigger_score,
            "state": state,
        })

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GIRI-RAKSHAK — FORECAST RAINFALL → RISK FUSION")
    print("=" * 70)

    print("\nLocation:", LOCATION_NAME)
    print("Latitude :", LAT)
    print("Longitude:", LON)

    # ========================================================
    # STEP 1 — RF SUSCEPTIBILITY
    # ========================================================

    print("\n" + "-" * 70)
    print("1. REAL RF SUSCEPTIBILITY")
    print("-" * 70)

    prediction = predict_at_coordinate(
        LAT,
        LON
    )

    if prediction is None:
        print(
            "\nRF prediction unavailable because "
            "one or more real predictors are missing."
        )
        return

    susceptibility = float(
        prediction["susceptibility"]
    )

    print(
        f"\nReal RF Susceptibility = "
        f"{susceptibility:.4f}"
    )

    print(
        f"Susceptibility Level = "
        f"{prediction['level']}"
    )

    # ========================================================
    # STEP 2 — WEATHER
    # ========================================================

    print("\n" + "-" * 70)
    print("2. REAL WEATHER / RAINFALL FORECAST")
    print("-" * 70)

    rainfall_df = fetch_weather_rainfall(
        LAT,
        LON
    )

    current_time, rain24, rain72 = (
        calculate_forecast_windows(
            rainfall_df
        )
    )

    print(
        f"\nReference time: {current_time}"
    )

    print(
        f"Forecast rainfall next 24h: "
        f"{rain24:.2f} mm"
    )

    print(
        f"Forecast rainfall next 72h: "
        f"{rain72:.2f} mm"
    )

    # ========================================================
    # STEP 3 — EXISTING DYNAMIC TRIGGER
    # ========================================================

    print("\n" + "-" * 70)
    print("3. EXISTING DYNAMIC TRIGGER")
    print("-" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We use the existing trigger functions.
    # No second rainfall algorithm is created.
    #
    # For the pilot configuration, use East Khasi Hills
    # thresholds only when demonstrating that pilot.
    # --------------------------------------------------------

    cfg = EAST_KHASI_HILLS

    print(
        "\nTrigger configuration:",
        cfg.district_name
    )

    print(
        "24h threshold:",
        cfg.threshold_24h_mm,
        "mm"
    )

    print(
        "72h threshold:",
        cfg.threshold_72h_mm,
        "mm"
    )

    forecast_results = calculate_forecast_trigger(
        rainfall_df,
        current_time,
        cfg
    )

    # ========================================================
    # STEP 4 — DISPLAY FORECAST TRIGGER
    # ========================================================

    print("\nForecast trigger results:")

    for result in forecast_results:

        print("\n" + "." * 60)

        print(
            "Forecast time:",
            result["forecast_time"]
        )

        print(
            f"24h rainfall = "
            f"{result['rainfall_24h_mm']:.2f} mm"
        )

        print(
            f"72h rainfall = "
            f"{result['rainfall_72h_mm']:.2f} mm"
        )

        print(
            f"ID ratio = "
            f"{result['id_ratio']:.3f}"
        )

        print(
            f"API = "
            f"{result['api_value']:.2f}"
        )

        print(
            f"Dynamic Trigger = "
            f"{result['trigger_score']:.4f}"
        )

        print(
            f"Dynamic State = "
            f"{result['state']}"
        )

        # ====================================================
        # STEP 5 — RISK FUSION
        # ====================================================

        fusion = fuse_risk(
            susceptibility_score=susceptibility,
            trigger_score=result["trigger_score"],
        )

        print("\nFINAL RISK")

        print(
            f"Static Susceptibility = "
            f"{susceptibility:.4f}"
        )

        print(
            f"Forecast Dynamic Trigger = "
            f"{result['trigger_score']:.4f}"
        )

        print(
            f"Final Risk Score = "
            f"{fusion.risk_score:.4f}"
        )

        print(
            f"Risk Level = "
            f"{fusion.risk_level}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("FORECAST RISK SUMMARY")
    print("=" * 70)

    print(
        f"\nLocation: {LOCATION_NAME}"
    )

    print(
        f"RF Susceptibility: "
        f"{susceptibility:.4f}"
    )

    print(
        f"Next 24h rainfall: "
        f"{rain24:.2f} mm"
    )

    print(
        f"Next 72h rainfall: "
        f"{rain72:.2f} mm"
    )

    print("\nThe system now combines:")

    print(
        "  LAT/LON"
        " → 8 REAL PREDICTORS"
        " → TRAINED RF"
    )

    print(
        "  LAT/LON"
        " → REAL WEATHER FORECAST"
        " → EXISTING DYNAMIC TRIGGER"
    )

    print(
        "  SUSCEPTIBILITY + FORECAST TRIGGER"
        " → FINAL RISK"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
    