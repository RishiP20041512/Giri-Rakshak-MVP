from pathlib import Path
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

from dynamic.config import GENERIC_NER_DISTRICT
from dynamic.rainfall_trigger import (
    antecedent_wetness_index,
    simplified_two_window_trigger,
    combined_trigger_score,
    alert_state,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_with_cost.gpkg"
)

INPUT_LAYER = "pilot_roads_cost"

OUTPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_dynamic_cost.gpkg"
)

OUTPUT_CSV = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_dynamic_cost.csv"
)


# ============================================================
# OPEN-METEO
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ============================================================
# PILOT LOCATION
# ============================================================
#
# This is a district-level dynamic rainfall value.
#
# For the first routing demonstration, we use the pilot
# region represented by the existing dynamic configuration.
#
# IMPORTANT:
# GENERIC_NER_DISTRICT is explicitly an uncalibrated
# prototype configuration.
#
# Do NOT describe these thresholds as NER-wide validated.
# ============================================================

LATITUDE = GENERIC_NER_DISTRICT.station_lat
LONGITUDE = GENERIC_NER_DISTRICT.station_lon

CFG = GENERIC_NER_DISTRICT


# ============================================================
# FETCH REAL RAINFALL
# ============================================================

def fetch_open_meteo_rainfall():

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,

        "hourly": (
            "precipitation"
        ),

        "daily": (
            "precipitation_sum"
        ),

        "timezone": "auto",

        "past_days": 15,
        "forecast_days": 7,
    }

    print("\nFetching real Open-Meteo rainfall...")

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    hourly = pd.DataFrame({
        "time": pd.to_datetime(
            data["hourly"]["time"]
        ),
        "precipitation_mm": (
            data["hourly"]["precipitation"]
        ),
    })

    hourly = hourly.sort_values(
        "time"
    ).reset_index(drop=True)

    return hourly


# ============================================================
# CALCULATE FORECAST WINDOWS
# ============================================================

def calculate_rainfall_windows(hourly):

    now = hourly["time"].max()

    # The last available timestamp is used as
    # the reference point.
    #
    # We calculate forward rainfall windows from
    # the latest available point.

    future = hourly[
        hourly["time"] > now - pd.Timedelta(hours=72)
    ].copy()

    cum_24h = (
        future[
            future["time"] <= now + pd.Timedelta(hours=24)
        ]["precipitation_mm"]
        .sum()
    )

    cum_72h = (
        future[
            future["time"] <= now + pd.Timedelta(hours=72)
        ]["precipitation_mm"]
        .sum()
    )

    return float(cum_24h), float(cum_72h)


# ============================================================
# CALCULATE DYNAMIC TRIGGER
# ============================================================

def calculate_dynamic_trigger(
    hourly,
    cfg,
):

    # --------------------------------------------------------
    # Use daily rainfall for antecedent wetness.
    # --------------------------------------------------------

    daily = (
        hourly
        .set_index("time")
        ["precipitation_mm"]
        .resample("1D")
        .sum()
    )

    daily = daily.tail(
        cfg.awi_window_days
    )

    awi_series = antecedent_wetness_index(
        daily,
        cfg,
    )

    api_value = float(
        awi_series.iloc[-1]
    )

    # --------------------------------------------------------
    # Real rainfall windows
    # --------------------------------------------------------

    latest_time = hourly["time"].max()

    last_24h = hourly[
        (
            hourly["time"]
            > latest_time - pd.Timedelta(hours=24)
        )
        &
        (
            hourly["time"]
            <= latest_time
        )
    ]

    last_72h = hourly[
        (
            hourly["time"]
            > latest_time - pd.Timedelta(hours=72)
        )
        &
        (
            hourly["time"]
            <= latest_time
        )
    ]

    cum_24h = float(
        last_24h["precipitation_mm"].sum()
    )

    cum_72h = float(
        last_72h["precipitation_mm"].sum()
    )

    id_ratio = simplified_two_window_trigger(
        cum_24h,
        cum_72h,
        cfg,
    )

    trigger = combined_trigger_score(
        id_ratio,
        api_value,
        cfg,
    )

    state = alert_state(
        trigger,
        cfg,
    )

    return {
        "timestamp": latest_time,
        "cum_24h_mm": cum_24h,
        "cum_72h_mm": cum_72h,
        "api_value": api_value,
        "id_exceedance_ratio": id_ratio,
        "dynamic_trigger": trigger,
        "dynamic_state": state,
    }


# ============================================================
# APPLY DYNAMIC PENALTY
# ============================================================

def dynamic_multiplier(state):

    mapping = {
        "normal": 1.00,
        "watch": 1.30,
        "warning": 2.00,
    }

    return mapping.get(
        str(state).lower(),
        1.00,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 5.2 — REAL DYNAMIC ROAD RISK")
    print("=" * 70)

    # --------------------------------------------------------
    # READ STEP 5.1
    # --------------------------------------------------------

    print("\n[1/6] Reading Step 5.1 road-cost layer...")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(
            f"Missing:\n{INPUT_GPKG}"
        )

    roads = gpd.read_file(
        INPUT_GPKG,
        layer=INPUT_LAYER,
    )

    print(
        f"Roads loaded: {len(roads):,}"
    )

    # --------------------------------------------------------
    # REAL WEATHER
    # --------------------------------------------------------

    print("\n[2/6] Downloading real rainfall...")

    hourly = fetch_open_meteo_rainfall()

    # --------------------------------------------------------
    # DYNAMIC TRIGGER
    # --------------------------------------------------------

    print("\n[3/6] Calculating dynamic rainfall trigger...")

    dynamic = calculate_dynamic_trigger(
        hourly,
        CFG,
    )

    print("\nREAL DYNAMIC CONDITIONS")
    print("-" * 40)

    print(
        f"Reference time       : "
        f"{dynamic['timestamp']}"
    )

    print(
        f"24h rainfall         : "
        f"{dynamic['cum_24h_mm']:.2f} mm"
    )

    print(
        f"72h rainfall         : "
        f"{dynamic['cum_72h_mm']:.2f} mm"
    )

    print(
        f"Antecedent wetness   : "
        f"{dynamic['api_value']:.2f}"
    )

    print(
        f"ID exceedance ratio  : "
        f"{dynamic['id_exceedance_ratio']:.3f}"
    )

    print(
        f"Dynamic trigger      : "
        f"{dynamic['dynamic_trigger']:.4f}"
    )

    print(
        f"Dynamic state        : "
        f"{dynamic['dynamic_state']}"
    )

    # --------------------------------------------------------
    # APPLY TO ROADS
    # --------------------------------------------------------

    print("\n[4/6] Applying dynamic trigger to roads...")

    roads["dynamic_trigger"] = (
        dynamic["dynamic_trigger"]
    )

    roads["dynamic_state"] = (
        dynamic["dynamic_state"]
    )

    roads["dynamic_penalty_multiplier"] = (
        dynamic_multiplier(
            dynamic["dynamic_state"]
        )
    )

    # --------------------------------------------------------
    # RE-CALCULATE COST
    # --------------------------------------------------------

    print("\n[5/6] Recalculating routing cost...")

    roads["routing_cost_min"] = (
        roads["base_travel_time_min"]
        *
        roads["risk_multiplier"]
        *
        roads["dynamic_penalty_multiplier"]
    )

    # Historical factor was already included in
    # risk_multiplier in Step 5.1.

    # --------------------------------------------------------
    # BLOCKED ROADS
    # --------------------------------------------------------

    roads.loc[
        roads["active_blocked"] == True,
        "routing_cost_min"
    ] = np.inf

    # --------------------------------------------------------
    # UPDATE STATUS
    # --------------------------------------------------------

    def update_status(row):

        if row["active_blocked"]:
            return "BLOCKED"

        if row["dynamic_state"] == "warning":
            return "AVOID_IF_ALTERNATIVE"

        if row["dynamic_state"] == "watch":
            return "CAUTION"

        if row["LSI_risk_class"] in [
            "HIGH",
            "VERY HIGH",
        ]:
            return "AVOID_IF_ALTERNATIVE"

        if row["LSI_risk_class"] == "MODERATE":
            return "CAUTION"

        return "AVAILABLE"

    roads["routing_status"] = roads.apply(
        update_status,
        axis=1,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    roads.to_file(
        OUTPUT_GPKG,
        layer="pilot_roads_dynamic_cost",
        driver="GPKG",
    )

    roads.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n[6/6] Saving complete.")

    print("\n" + "=" * 70)
    print("STEP 5.2 COMPLETE!")
    print("=" * 70)

    print(
        f"\nDynamic trigger: "
        f"{dynamic['dynamic_trigger']:.4f}"
    )

    print(
        f"Dynamic state: "
        f"{dynamic['dynamic_state']}"
    )

    print("\nRouting status:")

    print(
        roads["routing_status"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nBlocked roads: "
        f"{roads['active_blocked'].sum():,}"
    )

    print("\nOutput GeoPackage:")
    print(OUTPUT_GPKG)

    print("\nOutput CSV:")
    print(OUTPUT_CSV)

    print("\n" + "=" * 70)
    print("READY FOR STEP 6 — NETWORKX ROUTING")
    print("=" * 70)


if __name__ == "__main__":
    main()
    