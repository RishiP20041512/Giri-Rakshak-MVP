from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_active_incidents.gpkg"
)

INPUT_LAYER = "pilot_roads_active"

OUTPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_with_cost.gpkg"
)

OUTPUT_CSV = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_with_cost.csv"
)


# ============================================================
# ROUTING SETTINGS
# ============================================================

# Static susceptibility contribution
STATIC_WEIGHT = 0.60

# Dynamic trigger contribution
DYNAMIC_WEIGHT = 0.40

# Historical-event multiplier
HISTORICAL_EVENT_PENALTY = 1.20

# LSI-based route penalty
#
# LOW:
#   no extra penalty
#
# MODERATE:
#   1.30
#
# HIGH:
#   2.50
#
# VERY HIGH:
#   2.50
#
LSI_PENALTY = {
    "VERY LOW": 1.00,
    "LOW": 1.00,
    "MODERATE": 1.30,
    "HIGH": 2.50,
    "VERY HIGH": 2.50,
}


# Dynamic trigger penalty
#
# These are operational routing multipliers,
# not probabilities.
#
DYNAMIC_PENALTY = {
    "LOW": 1.00,
    "MEDIUM": 1.30,
    "HIGH": 2.00,
    "CRITICAL": 2.50,
}


# ============================================================
# BASE SPEED ESTIMATION
# ============================================================

# We use OSM maxspeed when available.
#
# When maxspeed is absent, use documented road-class
# defaults only for the routing prototype.
#
# These are not claimed to be observed speeds.

ROAD_CLASS_DEFAULT_SPEED_KMPH = {
    "motorway": 80,
    "motorway_link": 50,

    "trunk": 70,
    "trunk_link": 45,

    "primary": 60,
    "primary_link": 40,

    "secondary": 50,
    "secondary_link": 35,

    "tertiary": 40,
    "tertiary_link": 30,

    "unclassified": 30,
    "residential": 25,
    "living_street": 15,

    "service": 15,
    "track": 15,
    "road": 25,
}


def parse_speed(value):

    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text == "":
        return np.nan

    # Remove common text
    text = (
        text
        .replace("km/h", "")
        .replace("kph", "")
        .replace("mph", "")
    )

    # Handle values such as:
    # "30"
    # "30;40"
    # "50 mph"
    #
    first_value = text.split(";")[0].strip()

    try:
        speed = float(first_value)
    except ValueError:
        return np.nan

    # If original value contained mph,
    # convert to km/h.
    if "mph" in str(value).lower():
        speed = speed * 1.609344

    if speed <= 0:
        return np.nan

    return speed


# ============================================================
# RISK CLASS FROM LSI
# ============================================================

def classify_lsi(lsi):

    if pd.isna(lsi):
        return "NO_DATA"

    if lsi < 0.20:
        return "VERY LOW"

    elif lsi < 0.40:
        return "LOW"

    elif lsi < 0.60:
        return "MODERATE"

    elif lsi < 0.80:
        return "HIGH"

    else:
        return "VERY HIGH"


# ============================================================
# DYNAMIC TRIGGER CLASS
# ============================================================

def classify_dynamic_trigger(trigger):

    if trigger is None or pd.isna(trigger):
        return "LOW"

    trigger = float(trigger)

    if trigger < 0.25:
        return "LOW"

    elif trigger < 0.50:
        return "MEDIUM"

    elif trigger < 0.75:
        return "HIGH"

    else:
        return "CRITICAL"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 5.1 — CALCULATE ROAD ROUTING COST")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_GPKG.exists():

        raise FileNotFoundError(
            f"Input road dataset not found:\n"
            f"{INPUT_GPKG}"
        )

    # --------------------------------------------------------
    # READ ROADS
    # --------------------------------------------------------

    print("\n[1/7] Reading road dataset...")

    roads = gpd.read_file(
        INPUT_GPKG,
        layer=INPUT_LAYER
    )

    print(
        f"Roads loaded: {len(roads):,}"
    )

    # --------------------------------------------------------
    # BASE SPEED
    # --------------------------------------------------------

    print("\n[2/7] Calculating base travel speed...")

    roads["parsed_speed_kmph"] = roads[
        "maxspeed"
    ].apply(parse_speed)

    roads["speed_source"] = np.where(
        roads["parsed_speed_kmph"].notna(),
        "OSM_MAXSPEED",
        "ROAD_CLASS_DEFAULT"
    )

    roads["base_speed_kmph"] = (
        roads["parsed_speed_kmph"]
    )

    missing_speed = (
        roads["base_speed_kmph"].isna()
    )

    roads.loc[
        missing_speed,
        "base_speed_kmph"
    ] = roads.loc[
        missing_speed,
        "road_type"
    ].map(
        ROAD_CLASS_DEFAULT_SPEED_KMPH
    )

    # Final fallback only if an unknown road type exists.
    roads["base_speed_kmph"] = (
        roads["base_speed_kmph"]
        .fillna(25.0)
    )

    # --------------------------------------------------------
    # BASE TRAVEL TIME
    # --------------------------------------------------------

    roads["base_travel_time_min"] = (
        roads["length_km"]
        / roads["base_speed_kmph"]
        * 60.0
    )

    # --------------------------------------------------------
    # STATIC LSI
    # --------------------------------------------------------

    print("\n[3/7] Calculating static susceptibility penalty...")

    roads["LSI_mean"] = pd.to_numeric(
        roads["LSI_mean"],
        errors="coerce"
    )

    roads["LSI_risk_class"] = roads[
        "LSI_mean"
    ].apply(
        classify_lsi
    )

    roads["lsi_penalty_multiplier"] = (
        roads["LSI_risk_class"]
        .map(LSI_PENALTY)
        .fillna(1.0)
    )

    # --------------------------------------------------------
    # DYNAMIC TRIGGER
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # No current rainfall is invented here.
    #
    # Since no live trigger has been attached to every
    # road yet, the default dynamic trigger is 0.
    #
    # Later Step 5.2 will replace this with the real
    # current/forecast trigger.
    # --------------------------------------------------------

    print("\n[4/7] Creating dynamic-trigger fields...")

    roads["dynamic_trigger"] = 0.0

    roads["dynamic_state"] = "LOW"

    roads["dynamic_penalty_multiplier"] = 1.0

    # --------------------------------------------------------
    # HISTORICAL EVENT
    # --------------------------------------------------------

    print("\n[5/7] Applying historical-event factor...")

    roads["historical_landslide_count"] = (
        pd.to_numeric(
            roads["historical_landslide_count"],
            errors="coerce"
        )
        .fillna(0)
    )

    roads["historical_event_multiplier"] = np.where(
        roads["historical_landslide_count"] > 0,
        HISTORICAL_EVENT_PENALTY,
        1.0
    )

    # --------------------------------------------------------
    # ACTIVE BLOCKAGE
    # --------------------------------------------------------

    roads["active_blocked"] = (
        roads["active_blocked"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # CALCULATE ROUTING COST
    # --------------------------------------------------------

    print("\n[6/7] Calculating final routing cost...")

    roads["risk_multiplier"] = (
        STATIC_WEIGHT
        * roads["lsi_penalty_multiplier"]
        +
        DYNAMIC_WEIGHT
        * roads["dynamic_penalty_multiplier"]
    )

    roads["routing_cost_min"] = (
        roads["base_travel_time_min"]
        * roads["risk_multiplier"]
        * roads["historical_event_multiplier"]
    )

    # --------------------------------------------------------
    # BLOCKED ROAD = INFINITY
    # --------------------------------------------------------

    roads.loc[
        roads["active_blocked"],
        "routing_cost_min"
    ] = np.inf

    # --------------------------------------------------------
    # ROUTING STATUS
    # --------------------------------------------------------

    def routing_status(row):

        if row["active_blocked"]:
            return "BLOCKED"

        if row["LSI_risk_class"] in [
            "HIGH",
            "VERY HIGH"
        ]:
            return "AVOID_IF_ALTERNATIVE"

        if row["LSI_risk_class"] == "MODERATE":
            return "CAUTION"

        return "AVAILABLE"

    roads["routing_status"] = roads.apply(
        routing_status,
        axis=1
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\n[7/7] Saving outputs...")

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    roads.to_file(
        OUTPUT_GPKG,
        layer="pilot_roads_cost",
        driver="GPKG"
    )

    roads.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"\nTotal roads: "
        f"{len(roads):,}"
    )

    print("\nLSI risk classes:")

    print(
        roads["LSI_risk_class"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nRouting status:")

    print(
        roads["routing_status"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nSpeed source:")

    print(
        roads["speed_source"]
        .value_counts()
        .to_string()
    )

    print(
        "\nBlocked roads: "
        f"{roads['active_blocked'].sum():,}"
    )

    print(
        "\nHistorical-event roads: "
        f"{(roads['historical_landslide_count'] > 0).sum():,}"
    )

    # --------------------------------------------------------
    # SHOW HISTORICAL ROAD
    # --------------------------------------------------------

    historical = roads[
        roads["historical_landslide_count"] > 0
    ]

    if len(historical) > 0:

        print("\nHistorical-event road example:")

        print(
            historical[
                [
                    "road_id",
                    "district",
                    "road_type",
                    "LSI_mean",
                    "LSI_risk_class",
                    "historical_landslide_count",
                    "historical_event_multiplier",
                    "base_travel_time_min",
                    "routing_cost_min",
                    "routing_status",
                ]
            ].to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("STEP 5.1 COMPLETE!")
    print("=" * 70)

    print("\nGeoPackage:")
    print(OUTPUT_GPKG)

    print("\nCSV:")
    print(OUTPUT_CSV)

    print("\nNew routing fields:")
    print("  base_speed_kmph")
    print("  speed_source")
    print("  base_travel_time_min")
    print("  LSI_risk_class")
    print("  lsi_penalty_multiplier")
    print("  dynamic_trigger")
    print("  dynamic_state")
    print("  dynamic_penalty_multiplier")
    print("  historical_event_multiplier")
    print("  risk_multiplier")
    print("  routing_cost_min")
    print("  routing_status")

    print("\nIMPORTANT:")
    print(
        "Dynamic trigger is currently 0 because "
        "no current/forecast value is being invented."
    )

    print(
        "Step 5.2 will attach real dynamic rainfall "
        "information to the routing layer."
    )


if __name__ == "__main__":
    main()