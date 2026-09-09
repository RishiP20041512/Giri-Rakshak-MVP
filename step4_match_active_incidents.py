from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

ROAD_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_historical_risk.gpkg"
)

INCIDENT_CSV = (
    ROOT
    / "pilot_route_data"
    / "active_incidents.csv"
)

OUTPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_active_incidents.gpkg"
)

OUTPUT_CSV = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_active_incidents.csv"
)

MATCH_CSV = (
    ROOT
    / "pilot_route_data"
    / "active_incident_road_matches.csv"
)


# ============================================================
# SETTINGS
# ============================================================

ROAD_LAYER = "pilot_roads_historical_risk"

METRIC_CRS = "EPSG:6933"

# Maximum distance for associating an incident
# with a road.
MAX_MATCH_DISTANCE_M = 100.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 4.2 — MATCH ACTIVE INCIDENTS TO ROADS")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not ROAD_GPKG.exists():
        raise FileNotFoundError(
            f"Road dataset not found:\n{ROAD_GPKG}"
        )

    if not INCIDENT_CSV.exists():
        raise FileNotFoundError(
            f"Active incident file not found:\n{INCIDENT_CSV}"
        )

    # --------------------------------------------------------
    # READ ROADS
    # --------------------------------------------------------

    print("\n[1/6] Reading road network...")

    roads = gpd.read_file(
        ROAD_GPKG,
        layer=ROAD_LAYER
    )

    print(f"Roads loaded: {len(roads):,}")

    # --------------------------------------------------------
    # READ INCIDENTS
    # --------------------------------------------------------

    print("\n[2/6] Reading active incidents...")

    incidents = pd.read_csv(
        INCIDENT_CSV
    )

    print(
        f"Active incidents found: "
        f"{len(incidents):,}"
    )

    # --------------------------------------------------------
    # PREPARE ROAD FLAGS
    # --------------------------------------------------------

    roads["active_incident_count"] = 0

    roads["active_blocked"] = False

    roads["active_incident_severity"] = ""

    roads["active_incident_id"] = ""

    roads["active_incident_source"] = ""

    # --------------------------------------------------------
    # EMPTY INCIDENT FILE
    # --------------------------------------------------------

    if len(incidents) == 0:

        print("\nNo active incidents currently exist.")

        print(
            "\nTherefore no road will be marked BLOCKED."
        )

        print(
            "This is correct and intentional."
        )

        # Save unchanged road network with empty
        # active-incident fields.

        if OUTPUT_GPKG.exists():
            OUTPUT_GPKG.unlink()

        roads.to_file(
            OUTPUT_GPKG,
            layer="pilot_roads_active",
            driver="GPKG"
        )

        roads.drop(
            columns="geometry"
        ).to_csv(
            OUTPUT_CSV,
            index=False
        )

        pd.DataFrame(
            columns=[
                "incident_id",
                "road_id",
                "distance_to_road_m",
                "blocking",
                "severity",
                "verification_status",
                "source",
            ]
        ).to_csv(
            MATCH_CSV,
            index=False
        )

        print("\n" + "=" * 70)
        print("STEP 4.2 COMPLETE!")
        print("=" * 70)

        print("\nOutput:")
        print(OUTPUT_GPKG)

        print("\nBlocked roads: 0")

        return

    # --------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    required_columns = [
        "incident_id",
        "latitude",
        "longitude",
        "timestamp",
        "incident_type",
        "severity",
        "blocking",
        "verification_status",
        "source",
        "description",
    ]

    missing = [
        column
        for column in required_columns
        if column not in incidents.columns
    ]

    if missing:

        raise ValueError(
            "Missing incident columns:\n"
            + "\n".join(missing)
        )

    # --------------------------------------------------------
    # CREATE INCIDENT POINTS
    # --------------------------------------------------------

    print("\n[3/6] Creating incident points...")

    incidents["latitude"] = pd.to_numeric(
        incidents["latitude"],
        errors="coerce"
    )

    incidents["longitude"] = pd.to_numeric(
        incidents["longitude"],
        errors="coerce"
    )

    incidents = incidents.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    incident_points = gpd.GeoDataFrame(
        incidents.copy(),
        geometry=gpd.points_from_xy(
            incidents["longitude"],
            incidents["latitude"]
        ),
        crs="EPSG:4326"
    )

    # --------------------------------------------------------
    # PROJECT TO METRIC CRS
    # --------------------------------------------------------

    print("\n[4/6] Calculating nearest roads...")

    roads_metric = roads.to_crs(
        METRIC_CRS
    )

    incident_metric = incident_points.to_crs(
        METRIC_CRS
    )

    matches = []

    # --------------------------------------------------------
    # MATCH INCIDENTS
    # --------------------------------------------------------

    for _, incident in incident_metric.iterrows():

        distances = roads_metric.geometry.distance(
            incident.geometry
        )

        nearest_index = distances.idxmin()

        distance_m = float(
            distances.loc[nearest_index]
        )

        road = roads_metric.loc[
            nearest_index
        ]

        # Only associate the incident with a road
        # if it is reasonably close.

        if distance_m <= MAX_MATCH_DISTANCE_M:

            blocking_value = str(
                incident["blocking"]
            ).strip().lower()

            is_blocking = blocking_value in [
                "true",
                "1",
                "yes",
                "y",
                "blocked"
            ]

            matches.append(
                {
                    "incident_id":
                        incident["incident_id"],

                    "road_id":
                        road["road_id"],

                    "distance_to_road_m":
                        distance_m,

                    "blocking":
                        is_blocking,

                    "severity":
                        incident["severity"],

                    "verification_status":
                        incident[
                            "verification_status"
                        ],

                    "source":
                        incident["source"],
                }
            )

            # Attach incident information
            # to the road.

            roads.loc[
                nearest_index,
                "active_incident_count"
            ] += 1

            if is_blocking:

                roads.loc[
                    nearest_index,
                    "active_blocked"
                ] = True

            roads.loc[
                nearest_index,
                "active_incident_severity"
            ] = str(
                incident["severity"]
            )

            roads.loc[
                nearest_index,
                "active_incident_id"
            ] = str(
                incident["incident_id"]
            )

            roads.loc[
                nearest_index,
                "active_incident_source"
            ] = str(
                incident["source"]
            )

    # --------------------------------------------------------
    # SAVE MATCH TABLE
    # --------------------------------------------------------

    match_df = pd.DataFrame(
        matches
    )

    match_df.to_csv(
        MATCH_CSV,
        index=False
    )

    # --------------------------------------------------------
    # SAVE ROAD DATA
    # --------------------------------------------------------

    print("\n[5/6] Saving road incident layer...")

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    roads.to_file(
        OUTPUT_GPKG,
        layer="pilot_roads_active",
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

    print("\n[6/6] RESULTS")
    print("=" * 70)

    print(
        f"Active incidents: "
        f"{len(incidents):,}"
    )

    print(
        f"Incidents matched to roads: "
        f"{len(match_df):,}"
    )

    print(
        f"Currently blocked roads: "
        f"{roads['active_blocked'].sum():,}"
    )

    print(
        f"Roads with active incidents: "
        f"{(roads['active_incident_count'] > 0).sum():,}"
    )

    print("\n" + "=" * 70)
    print("STEP 4.2 COMPLETE!")
    print("=" * 70)

    print("\nOutput GeoPackage:")
    print(OUTPUT_GPKG)

    print("\nOutput CSV:")
    print(OUTPUT_CSV)

    print("\nIncident-road matches:")
    print(MATCH_CSV)

    print("\nIMPORTANT:")
    print(
        "Only active incidents within "
        f"{MAX_MATCH_DISTANCE_M} m of a road are matched."
    )

    print(
        "Only incidents marked blocking are "
        "used to mark a road as blocked."
    )

    print(
        "Historical incidents are NOT treated "
        "as active incidents."
    )


if __name__ == "__main__":
    main()
    