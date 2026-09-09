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
    / "pilot_roads_with_susceptibility.gpkg"
)

LANDSLIDE_CSV = (
    ROOT
    / "pilot_route_data"
    / "pilot_historical_landslide_points.csv"
)

OUTPUT_DIR = ROOT / "pilot_route_data"

OUTPUT_GPKG = (
    OUTPUT_DIR
    / "pilot_roads_with_historical_landslides.gpkg"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "pilot_roads_with_historical_landslides.csv"
)

MATCH_OUTPUT_CSV = (
    OUTPUT_DIR
    / "landslide_road_matches.csv"
)


# ============================================================
# SETTINGS
# ============================================================

ROAD_LAYER = "pilot_roads_risk"

# Projected CRS already used by your RF raster.
# Suitable for distance measurements.
METRIC_CRS = "EPSG:6933"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 3.3 — MATCH HISTORICAL LANDSLIDES TO REAL OSM ROADS")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not ROAD_GPKG.exists():
        raise FileNotFoundError(
            f"\nRoad susceptibility file not found:\n"
            f"{ROAD_GPKG}"
        )

    if not LANDSLIDE_CSV.exists():
        raise FileNotFoundError(
            f"\nPilot landslide file not found:\n"
            f"{LANDSLIDE_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # READ ROADS
    # --------------------------------------------------------

    print("\n[1/6] Reading road network...")

    roads = gpd.read_file(
        ROAD_GPKG,
        layer=ROAD_LAYER
    )

    print(
        f"Roads loaded: {len(roads):,}"
    )

    print(
        f"Original CRS: {roads.crs}"
    )

    # --------------------------------------------------------
    # READ LANDSLIDES
    # --------------------------------------------------------

    print("\n[2/6] Reading pilot historical landslides...")

    landslides = pd.read_csv(
        LANDSLIDE_CSV
    )

    print(
        f"Pilot landslide points: "
        f"{len(landslides):,}"
    )

    if len(landslides) == 0:
        raise RuntimeError(
            "No historical landslide points found."
        )

    # --------------------------------------------------------
    # CREATE LANDSLIDE POINT GEOMETRIES
    # --------------------------------------------------------

    landslide_gdf = gpd.GeoDataFrame(
        landslides.copy(),
        geometry=gpd.points_from_xy(
            landslides["longitude"],
            landslides["latitude"]
        ),
        crs="EPSG:4326"
    )

    # --------------------------------------------------------
    # PROJECT TO METRIC CRS
    # --------------------------------------------------------

    print("\n[3/6] Preparing distance calculations...")

    roads_metric = roads.to_crs(
        METRIC_CRS
    )

    landslides_metric = landslide_gdf.to_crs(
        METRIC_CRS
    )

    print(
        f"Metric CRS: {METRIC_CRS}"
    )

    # --------------------------------------------------------
    # FIND NEAREST ROAD
    # --------------------------------------------------------

    print("\n[4/6] Finding nearest road for each landslide...")

    matches = []

    for _, landslide in landslides_metric.iterrows():

        point = landslide.geometry

        # Calculate distance from this landslide
        # to every road.
        distances = roads_metric.geometry.distance(
            point
        )

        nearest_index = distances.idxmin()

        nearest_distance = float(
            distances.loc[nearest_index]
        )

        nearest_road = roads_metric.loc[
            nearest_index
        ]

        matches.append(
            {
                "landslide_id": landslide[
                    "landslide_id"
                ],

                "latitude": landslide[
                    "latitude"
                ],

                "longitude": landslide[
                    "longitude"
                ],

                "title": landslide[
                    "title"
                ],

                "road_index": nearest_index,

                "road_id": nearest_road[
                    "road_id"
                ],

                "district": nearest_road[
                    "district"
                ],

                "road_name": nearest_road[
                    "road_name"
                ],

                "road_type": nearest_road[
                    "road_type"
                ],

                "distance_to_road_m": nearest_distance,

                "LSI_mean": nearest_road[
                    "LSI_mean"
                ],

                "LSI_max": nearest_road[
                    "LSI_max"
                ],

                "risk_class": nearest_road[
                    "risk_class"
                ],
            }
        )

    match_df = pd.DataFrame(
        matches
    )

    # --------------------------------------------------------
    # SAVE MATCH TABLE
    # --------------------------------------------------------

    match_df.to_csv(
        MATCH_OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # ATTACH TO ROADS
    # --------------------------------------------------------

    print("\n[5/6] Attaching historical landslide information to roads...")

    roads["historical_landslide_count"] = 0

    roads["nearest_landslide_distance_m"] = np.nan

    roads["historical_landslide_flag"] = False

    # --------------------------------------------------------
    # PROCESS EACH MATCH
    # --------------------------------------------------------

    for _, match in match_df.iterrows():

        road_index = match[
            "road_index"
        ]

        # Increase count
        roads.loc[
            road_index,
            "historical_landslide_count"
        ] += 1

        # Store nearest historical landslide distance
        current_distance = roads.loc[
            road_index,
            "nearest_landslide_distance_m"
        ]

        new_distance = match[
            "distance_to_road_m"
        ]

        if (
            pd.isna(current_distance)
            or new_distance < current_distance
        ):

            roads.loc[
                road_index,
                "nearest_landslide_distance_m"
            ] = new_distance

        # Mark that this road has a matched
        # historical landslide.
        roads.loc[
            road_index,
            "historical_landslide_flag"
        ] = True

    # --------------------------------------------------------
    # SAVE FINAL ROAD DATA
    # --------------------------------------------------------

    print("\nSaving final road dataset...")

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    roads.to_file(
        OUTPUT_GPKG,
        layer="pilot_roads_historical",
        driver="GPKG"
    )

    roads.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print("\n[6/6] RESULTS")
    print("=" * 70)

    matched_roads = (
        roads["historical_landslide_flag"]
        == True
    ).sum()

    total_historical = (
        roads["historical_landslide_count"]
        .sum()
    )

    print(
        f"Historical landslide points: "
        f"{len(landslides):,}"
    )

    print(
        f"Roads matched to historical landslides: "
        f"{matched_roads:,}"
    )

    print(
        f"Total historical landslide matches: "
        f"{int(total_historical):,}"
    )

    print("\nLANDSLIDE → ROAD MATCH:")

    display_columns = [
        "landslide_id",
        "latitude",
        "longitude",
        "district",
        "road_id",
        "road_name",
        "road_type",
        "distance_to_road_m",
        "LSI_mean",
        "risk_class",
    ]

    print(
        match_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\nDistance statistics:")

    print(
        match_df[
            "distance_to_road_m"
        ].describe()
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3.3 COMPLETE!")
    print("=" * 70)

    print("\nFinal road GeoPackage:")
    print(OUTPUT_GPKG)

    print("\nFinal road CSV:")
    print(OUTPUT_CSV)

    print("\nLandslide-road match table:")
    print(MATCH_OUTPUT_CSV)

    print("\nNew road fields:")
    print("  historical_landslide_count")
    print("  nearest_landslide_distance_m")
    print("  historical_landslide_flag")

    print(
        "\nIMPORTANT:"
    )
    print(
        "This step identifies the nearest real OSM road "
        "to each real historical landslide coordinate."
    )

    print(
        "It does NOT automatically declare the road blocked."
    )

    print(
        "Current historical data is evidence of past events, "
        "not a current active incident."
    )


if __name__ == "__main__":
    main()