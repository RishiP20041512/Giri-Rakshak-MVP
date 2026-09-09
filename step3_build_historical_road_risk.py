from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

INPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_with_historical_landslides.gpkg"
)

INPUT_LAYER = "pilot_roads_historical"

OUTPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_historical_risk.gpkg"
)

OUTPUT_CSV = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_historical_risk.csv"
)


def main():

    print("=" * 70)
    print("STEP 3.4 — BUILD HISTORICAL ROAD-RISK LAYER")
    print("=" * 70)

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_GPKG}"
        )

    print("\n[1/5] Reading road dataset...")

    roads = gpd.read_file(
        INPUT_GPKG,
        layer=INPUT_LAYER
    )

    print(f"Roads loaded: {len(roads):,}")

    # --------------------------------------------------------
    # HISTORICAL LANDSLIDE COUNT
    # --------------------------------------------------------

    print("\n[2/5] Creating historical indicators...")

    roads["historical_landslide_count"] = (
        pd.to_numeric(
            roads["historical_landslide_count"],
            errors="coerce"
        ).fillna(0)
    )

    roads["nearest_landslide_distance_m"] = (
        pd.to_numeric(
            roads["nearest_landslide_distance_m"],
            errors="coerce"
        )
    )

    # --------------------------------------------------------
    # HISTORICAL FLAG
    # --------------------------------------------------------

    roads["historical_landslide_flag"] = (
        roads["historical_landslide_count"] > 0
    )

    # --------------------------------------------------------
    # HISTORICAL RISK INDICATOR
    # --------------------------------------------------------
    #
    # This is deliberately NOT merged into the RF score.
    #
    # 0 = no matched historical event
    # 1 = at least one matched historical event
    #
    # It is an evidence flag, not a probability.
    # --------------------------------------------------------

    roads["historical_event_indicator"] = np.where(
        roads["historical_landslide_count"] > 0,
        1,
        0
    )

    # --------------------------------------------------------
    # HISTORICAL RISK LABEL
    # --------------------------------------------------------

    roads["historical_risk_status"] = np.where(
        roads["historical_landslide_count"] > 0,
        "HISTORICAL_EVENT",
        "NO_MATCHED_HISTORICAL_EVENT"
    )

    # --------------------------------------------------------
    # COMBINED INFORMATION LABEL
    # --------------------------------------------------------

    def combined_status(row):

        lsi = row["LSI_mean"]
        historical = row[
            "historical_landslide_count"
        ]

        if pd.isna(lsi):
            return "NO_DATA"

        if historical > 0 and lsi >= 0.60:
            return "HISTORICAL_EVENT_HIGH_LSI"

        if historical > 0:
            return "HISTORICAL_EVENT"

        if lsi >= 0.80:
            return "VERY_HIGH_LSI"

        if lsi >= 0.60:
            return "HIGH_LSI"

        if lsi >= 0.40:
            return "MODERATE_LSI"

        if lsi >= 0.20:
            return "LOW_LSI"

        return "VERY_LOW_LSI"

    roads["combined_historical_status"] = roads.apply(
        combined_status,
        axis=1
    )

    # --------------------------------------------------------
    # QUALITY CHECK
    # --------------------------------------------------------

    print("\n[3/5] Quality check...")

    historical_roads = (
        roads["historical_landslide_flag"]
        == True
    )

    print(
        f"Roads with historical event: "
        f"{historical_roads.sum():,}"
    )

    print(
        f"Roads without matched event: "
        f"{(~historical_roads).sum():,}"
    )

    print("\nHistorical event roads:")

    if historical_roads.any():

        print(
            roads.loc[
                historical_roads,
                [
                    "road_id",
                    "district",
                    "road_name",
                    "road_type",
                    "LSI_mean",
                    "LSI_max",
                    "historical_landslide_count",
                    "nearest_landslide_distance_m",
                    "historical_risk_status",
                    "combined_historical_status",
                ],
            ].to_string(index=False)
        )

    # --------------------------------------------------------
    # SUMMARY BY DISTRICT
    # --------------------------------------------------------

    print("\nHistorical-event count by district:")

    district_summary = (
        roads.groupby("district")
        .agg(
            road_segments=("road_id", "count"),
            historical_event_roads=(
                "historical_landslide_flag",
                "sum"
            ),
            total_historical_matches=(
                "historical_landslide_count",
                "sum"
            ),
        )
        .reset_index()
    )

    print(
        district_summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\n[4/5] Saving GeoPackage...")

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    roads.to_file(
        OUTPUT_GPKG,
        layer="pilot_roads_historical_risk",
        driver="GPKG"
    )

    print("\n[5/5] Saving CSV...")

    roads.drop(
        columns="geometry"
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 70)
    print("STEP 3.4 COMPLETE!")
    print("=" * 70)

    print("\nOutput GeoPackage:")
    print(OUTPUT_GPKG)

    print("\nOutput CSV:")
    print(OUTPUT_CSV)

    print("\nNew fields:")
    print("  historical_event_indicator")
    print("  historical_risk_status")
    print("  combined_historical_status")

    print("\nIMPORTANT:")
    print(
        "Historical events are past evidence."
    )
    print(
        "They do NOT mean the road is currently blocked."
    )
    print(
        "Current blockage will be handled by the ACTIVE INCIDENT layer."
    )


if __name__ == "__main__":
    main()