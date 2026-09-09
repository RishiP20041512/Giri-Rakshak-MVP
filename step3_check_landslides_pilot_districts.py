from pathlib import Path

import geopandas as gpd
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

LANDSLIDE_CSV = (
    ROOT
    / "pilot_route_data"
    / "historical_landslide_points.csv"
)

ROAD_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_road_network.gpkg"
)

OUTPUT_DIR = ROOT / "pilot_route_data"

OUTPUT_ALL = (
    OUTPUT_DIR
    / "historical_landslides_district_check.csv"
)

OUTPUT_PILOT = (
    OUTPUT_DIR
    / "pilot_historical_landslide_points.csv"
)


# ============================================================
# YOUR 10 PILOT DISTRICTS
# ============================================================

PILOT_DISTRICTS = {
    "Meghalaya": [
        "East Khasi Hills",
        "West Khasi Hills",
        "Ri-Bhoi",
        "East Jaintia Hills",
        "West Jaintia Hills",
    ],
    "Assam": [
        "Kamrup Metropolitan",
        "Kamrup",
        "Dima Hasao",
        "Karbi Anglong",
        "West Karbi Anglong",
    ],
}


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_name(value):

    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Normalize common dash differences
    value = value.replace("–", "-")
    value = value.replace("—", "-")

    return value.lower()


PILOT_LOOKUP = {}

for state, districts in PILOT_DISTRICTS.items():

    for district in districts:

        PILOT_LOOKUP[
            normalize_name(district)
        ] = {
            "state": state,
            "district": district,
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 3.2 — CHECK HISTORICAL LANDSLIDES AGAINST PILOT DISTRICTS")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not LANDSLIDE_CSV.exists():

        raise FileNotFoundError(
            f"Landslide coordinate file not found:\n"
            f"{LANDSLIDE_CSV}"
        )

    if not ROAD_GPKG.exists():

        raise FileNotFoundError(
            f"Road GeoPackage not found:\n"
            f"{ROAD_GPKG}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # READ LANDSLIDE POINTS
    # --------------------------------------------------------

    print("\n[1/5] Reading real landslide points...")

    landslides = pd.read_csv(
        LANDSLIDE_CSV
    )

    print(
        f"Landslide points: {len(landslides):,}"
    )

    # --------------------------------------------------------
    # CREATE POINT GEOMETRIES
    # --------------------------------------------------------

    print("\n[2/5] Creating spatial points...")

    landslide_gdf = gpd.GeoDataFrame(
        landslides.copy(),
        geometry=gpd.points_from_xy(
            landslides["longitude"],
            landslides["latitude"]
        ),
        crs="EPSG:4326"
    )

    # --------------------------------------------------------
    # READ PILOT DISTRICTS
    # --------------------------------------------------------

    print("\n[3/5] Reading pilot district boundaries...")

    districts = gpd.read_file(
        ROAD_GPKG,
        layer="pilot_districts"
    )

    print(
        f"District polygons loaded: "
        f"{len(districts):,}"
    )

    print("\nDistrict layer columns:")

    for column in districts.columns:

        print(f"  {column}")

    # --------------------------------------------------------
    # FIND DISTRICT NAME COLUMN
    # --------------------------------------------------------

    possible_district_columns = [
        "district",
        "District",
        "name",
        "NAME",
        "name_2",
        "NAME_2",
        "admin_name",
        "official_name",
    ]

    district_column = None

    for column in possible_district_columns:

        if column in districts.columns:

            district_column = column
            break

    if district_column is None:

        raise ValueError(
            "\nCould not find the district-name column.\n"
            "Columns available:\n"
            + "\n".join(
                str(c)
                for c in districts.columns
            )
        )

    print(
        f"\nUsing district column: "
        f"{district_column}"
    )

    # --------------------------------------------------------
    # REPROJECT
    # --------------------------------------------------------

    if districts.crs != landslide_gdf.crs:

        landslide_gdf = landslide_gdf.to_crs(
            districts.crs
        )

    # --------------------------------------------------------
    # SPATIAL JOIN
    # --------------------------------------------------------

    print("\n[4/5] Spatially matching landslides to districts...")

    matched = gpd.sjoin(
        landslide_gdf,
        districts[
            [district_column, "geometry"]
        ],
        how="left",
        predicate="within"
    )

    # Rename matched district
    matched["matched_district"] = matched[
        district_column
    ]

    # --------------------------------------------------------
    # DETERMINE PILOT STATUS
    # --------------------------------------------------------

    pilot_states = []
    pilot_district_names = []
    pilot_status = []

    for value in matched["matched_district"]:

        normalized = normalize_name(value)

        if normalized in PILOT_LOOKUP:

            info = PILOT_LOOKUP[normalized]

            pilot_states.append(
                info["state"]
            )

            pilot_district_names.append(
                info["district"]
            )

            pilot_status.append(
                "INSIDE_PILOT"
            )

        else:

            pilot_states.append(
                None
            )

            pilot_district_names.append(
                None
            )

            pilot_status.append(
                "OUTSIDE_PILOT"
            )

    matched["pilot_state"] = pilot_states

    matched["pilot_district"] = (
        pilot_district_names
    )

    matched["pilot_status"] = (
        pilot_status
    )

    # --------------------------------------------------------
    # SAVE ALL CHECK RESULTS
    # --------------------------------------------------------

    output_columns = [
        "landslide_id",
        "source_record",
        "coordinate_number",
        "latitude",
        "longitude",
        "title",
        "matched_district",
        "pilot_state",
        "pilot_district",
        "pilot_status",
    ]

    # Keep only columns that exist
    output_columns = [
        c for c in output_columns
        if c in matched.columns
    ]

    result = matched[
        output_columns
    ].copy()

    result.to_csv(
        OUTPUT_ALL,
        index=False
    )

    # --------------------------------------------------------
    # KEEP PILOT ONLY
    # --------------------------------------------------------

    pilot_points = result[
        result["pilot_status"]
        == "INSIDE_PILOT"
    ].copy()

    pilot_points.to_csv(
        OUTPUT_PILOT,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n[5/5] RESULTS")
    print("=" * 70)

    total = len(result)

    inside = (
        result["pilot_status"]
        == "INSIDE_PILOT"
    ).sum()

    outside = (
        result["pilot_status"]
        == "OUTSIDE_PILOT"
    ).sum()

    print(
        f"Total real landslide points: "
        f"{total:,}"
    )

    print(
        f"Inside 10 pilot districts: "
        f"{inside:,}"
    )

    print(
        f"Outside pilot districts: "
        f"{outside:,}"
    )

    # --------------------------------------------------------
    # SHOW PILOT POINTS
    # --------------------------------------------------------

    if inside > 0:

        print("\nLandslides inside pilot districts:")

        print(
            pilot_points[
                [
                    "landslide_id",
                    "latitude",
                    "longitude",
                    "pilot_state",
                    "pilot_district",
                    "title",
                ]
            ].to_string(
                index=False
            )
        )

        print("\nCount by pilot district:")

        print(
            pilot_points[
                "pilot_district"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )

    else:

        print(
            "\nNo historical landslide points "
            "fall inside the 10 pilot districts."
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3.2 COMPLETE!")
    print("=" * 70)

    print("\nAll-point audit:")
    print(OUTPUT_ALL)

    print("\nPilot-only landslides:")
    print(OUTPUT_PILOT)

    print(
        "\nIMPORTANT: "
        "Only actual coordinates from the source CSV "
        "were used."
    )


if __name__ == "__main__":
    main()