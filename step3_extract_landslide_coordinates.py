from pathlib import Path
import re
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

# Change this ONLY if you copied the CSV somewhere else
INPUT_CSV = ROOT / "NER_Landslide_Incidences.csv"

OUTPUT_DIR = ROOT / "pilot_route_data"

OUTPUT_CSV = OUTPUT_DIR / "historical_landslide_points.csv"


# ============================================================
# DMS CONVERSION
# ============================================================

def dms_to_decimal(degrees, minutes, seconds, hemisphere):

    value = (
        float(degrees)
        + float(minutes) / 60
        + float(seconds) / 3600
    )

    if hemisphere.upper() in ["S", "W"]:
        value = -value

    return value


# ============================================================
# EXTRACT COORDINATES
# ============================================================

def extract_coordinates(text):

    if not isinstance(text, str):
        return []

    # Latitude:
    # Example:
    # Lat: 24°51’19.3”N
    lat_pattern = re.compile(
        r"Lat:\s*"
        r"(\d{1,2})[°º]\s*"
        r"(\d{1,2})[\'’′]\s*"
        r"([\d.]+)[\"”″]?\s*"
        r"([NS])",
        re.IGNORECASE
    )

    # Longitude:
    # Example:
    # Lon: 93°37’26.5”E
    lon_pattern = re.compile(
        r"Lon:\s*"
        r"(\d{1,3})[°º]\s*"
        r"(\d{1,2})[\'’′]\s*"
        r"([\d.]+)[\"”″]?\s*"
        r"([EW])",
        re.IGNORECASE
    )

    lat_matches = list(lat_pattern.finditer(text))
    lon_matches = list(lon_pattern.finditer(text))

    coordinates = []

    # Pair latitude and longitude in their appearance order.
    for lat_match, lon_match in zip(lat_matches, lon_matches):

        lat = dms_to_decimal(
            lat_match.group(1),
            lat_match.group(2),
            lat_match.group(3),
            lat_match.group(4)
        )

        lon = dms_to_decimal(
            lon_match.group(1),
            lon_match.group(2),
            lon_match.group(3),
            lon_match.group(4)
        )

        coordinates.append(
            {
                "latitude": lat,
                "longitude": lon
            }
        )

    return coordinates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 3.1 — EXTRACT REAL LANDSLIDE COORDINATES")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK INPUT
    # --------------------------------------------------------

    if not INPUT_CSV.exists():

        raise FileNotFoundError(
            f"\nCSV not found:\n{INPUT_CSV}\n\n"
            "Copy NER_Landslide_Incidences(4).csv "
            "into the project folder."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # READ CSV
    # --------------------------------------------------------

    print("\n[1/4] Reading landslide CSV...")

    df = pd.read_csv(
        INPUT_CSV
    )

    print(f"Records in CSV: {len(df):,}")

    required_columns = [
        "Title",
        "LandslideIncidence"
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: {column}"
            )

    # --------------------------------------------------------
    # EXTRACT
    # --------------------------------------------------------

    print("\n[2/4] Extracting coordinates...")

    output_rows = []

    point_id = 1

    records_with_coordinates = 0

    for record_index, row in df.iterrows():

        text = row["LandslideIncidence"]

        coordinates = extract_coordinates(
            text
        )

        if coordinates:
            records_with_coordinates += 1

        for coordinate_number, coord in enumerate(
            coordinates,
            start=1
        ):

            output_rows.append(
                {
                    "landslide_id": point_id,
                    "source_record": int(record_index),
                    "coordinate_number": coordinate_number,
                    "latitude": coord["latitude"],
                    "longitude": coord["longitude"],
                    "title": str(row["Title"]).strip()
                }
            )

            point_id += 1

    # --------------------------------------------------------
    # CREATE DATAFRAME
    # --------------------------------------------------------

    landslide_points = pd.DataFrame(
        output_rows
    )

    # --------------------------------------------------------
    # VALIDATE
    # --------------------------------------------------------

    print("\n[3/4] Validating coordinates...")

    if len(landslide_points) == 0:

        raise RuntimeError(
            "No coordinates were extracted."
        )

    # Basic geographic sanity check
    valid_coordinates = landslide_points[
        landslide_points["latitude"].between(
            20, 35
        )
        &
        landslide_points["longitude"].between(
            85, 100
        )
    ].copy()

    invalid_count = (
        len(landslide_points)
        - len(valid_coordinates)
    )

    print(
        f"Coordinate points extracted: "
        f"{len(landslide_points):,}"
    )

    print(
        f"Source records containing coordinates: "
        f"{records_with_coordinates:,}"
    )

    print(
        f"Coordinates passing NER-area sanity check: "
        f"{len(valid_coordinates):,}"
    )

    print(
        f"Coordinates outside sanity bounds: "
        f"{invalid_count:,}"
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    print("\n[4/4] Saving...")

    valid_coordinates.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 70)
    print("STEP 3.1 COMPLETE!")
    print("=" * 70)

    print("\nOutput:")
    print(OUTPUT_CSV)

    print("\nColumns:")
    for column in valid_coordinates.columns:
        print(f"  {column}")

    print("\nExtracted points:")
    print(
        valid_coordinates[
            [
                "landslide_id",
                "latitude",
                "longitude",
                "title"
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()