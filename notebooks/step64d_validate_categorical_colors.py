"""
STEP 64D-C — VALIDATE BHUVAN CATEGORICAL COLOR MAPPING

Purpose
-------
Validate the relationship between:

    Bhuvan WMS rendered RGB color
                    ↓
        actual Bhuvan class

using the existing training points.

Factors:
    1. Geomorphology
    2. LUCC

Final modeling variables:
    geomorph_origin
    Level_I

This script does NOT create final categorical rasters.
"""

from pathlib import Path
import time

import requests
import numpy as np
import pandas as pd
import geopandas as gpd

from PIL import Image
from io import BytesIO


# ============================================================
# PROJECT
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

PROCESSED = PROJECT / "processed"

OUT_DIR = (
    PROCESSED
    / "categorical_color_validation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT FILES
# ============================================================

POINTS_FILE = (
    PROCESSED
    / "ner_training_points.geojson"
)

GEOMORPH_FILE = (
    PROCESSED
    / "ner_geomorphology.csv"
)

LUCC_FILE = (
    PROCESSED
    / "ner_lulc.csv"
)


# ============================================================
# BHUVAN WMS
# ============================================================

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)


# ============================================================
# Bhuvan layers
# ============================================================

GEOMORPH_LAYERS = {

    "AR":
        "geomorphology:AR_GM50K_0506",

    "AS":
        "geomorphology:AS_GM50K_0506",

    "MN":
        "geomorphology:MN_GM50K_0506",

    "ML":
        "geomorphology:ML_GM50K_0506",

    "MZ":
        None,

    "NL":
        "geomorphology:NL_GM50K_0506",

    "SK":
        "geomorphology:SK_GM50K_0506",

    "TR":
        "geomorphology:TR_GM50K_0506",
}


LUCC_LAYERS = {

    "AR":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_AR",

    "AS":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_AS",

    "MN":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_MN",

    "ML":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_ML",

    "MZ":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_MZ",

    "NL":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_NL",

    "SK":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_SK",

    "TR":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_TR",
}


# ============================================================
# HTTP SESSION
# ============================================================

session = requests.Session()

session.headers.update({

    "User-Agent":
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "Chrome/131.0 Safari/537.36"

})


# ============================================================
# ROBUST HTTP REQUEST
# ============================================================

def get_request(
    url,
    attempts=3,
    timeout=(30, 180)
):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        print(
            f"      HTTP attempt "
            f"{attempt}/{attempts}"
        )

        try:

            response = session.get(
                url,
                timeout=timeout
            )

            print(
                f"      HTTP {response.status_code}"
            )

            response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:

            last_error = e

            print(
                f"      Request failed: {e}"
            )

            if attempt < attempts:

                print(
                    "      Waiting 5 seconds..."
                )

                time.sleep(5)

    raise last_error


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print(
    "STEP 64D-C — BHUVAN COLOR VALIDATION"
)
print("=" * 70)


# ============================================================
# 1. LOAD TRAINING POINTS
# ============================================================

print()
print(
    "[1] Loading training points"
)

points = gpd.read_file(
    POINTS_FILE
)

print(
    f"Training points: {len(points)}"
)

print(
    "Point columns:"
)

print(
    list(points.columns)
)


# ============================================================
# 2. LOAD GEOMORPHOLOGY
# ============================================================

print()
print(
    "[2] Loading geomorphology"
)

geomorph = pd.read_csv(
    GEOMORPH_FILE
)

print(
    f"Geomorphology rows: {len(geomorph)}"
)

print(
    "Geomorphology columns:"
)

print(
    list(geomorph.columns)
)


# ============================================================
# 3. LOAD LUCC
# ============================================================

print()
print(
    "[3] Loading LUCC"
)

lulc = pd.read_csv(
    LUCC_FILE
)

print(
    f"LUCC rows: {len(lulc)}"
)

print(
    "LUCC columns:"
)

print(
    list(lulc.columns)
)


# ============================================================
# 4. IDENTIFY COORDINATE COLUMNS
# ============================================================

print()
print(
    "[4] Identifying coordinate columns"
)


def find_coordinate_columns(
    dataframe,
    name
):

    columns_lower = {
        str(c).lower(): c
        for c in dataframe.columns
    }

    longitude_candidates = [
        "longitude",
        "lon",
        "long",
        "x"
    ]

    latitude_candidates = [
        "latitude",
        "lat",
        "y"
    ]

    lon_col = None
    lat_col = None

    for candidate in longitude_candidates:

        if candidate in columns_lower:

            lon_col = (
                columns_lower[candidate]
            )

            break

    for candidate in latitude_candidates:

        if candidate in columns_lower:

            lat_col = (
                columns_lower[candidate]
            )

            break

    if lon_col is None or lat_col is None:

        raise ValueError(
            f"\nCould not identify coordinates "
            f"in {name}.\n"
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )

    print(
        f"{name}: longitude = {lon_col}, "
        f"latitude = {lat_col}"
    )

    return lon_col, lat_col


geomorph_lon, geomorph_lat = (
    find_coordinate_columns(
        geomorph,
        "Geomorphology"
    )
)

lulc_lon, lulc_lat = (
    find_coordinate_columns(
        lulc,
        "LUCC"
    )
)


# ============================================================
# 5. EXTRACT TRAINING POINT COORDINATES
# ============================================================

points["lon"] = (
    points.geometry.x
)

points["lat"] = (
    points.geometry.y
)


# ============================================================
# 6. CREATE JOIN KEYS
# ============================================================

points["lon_key"] = (
    points["lon"]
    .round(8)
)

points["lat_key"] = (
    points["lat"]
    .round(8)
)


geomorph["lon_key"] = (
    geomorph[geomorph_lon]
    .round(8)
)

geomorph["lat_key"] = (
    geomorph[geomorph_lat]
    .round(8)
)


lulc["lon_key"] = (
    lulc[lulc_lon]
    .round(8)
)

lulc["lat_key"] = (
    lulc[lulc_lat]
    .round(8)
)


# ============================================================
# 7. JOIN GEOMORPHOLOGY
# ============================================================

print()
print(
    "[5] Joining geomorphology attributes"
)


geomorph_small = geomorph[
    [
        "lon_key",
        "lat_key",
        "geomorphology"
    ]
].copy()


points = points.merge(
    geomorph_small,
    on=[
        "lon_key",
        "lat_key"
    ],
    how="left"
)


print(
    "Geomorphology valid:",
    points["geomorphology"]
    .notna()
    .sum()
)


# ============================================================
# 8. JOIN LUCC
# ============================================================

print()
print(
    "[6] Joining LUCC attributes"
)


lulc_small = lulc[
    [
        "lon_key",
        "lat_key",
        "Level_I"
    ]
].copy()


points = points.merge(
    lulc_small,
    on=[
        "lon_key",
        "lat_key"
    ],
    how="left"
)


print(
    "LUCC valid:",
    points["Level_I"]
    .notna()
    .sum()
)


# ============================================================
# 9. STATE BOUNDARY
# ============================================================

print()
print(
    "[7] Identifying state"
)


BOUNDARY_FILE = (
    PROJECT
    / "raw_data"
    / "boundaries"
    / "geoBoundaries-IND-ADM1.geojson"
)


boundary = gpd.read_file(
    BOUNDARY_FILE
)

boundary = boundary.to_crs(
    "EPSG:4326"
)


points_geo = gpd.GeoDataFrame(
    points.copy(),
    geometry=gpd.points_from_xy(
        points["lon"],
        points["lat"]
    ),
    crs="EPSG:4326"
)


joined = gpd.sjoin(
    points_geo,
    boundary[
        [
            "shapeName",
            "geometry"
        ]
    ],
    how="left",
    predicate="within"
)


points["state_raw"] = (
    joined["shapeName"].values
)


# ============================================================
# STATE NORMALIZATION
# ============================================================

STATE_MAP = {

    "Arunāchal Pradesh":
        "AR",

    "Arunachal Pradesh":
        "AR",

    "Assam":
        "AS",

    "Manipur":
        "MN",

    "Meghālaya":
        "ML",

    "Meghalaya":
        "ML",

    "Mizoram":
        "MZ",

    "Nāgāland":
        "NL",

    "Nagaland":
        "NL",

    "Sikkim":
        "SK",

    "Tripura":
        "TR",
}


points["state"] = (
    points["state_raw"]
    .map(STATE_MAP)
)


print()
print(
    "State distribution:"
)

print(
    points["state"]
    .value_counts(
        dropna=False
    )
    .to_string()
)


# ============================================================
# BBOX FROM TRAINING POINTS
# ============================================================

def create_bbox(
    subset,
    padding=0.05
):

    xmin = (
        subset["lon"].min()
        - padding
    )

    xmax = (
        subset["lon"].max()
        + padding
    )

    ymin = (
        subset["lat"].min()
        - padding
    )

    ymax = (
        subset["lat"].max()
        + padding
    )

    return (
        xmin,
        ymin,
        xmax,
        ymax
    )


# ============================================================
# DOWNLOAD STATE MAP
# ============================================================

def download_state_map(
    state,
    layer,
    factor,
    subset
):

    xmin, ymin, xmax, ymax = (
        create_bbox(
            subset
        )
    )

    width = 3000
    height = 3000

    url = (
        WMS_URL
        + "?SERVICE=WMS"
        + "&VERSION=1.1.1"
        + "&REQUEST=GetMap"
        + "&LAYERS="
        + layer
        + "&STYLES="
        + "&SRS=EPSG:4326"
        + "&BBOX="
        + f"{xmin},{ymin},{xmax},{ymax}"
        + f"&WIDTH={width}"
        + f"&HEIGHT={height}"
        + "&FORMAT=image/png"
        + "&TRANSPARENT=true"
    )

    print()
    print(
        f"    Downloading "
        f"{state} {factor}"
    )

    print(
        f"    BBOX = "
        f"{xmin:.6f},"
        f"{ymin:.6f},"
        f"{xmax:.6f},"
        f"{ymax:.6f}"
    )

    response = get_request(
        url,
        attempts=3,
        timeout=(30, 180)
    )

    image = Image.open(
        BytesIO(
            response.content
        )
    ).convert(
        "RGBA"
    )

    arr = np.array(
        image
    )

    path = (
        OUT_DIR
        / f"{state}_{factor}_validation.png"
    )

    image.save(
        path
    )

    print(
        f"    Saved: {path}"
    )

    return (
        arr,
        xmin,
        ymin,
        xmax,
        ymax
    )


# ============================================================
# EXTRACT RGB AT TRAINING POINTS
# ============================================================

def extract_point_colors(
    subset,
    image,
    xmin,
    ymin,
    xmax,
    ymax
):

    height, width = (
        image.shape[:2]
    )

    rows = []

    for idx, point in subset.iterrows():

        lon = float(
            point["lon"]
        )

        lat = float(
            point["lat"]
        )

        x_fraction = (
            (lon - xmin)
            / (xmax - xmin)
        )

        y_fraction = (
            (ymax - lat)
            / (ymax - ymin)
        )

        col = int(
            round(
                x_fraction
                * (width - 1)
            )
        )

        row = int(
            round(
                y_fraction
                * (height - 1)
            )
        )

        if (
            row < 0
            or row >= height
            or col < 0
            or col >= width
        ):

            continue

        rgba = (
            image[
                row,
                col
            ]
        )

        r = int(
            rgba[0]
        )

        g = int(
            rgba[1]
        )

        b = int(
            rgba[2]
        )

        a = int(
            rgba[3]
        )

        rows.append({

            "point_index":
                idx,

            "state":
                point["state"],

            "lat":
                lat,

            "lon":
                lon,

            "row":
                row,

            "col":
                col,

            "R":
                r,

            "G":
                g,

            "B":
                b,

            "A":
                a,

            "rgba":
                f"{r},{g},{b},{a}",

            "geomorphology":
                point["geomorphology"],

            "Level_I":
                point["Level_I"],

            "label":
                point["label"]

        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROCESS FACTOR
# ============================================================

def process_factor(
    factor_name,
    layer_dictionary,
    class_column
):

    all_results = []

    print()
    print("=" * 70)
    print(
        f"PROCESSING {factor_name.upper()}"
    )
    print("=" * 70)

    for state in [
        "AR",
        "AS",
        "MN",
        "ML",
        "MZ",
        "NL",
        "SK",
        "TR"
    ]:

        layer = (
            layer_dictionary
            .get(state)
        )

        state_points = points[
            points["state"] == state
        ].copy()

        if state_points.empty:

            print()
            print(
                f"{state}: no training points"
            )

            continue

        if layer is None:

            print()
            print(
                f"{state}: Bhuvan layer unavailable"
            )

            continue

        state_points = state_points[
            state_points[class_column]
            .notna()
        ].copy()

        if state_points.empty:

            print()
            print(
                f"{state}: no valid class values"
            )

            continue

        print()
        print(
            f"{state}: "
            f"{len(state_points)} valid points"
        )

        try:

            (
                image,
                xmin,
                ymin,
                xmax,
                ymax
            ) = download_state_map(
                state,
                layer,
                factor_name,
                state_points
            )

        except Exception as e:

            print()
            print(
                f"{state}: GetMap failed"
            )

            print(
                e
            )

            continue

        result = extract_point_colors(
            state_points,
            image,
            xmin,
            ymin,
            xmax,
            ymax
        )

        if not result.empty:

            all_results.append(
                result
            )

            print(
                f"    Colors extracted: "
                f"{len(result)}"
            )

    if all_results:

        return pd.concat(
            all_results,
            ignore_index=True
        )

    return pd.DataFrame()


# ============================================================
# 10. GEOMORPHOLOGY VALIDATION
# ============================================================

geomorph_results = process_factor(
    "geomorphology",
    GEOMORPH_LAYERS,
    "geomorphology"
)


geomorph_output = (
    PROCESSED
    / "step64d_geomorphology_color_validation.csv"
)


geomorph_results.to_csv(
    geomorph_output,
    index=False
)


print()
print(
    f"Saved: {geomorph_output}"
)


# ============================================================
# 11. LUCC VALIDATION
# ============================================================

lulc_results = process_factor(
    "lulc",
    LUCC_LAYERS,
    "Level_I"
)


lulc_output = (
    PROCESSED
    / "step64d_lulc_color_validation.csv"
)


lulc_results.to_csv(
    lulc_output,
    index=False
)


print()
print(
    f"Saved: {lulc_output}"
)


# ============================================================
# 12. COLOR → CLASS TABLE
# ============================================================

def make_color_class_table(
    data,
    class_column,
    output_name
):

    if data.empty:

        print(
            f"No data for {output_name}"
        )

        return

    valid = data[
        data["A"] > 0
    ].copy()

    # Remove pure white background

    valid = valid[
        ~(
            (valid["R"] == 255)
            &
            (valid["G"] == 255)
            &
            (valid["B"] == 255)
        )
    ].copy()

    if valid.empty:

        print(
            f"No usable colored pixels "
            f"for {output_name}"
        )

        return

    table = (
        valid
        .groupby(
            [
                "R",
                "G",
                "B",
                "A",
                "rgba",
                class_column
            ],
            dropna=False
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    totals = (
        table
        .groupby(
            "rgba"
        )["count"]
        .transform(
            "sum"
        )
    )

    table["color_total"] = (
        totals
    )

    table["class_fraction"] = (
        table["count"]
        / table["color_total"]
    )

    table = table.sort_values(
        [
            "rgba",
            "count"
        ],
        ascending=[
            True,
            False
        ]
    )

    # --------------------------------------------------------
    # Dominant class for each RGB
    # --------------------------------------------------------

    dominant = (
        table
        .sort_values(
            "count",
            ascending=False
        )
        .groupby(
            "rgba"
        )
        .first()
        .reset_index()
    )

    dominant = dominant[
        [
            "rgba",
            class_column,
            "count",
            "color_total",
            "class_fraction"
        ]
    ].copy()

    dominant = dominant.rename(
        columns={
            class_column:
                "dominant_class"
        }
    )

    dominant["mapping_confidence"] = (
        dominant["class_fraction"]
    )

    # --------------------------------------------------------
    # Detailed table
    # --------------------------------------------------------

    detail_path = (
        PROCESSED
        / output_name
    )

    table.to_csv(
        detail_path,
        index=False
    )

    # --------------------------------------------------------
    # Dominant table
    # --------------------------------------------------------

    dominant_path = (
        PROCESSED
        / output_name.replace(
            ".csv",
            "_dominant.csv"
        )
    )

    dominant.to_csv(
        dominant_path,
        index=False
    )

    print()
    print(
        f"Saved detailed table:"
    )

    print(
        detail_path
    )

    print()
    print(
        f"Saved dominant mapping:"
    )

    print(
        dominant_path
    )

    print()
    print(
        "Dominant RGB → class mappings:"
    )

    print()

    print(
        dominant[
            [
                "rgba",
                "dominant_class",
                "count",
                "color_total",
                "mapping_confidence"
            ]
        ]
        .sort_values(
            "count",
            ascending=False
        )
        .head(50)
        .to_string(
            index=False
        )
    )


# ============================================================
# 13. CREATE SUMMARIES
# ============================================================

print()
print("=" * 70)
print(
    "CREATING COLOR → CLASS TABLES"
)
print("=" * 70)


if not geomorph_results.empty:

    make_color_class_table(
        geomorph_results,
        "geomorphology",
        "step64d_geomorphology_color_class_table.csv"
    )


if not lulc_results.empty:

    make_color_class_table(
        lulc_results,
        "Level_I",
        "step64d_lulc_color_class_table.csv"
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print(
    "STEP 64D-C COMPLETED"
)
print("=" * 70)

print()
print(
    "Output directory:"
)

print(
    OUT_DIR
)

print()
print(
    "IMPORTANT:"
)

print(
    "RGB → class mappings are diagnostic only."
)

print(
    "Do NOT create final categorical rasters yet."
)

print()
print(
    "Mizoram geomorphology remains NoData because "
    "the Bhuvan layer is unavailable."
)