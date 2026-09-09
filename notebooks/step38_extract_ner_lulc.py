import geopandas as gpd
import pandas as pd
import requests
import re
import time
import unicodedata
import html as html_module


# =========================================================
# STEP 38 - NER LULC EXTRACTION
# =========================================================

print("=" * 70)
print("STEP 38 - NER LULC EXTRACTION")
print("=" * 70)


# =========================================================
# FILE PATHS
# =========================================================

POINTS_FILE = r".\processed\ner_training_points.geojson"

BOUNDARY_FILE = (
    r".\raw_data\boundaries\geoBoundaries-IND-ADM1.geojson"
)

OUTPUT_FILE = (
    r".\processed\ner_lulc.csv"
)

SUMMARY_FILE = (
    r".\processed\ner_lulc_summary.csv"
)


# =========================================================
# BHUVAN WMS
# =========================================================

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)


# =========================================================
# BHUVAN LULC LAYERS
# =========================================================

LAYERS = {

    "ARUNACHAL PRADESH":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_AR",

    "ASSAM":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_AS",

    "MANIPUR":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_MN",

    "MEGHALAYA":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_ML",

    "MIZORAM":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_MZ",

    "NAGALAND":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_NL",

    "SIKKIM":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_SK",

    "TRIPURA":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_TR",
}


# =========================================================
# NORMALIZE STATE NAME
# =========================================================

def normalize_state_name(name):

    if name is None:
        return ""

    name = str(name)

    name = unicodedata.normalize(
        "NFKD",
        name
    )

    name = "".join(
        c for c in name
        if not unicodedata.combining(c)
    )

    return name.upper().strip()


# =========================================================
# CLEAN HTML CELL
# =========================================================

def clean_html_cell(value):

    if value is None:
        return ""

    # Remove HTML tags
    value = re.sub(
        r"<.*?>",
        "",
        value,
        flags=re.DOTALL
    )

    # Decode HTML entities
    value = html_module.unescape(value)

    # Clean whitespace
    value = (
        value
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\t", " ")
    )

    value = " ".join(
        value.split()
    ).strip()

    return value


# =========================================================
# READ TRAINING POINTS
# =========================================================

print("\nReading training points...")

points = gpd.read_file(
    POINTS_FILE
)

print(
    "Number of points:",
    len(points)
)

print(
    "Columns:",
    list(points.columns)
)

print(
    "CRS:",
    points.crs
)


# =========================================================
# ENSURE EPSG:4326
# =========================================================

if points.crs is None:

    points = points.set_crs(
        "EPSG:4326"
    )

else:

    points = points.to_crs(
        "EPSG:4326"
    )


# =========================================================
# FIND STATE COLUMN
# =========================================================

possible_state_columns = [

    "state",
    "State",
    "STATE",
    "st_nm",
    "ST_NM",
    "shapeName",
    "ShapeName",
    "NAME_1",
    "NAME1",

]


state_column = None


for col in possible_state_columns:

    if col in points.columns:

        state_column = col

        break


# =========================================================
# IDENTIFY STATE USING BOUNDARY IF NECESSARY
# =========================================================

if state_column is None:

    print(
        "\nNo state column found in training points."
    )

    print(
        "Using NER boundary to identify states..."
    )

    boundary = gpd.read_file(
        BOUNDARY_FILE
    )

    print(
        "Boundary columns:",
        list(boundary.columns)
    )


    # -----------------------------------------------------
    # CRS
    # -----------------------------------------------------

    if boundary.crs is None:

        boundary = boundary.set_crs(
            "EPSG:4326"
        )

    else:

        boundary = boundary.to_crs(
            "EPSG:4326"
        )


    # -----------------------------------------------------
    # FIND STATE NAME FIELD
    # -----------------------------------------------------

    boundary_state_column = None


    possible_boundary_columns = [

        "shapeName",
        "ShapeName",
        "NAME_1",
        "NAME1",
        "state",
        "State",
        "STATE",
        "ST_NM",

    ]


    for col in possible_boundary_columns:

        if col in boundary.columns:

            boundary_state_column = col

            break


    if boundary_state_column is None:

        raise ValueError(
            "Could not identify state-name column "
            "in NER boundary."
        )


    print(
        "Using boundary state column:",
        boundary_state_column
    )


    # -----------------------------------------------------
    # KEEP ONLY NECESSARY COLUMNS
    # -----------------------------------------------------

    boundary_small = boundary[
        [boundary_state_column, "geometry"]
    ].copy()


    # -----------------------------------------------------
    # SPATIAL JOIN
    # -----------------------------------------------------

    points = gpd.sjoin(

        points,

        boundary_small,

        how="left",

        predicate="within"

    )


    points["state_name"] = points[
        boundary_state_column
    ]


else:

    print(
        "\nUsing existing state column:",
        state_column
    )

    points["state_name"] = points[
        state_column
    ]


# =========================================================
# NORMALIZE STATES
# =========================================================

points["state_normalized"] = (
    points["state_name"]
    .apply(normalize_state_name)
)


# =========================================================
# STATE DISTRIBUTION
# =========================================================

print("\nState distribution:")

print(
    points[
        "state_normalized"
    ].value_counts(
        dropna=False
    )
)


# =========================================================
# CHECK UNKNOWN STATES
# =========================================================

unknown_states = sorted(

    set(
        points[
            "state_normalized"
        ].dropna()
    )

    - set(
        LAYERS.keys()
    )

)


if unknown_states:

    print(
        "\nWARNING - Unknown states found:"
    )

    for state in unknown_states:

        print(
            "  ",
            state
        )


# =========================================================
# HTTP SESSION
# =========================================================

session = requests.Session()


# =========================================================
# GET LULC FROM BHUVAN
# =========================================================

def get_lulc(
    lon,
    lat,
    layer
):

    """
    Query Bhuvan WMS GetFeatureInfo.

    Returns a dictionary containing the actual
    Bhuvan LULC fields.
    """

    # -----------------------------------------------------
    # SMALL BOUNDING BOX
    # -----------------------------------------------------

    delta = 0.01

    bbox = (

        f"{lon - delta},"
        f"{lat - delta},"
        f"{lon + delta},"
        f"{lat + delta}"

    )


    # -----------------------------------------------------
    # WMS PARAMETERS
    # -----------------------------------------------------

    params = {

        "SERVICE": "WMS",

        "VERSION": "1.1.1",

        "REQUEST": "GetFeatureInfo",

        "LAYERS": layer,

        "QUERY_LAYERS": layer,

        "STYLES": "",

        "SRS": "EPSG:4326",

        "BBOX": bbox,

        "WIDTH": 101,

        "HEIGHT": 101,

        "X": 50,

        "Y": 50,

        "INFO_FORMAT": "text/html",

    }


    try:

        response = session.get(

            WMS_URL,

            params=params,

            timeout=60

        )


        # -------------------------------------------------
        # CHECK HTTP STATUS
        # -------------------------------------------------

        if response.status_code != 200:

            return None


        html = response.text


        # -------------------------------------------------
        # FIND ALL TABLE ROWS
        # -------------------------------------------------

        rows = re.findall(

            r"<tr[^>]*>(.*?)</tr>",

            html,

            flags=re.IGNORECASE | re.DOTALL

        )


        if not rows:

            return None


        # -------------------------------------------------
        # FIND HEADER ROW
        # -------------------------------------------------

        headers = None


        for row_html in rows:

            header_cells = re.findall(

                r"<th[^>]*>(.*?)</th>",

                row_html,

                flags=re.IGNORECASE | re.DOTALL

            )


            if header_cells:

                headers = [

                    clean_html_cell(cell)

                    for cell in header_cells

                ]

                break


        if headers is None:

            return None


        # -------------------------------------------------
        # FIND DATA ROW
        # -------------------------------------------------

        data_values = None


        for row_html in rows:

            data_cells = re.findall(

                r"<td[^>]*>(.*?)</td>",

                row_html,

                flags=re.IGNORECASE | re.DOTALL

            )


            if len(data_cells) == len(headers):

                cleaned = [

                    clean_html_cell(cell)

                    for cell in data_cells

                ]

                data_values = cleaned

                break


        # -------------------------------------------------
        # IF NORMAL </td> PARSING FAILED
        # -------------------------------------------------

        if data_values is None:

            # Bhuvan may omit the closing </td>
            # for the final field.

            for row_html in rows:

                data_cells = re.findall(

                    r"<td[^>]*>(.*?)"
                    r"(?=<td|</tr>)",

                    row_html,

                    flags=re.IGNORECASE | re.DOTALL

                )


                if len(data_cells) >= 9:

                    cleaned = [

                        clean_html_cell(cell)

                        for cell in data_cells

                    ]

                    data_values = cleaned

                    break


        if data_values is None:

            return None


        # -------------------------------------------------
        # CREATE HEADER -> VALUE DICTIONARY
        # -------------------------------------------------

        record = {}

        for i, header in enumerate(headers):

            if i < len(data_values):

                record[header] = data_values[i]


        # -------------------------------------------------
        # EXPECTED FIELDS
        # -------------------------------------------------

        required_fields = [

            "REG_DESCRI",
            "LULC_1",
            "LULC_2",
            "LULC_3",
            "Level_I",
            "Level_II",
            "Level_III",
            "Area_SqKm",
            "Level_IV",

        ]


        # -------------------------------------------------
        # CHECK THAT LULC FIELDS EXIST
        # -------------------------------------------------

        available_fields = [

            field

            for field in required_fields

            if field in record

        ]


        if not available_fields:

            return None


        # -------------------------------------------------
        # RETURN ONLY EXPECTED LULC FIELDS
        # -------------------------------------------------

        result = {}

        for field in required_fields:

            result[field] = record.get(
                field,
                None
            )


        return result


    except requests.exceptions.Timeout:

        print(
            f"\nTimeout at "
            f"({lon}, {lat})"
        )

        return None


    except requests.exceptions.RequestException as e:

        print(
            f"\nRequest error at "
            f"({lon}, {lat}): {e}"
        )

        return None


    except Exception as e:

        print(
            f"\nUnexpected error at "
            f"({lon}, {lat}): {e}"
        )

        return None


# =========================================================
# EXTRACT LULC
# =========================================================

print("\n" + "=" * 70)

print(
    "Starting Bhuvan LULC extraction..."
)

print("=" * 70)

print(
    "Total points:",
    len(points)
)


results = []


total = len(points)


# =========================================================
# LOOP THROUGH TRAINING POINTS
# =========================================================

for position, (_, row) in enumerate(

    points.iterrows(),

    start=1

):

    # -----------------------------------------------------
    # COORDINATES
    # -----------------------------------------------------

    lon = row.geometry.x

    lat = row.geometry.y


    # -----------------------------------------------------
    # STATE
    # -----------------------------------------------------

    state = row[
        "state_normalized"
    ]


    # -----------------------------------------------------
    # BHUVAN LAYER
    # -----------------------------------------------------

    layer = LAYERS.get(
        state
    )


    # -----------------------------------------------------
    # QUERY
    # -----------------------------------------------------

    lulc = None


    if layer is not None:

        lulc = get_lulc(

            lon,

            lat,

            layer

        )


    # -----------------------------------------------------
    # PRESERVE ORIGINAL POINT INFORMATION
    # -----------------------------------------------------

    result = {

        "lat":
            lat,

        "lon":
            lon,

        "date":
            row.get(
                "date",
                None
            ),

        "source":
            row.get(
                "source",
                None
            ),

        "region":
            row.get(
                "region",
                None
            ),

        "label":
            row.get(
                "label",
                None
            ),

        "state":
            state,

        "REG_DESCRI":
            None,

        "LULC_1":
            None,

        "LULC_2":
            None,

        "LULC_3":
            None,

        "Level_I":
            None,

        "Level_II":
            None,

        "Level_III":
            None,

        "Area_SqKm":
            None,

        "Level_IV":
            None,

    }


    # -----------------------------------------------------
    # ADD BHUVAN RESULT
    # -----------------------------------------------------

    if lulc is not None:

        result.update(
            lulc
        )


    results.append(
        result
    )


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if (

        position == 1

        or position % 25 == 0

        or position == total

    ):

        valid = sum(

            r["Level_I"] not in [
                None,
                ""
            ]

            for r in results

        )


        missing = (
            position - valid
        )


        print(

            f"Processed "
            f"{position}/{total}"
            f" | Valid: {valid}"
            f" | Missing: {missing}"

        )


    # -----------------------------------------------------
    # SMALL DELAY
    # -----------------------------------------------------

    time.sleep(0.15)


# =========================================================
# CREATE DATAFRAME
# =========================================================

df = pd.DataFrame(
    results
)


# =========================================================
# DATA QUALITY CHECK
# =========================================================

print("\n" + "=" * 70)

print(
    "DATA QUALITY CHECK"
)

print("=" * 70)


# ---------------------------------------------------------
# VALID LEVEL-I
# ---------------------------------------------------------

valid_level1 = (

    df["Level_I"]
    .notna()

    &

    (
        df["Level_I"]
        .astype(str)
        .str.strip()
        != ""
    )

)


# ---------------------------------------------------------
# VALID LEVEL-IV
# ---------------------------------------------------------

valid_level4 = (

    df["Level_IV"]
    .notna()

    &

    (
        df["Level_IV"]
        .astype(str)
        .str.strip()
        != ""
    )

)


# =========================================================
# CHECK FOR NUMERIC VALUES IN LEVEL-IV
# =========================================================

numeric_level4 = pd.to_numeric(

    df["Level_IV"],

    errors="coerce"

)


numeric_mask = (

    numeric_level4.notna()

    &

    df["Level_IV"].notna()

)


numeric_count = numeric_mask.sum()


print(
    "Valid Level-I:",
    valid_level1.sum()
)

print(
    "Valid Level-IV:",
    valid_level4.sum()
)

print(
    "Numeric values incorrectly in Level-IV:",
    numeric_count
)


# ---------------------------------------------------------
# REMOVE NUMERIC VALUES FROM LEVEL-IV
# ---------------------------------------------------------

if numeric_count > 0:

    print(
        "\nWARNING:"
    )

    print(
        "Numeric values were found in Level-IV."
    )

    print(
        "They will be set to missing rather than "
        "being treated as LULC classes."
    )

    df.loc[
        numeric_mask,
        "Level_IV"
    ] = None


# =========================================================
# SAVE MAIN CSV
# =========================================================

df.to_csv(

    OUTPUT_FILE,

    index=False,

    encoding="utf-8-sig"

)


# =========================================================
# FINAL COUNTS
# =========================================================

valid_count = (

    df["Level_I"]
    .notna()
    .sum()

)


missing_count = (

    len(df)
    - valid_count

)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")

print("=" * 70)

print(
    "LULC EXTRACTION COMPLETE"
)

print("=" * 70)

print(
    "Output:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(df)
)

print(
    "Valid:",
    valid_count
)

print(
    "Missing:",
    missing_count
)


# =========================================================
# LEVEL-I SUMMARY
# =========================================================

print("\n")

print(
    "LULC Level-I classes:"
)

level1_counts = (

    df["Level_I"]
    .value_counts(
        dropna=False
    )

)

print(
    level1_counts
)


# =========================================================
# LEVEL-II SUMMARY
# =========================================================

print("\n")

print(
    "LULC Level-II classes:"
)

level2_counts = (

    df["Level_II"]
    .value_counts(
        dropna=False
    )

)

print(
    level2_counts.head(30)
)


# =========================================================
# LEVEL-III SUMMARY
# =========================================================

print("\n")

print(
    "LULC Level-III classes:"
)

level3_counts = (

    df["Level_III"]
    .value_counts(
        dropna=False
    )

)

print(
    level3_counts.head(30)
)


# =========================================================
# LEVEL-IV SUMMARY
# =========================================================

print("\n")

print(
    "LULC Level-IV classes:"
)

level4_counts = (

    df["Level_IV"]
    .value_counts(
        dropna=False
    )

)

print(
    level4_counts.head(50)
)


# =========================================================
# SAVE LEVEL-IV SUMMARY
# =========================================================

level4_counts.rename(
    "count"
).to_csv(

    SUMMARY_FILE,

    encoding="utf-8-sig"

)


# =========================================================
# FINAL MESSAGE
# =========================================================

print("\n")

print(
    "Class summary saved to:",
    SUMMARY_FILE
)

print(
    "\nDone."
)