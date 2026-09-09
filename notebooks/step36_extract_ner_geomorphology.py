import geopandas as gpd
import pandas as pd
import requests
import re
import time
import unicodedata
import html as html_module


# =========================================================
# FILE PATHS
# =========================================================

POINTS_FILE = r".\processed\ner_training_points.geojson"

BOUNDARY_FILE = r".\raw_data\boundaries\geoBoundaries-IND-ADM1.geojson"

OUTPUT_FILE = r".\processed\ner_geomorphology.csv"


# =========================================================
# BHUVAN WMS
# =========================================================

WMS_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"


# =========================================================
# BHUVAN GEOMORPHOLOGY 1:50,000 LAYERS
# =========================================================

LAYERS = {
    "ARUNACHAL PRADESH": "geomorphology:AR_GM50K_0506",
    "ASSAM": "geomorphology:AS_GM50K_0506",
    "MANIPUR": "geomorphology:MN_GM50K_0506",
    "MEGHALAYA": "geomorphology:ML_GM50K_0506",
    "MIZORAM": "geomorphology:MZ_GM50K_0506",
    "NAGALAND": "geomorphology:NL_GM50K_0506",
    "SIKKIM": "geomorphology:SK_GM50K_0506",
    "TRIPURA": "geomorphology:TR_GM50K_0506",
}


# =========================================================
# NORMALIZE STATE NAMES
# =========================================================

def normalize_state_name(name):

    if name is None:
        return ""

    name = str(name)

    # Remove accents from names such as:
    # Nāgāland -> NAGALAND
    # Arunāchal Pradesh -> ARUNACHAL PRADESH

    name = unicodedata.normalize("NFKD", name)

    name = "".join(
        c for c in name
        if not unicodedata.combining(c)
    )

    return name.upper().strip()


# =========================================================
# READ TRAINING POINTS
# =========================================================

print("=" * 60)
print("STEP 36 - NER GEOMORPHOLOGY EXTRACTION")
print("=" * 60)

print("\nReading training points...")

points = gpd.read_file(POINTS_FILE)

print("Number of points:", len(points))
print("Columns:", list(points.columns))
print("CRS:", points.crs)


# =========================================================
# MAKE SURE POINTS ARE EPSG:4326
# =========================================================

if points.crs is None:

    points = points.set_crs("EPSG:4326")

else:

    points = points.to_crs("EPSG:4326")


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
# IF TRAINING POINTS DON'T HAVE STATE,
# FIND STATE USING NER BOUNDARY
# =========================================================

if state_column is None:

    print("\nNo state column found in training points.")

    print(
        "We will identify each point's state "
        "using the NER boundary."
    )

    print("\nReading NER boundary...")

    boundary = gpd.read_file(BOUNDARY_FILE)

    print(
        "Boundary columns:",
        list(boundary.columns)
    )

    if boundary.crs is None:

        boundary = boundary.set_crs("EPSG:4326")

    else:

        boundary = boundary.to_crs("EPSG:4326")


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
            "Could not identify the state-name "
            "column in the boundary file."
        )


    print(
        "Using boundary state column:",
        boundary_state_column
    )


    # -----------------------------------------------------
    # KEEP ONLY STATE NAME + GEOMETRY
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
# NORMALIZE STATE NAMES
# =========================================================

points["state_normalized"] = points[
    "state_name"
].apply(normalize_state_name)


# =========================================================
# PRINT STATE DISTRIBUTION
# =========================================================

print("\nState distribution:")

print(
    points["state_normalized"]
    .value_counts(dropna=False)
)


# =========================================================
# CHECK FOR UNKNOWN STATES
# =========================================================

unknown_states = sorted(
    set(points["state_normalized"].dropna())
    - set(LAYERS.keys())
)


if unknown_states:

    print("\nWARNING: Unknown states found:")

    for state in unknown_states:

        print("  ", state)


# =========================================================
# CREATE HTTP SESSION
# =========================================================

session = requests.Session()


# =========================================================
# FUNCTION TO QUERY BHUVAN
# =========================================================

def get_geomorphology(lon, lat, layer):

    """
    Query Bhuvan WMS GetFeatureInfo for one point.

    Returns:
        Geomorphology description as string
        or None if no feature is returned.
    """

    # Small bounding box around point
    delta = 0.01

    bbox = (
        f"{lon - delta},"
        f"{lat - delta},"
        f"{lon + delta},"
        f"{lat + delta}"
    )


    # -----------------------------------------------------
    # WMS GETFEATUREINFO PARAMETERS
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
        # CHECK HTTP RESPONSE
        # -------------------------------------------------

        if response.status_code != 200:

            return None


        html = response.text


        # -------------------------------------------------
        # BHUVAN RETURNS HTML LIKE:
        #
        # <tr>
        # <td>AR_GM50K_0506.3605</td>
        # <td>Structural Origin-Moderately Dissected
        # Hills and Valleys
        #
        # </tr>
        #
        # The Description cell may not have a closing
        # </td>, so we capture everything until </tr>.
        # -------------------------------------------------

        match = re.search(

            r"<tr>\s*"
            r"<td>\s*[^<]*GM50K[^<]*</td>\s*"
            r"<td>\s*(.*?)\s*</tr>",

            html,

            flags=re.IGNORECASE | re.DOTALL
        )


        if not match:

            return None


        # -------------------------------------------------
        # GET DESCRIPTION
        # -------------------------------------------------

        description = match.group(1)


        # Remove any HTML tags

        description = re.sub(
            r"<.*?>",
            "",
            description
        )


        # Decode HTML entities

        description = html_module.unescape(
            description
        )


        # Clean whitespace

        description = (
            description
            .replace("\n", " ")
            .replace("\r", " ")
            .replace("\t", " ")
        )


        description = " ".join(
            description.split()
        ).strip()


        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        if description:

            return description


        return None


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
# EXTRACT GEOMORPHOLOGY
# =========================================================

print("\n" + "=" * 60)

print("Starting Bhuvan geomorphology extraction...")

print("=" * 60)

print(
    "Total points:",
    len(points)
)


results = []


total = len(points)


# ---------------------------------------------------------
# LOOP THROUGH ALL POINTS
# ---------------------------------------------------------

for position, (_, row) in enumerate(
    points.iterrows(),
    start=1
):

    # -----------------------------------------------------
    # GET COORDINATES
    # -----------------------------------------------------

    lon = row.geometry.x

    lat = row.geometry.y


    # -----------------------------------------------------
    # GET STATE
    # -----------------------------------------------------

    state = row["state_normalized"]


    # -----------------------------------------------------
    # GET BHUVAN LAYER
    # -----------------------------------------------------

    layer = LAYERS.get(state)


    geomorphology = None


    # -----------------------------------------------------
    # QUERY BHUVAN
    # -----------------------------------------------------

    if layer is not None:

        geomorphology = get_geomorphology(
            lon,
            lat,
            layer
        )


    # -----------------------------------------------------
    # STORE RESULT
    # -----------------------------------------------------

    results.append({

        "longitude": lon,

        "latitude": lat,

        "state": state,

        "geomorphology": geomorphology,

    })


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if (
        position == 1
        or position % 25 == 0
        or position == total
    ):

        valid = sum(
            r["geomorphology"] is not None
            for r in results
        )


        missing = position - valid


        print(
            f"Processed {position}/{total}"
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

df = pd.DataFrame(results)


# =========================================================
# SAVE CSV
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# =========================================================
# FINAL SUMMARY
# =========================================================

valid_count = df[
    "geomorphology"
].notna().sum()


missing_count = df[
    "geomorphology"
].isna().sum()


print("\n")

print("=" * 60)

print("GEOMORPHOLOGY EXTRACTION COMPLETE")

print("=" * 60)

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
# SHOW GEOMORPHOLOGY CLASSES
# =========================================================

print("\nGeomorphology classes:")

class_counts = (
    df["geomorphology"]
    .value_counts(dropna=False)
)


print(class_counts.head(30))


# =========================================================
# SAVE A SECOND SUMMARY FILE
# =========================================================

SUMMARY_FILE = (
    r".\processed"
    r"\ner_geomorphology_summary.csv"
)


class_counts.rename(
    "count"
).to_csv(
    SUMMARY_FILE,
    encoding="utf-8-sig"
)


print(
    "\nClass summary saved to:",
    SUMMARY_FILE
)


print("\nDone.")