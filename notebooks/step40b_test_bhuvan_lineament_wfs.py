# ============================================================
# STEP 40B — TEST BHUVAN LINEAMENT WFS
# ============================================================
#
# Purpose:
#   Check whether Bhuvan exposes the lineament layers through
#   WFS and whether actual line geometries can be retrieved.
#
# This is necessary because the final predictor must be:
#
#       LINEAMENT DENSITY
#
# and not simply a WMS pixel/class value.
#
# ============================================================

import requests
from pathlib import Path


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# 2. BHUVAN WMS / GEOSERVER URL
# ------------------------------------------------------------

WMS_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"

# GeoServer WFS commonly uses the same endpoint.
WFS_URL = WMS_URL


# ------------------------------------------------------------
# 3. TEST LAYERS
# ------------------------------------------------------------

LINEAMENT_LAYERS = {
    "ARUNACHAL PRADESH": "lineament:AR_LN50K_0506",
    "ASSAM": "lineament:AS_LN50K_0506",
    "MANIPUR": "lineament:MN_LN50K_0506",
    "MEGHALAYA": "lineament:ML_LN50K_0506",
    "MIZORAM": "lineament:MZ_LN50K_0506",
    "NAGALAND": "lineament:NL_LN50K_0506",
    "SIKKIM": "lineament:SK_LN50K_0506",
    "TRIPURA": "lineament:TR_LN50K_0506",
}


# ------------------------------------------------------------
# 4. TEST AREA
# ------------------------------------------------------------
#
# Small test area around the previous Arunachal test point.
#
# Longitude: 92.63415
# Latitude : 27.45
#
# ------------------------------------------------------------

TEST_LON = 92.63415
TEST_LAT = 27.45

BBOX_SIZE = 0.05

MINX = TEST_LON - BBOX_SIZE
MAXX = TEST_LON + BBOX_SIZE
MINY = TEST_LAT - BBOX_SIZE
MAXY = TEST_LAT + BBOX_SIZE


# ------------------------------------------------------------
# 5. PRINT HEADER
# ------------------------------------------------------------

print("=" * 70)
print("STEP 40B — BHUVAN LINEAMENT WFS TEST")
print("=" * 70)

print("\nWFS URL:")
print(WFS_URL)

print("\nTest bounding box:")
print(
    f"{MINX}, {MINY}, {MAXX}, {MAXY}"
)


# ------------------------------------------------------------
# 6. TEST WFS GETCAPABILITIES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("1. TESTING WFS GETCAPABILITIES")
print("-" * 70)

capabilities_params = {
    "service": "WFS",
    "request": "GetCapabilities",
    "version": "1.1.0",
}


try:

    response = requests.get(
        WFS_URL,
        params=capabilities_params,
        timeout=60
    )

    print("HTTP status:", response.status_code)

    print(
        "Response size:",
        len(response.content),
        "bytes"
    )

    print(
        "Content type:",
        response.headers.get("Content-Type")
    )

    if response.status_code != 200:

        raise RuntimeError(
            "WFS GetCapabilities request failed."
        )

    capabilities_text = response.text

    # Save capabilities for inspection

    output_dir = (
        PROJECT_ROOT
        / "raw_data"
        / "lineaments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    capabilities_file = (
        output_dir
        / "bhuvan_lineament_wfs_capabilities.xml"
    )

    with open(
        capabilities_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(capabilities_text)

    print(
        "\nCapabilities saved to:"
    )

    print(capabilities_file)


except Exception as e:

    print("\nERROR:")
    print(e)

    raise


# ------------------------------------------------------------
# 7. CHECK WHICH LINEAMENT LAYERS EXIST
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("2. CHECKING LINEAMENT LAYERS IN WFS")
print("-" * 70)

for state, layer_name in LINEAMENT_LAYERS.items():

    if layer_name in capabilities_text:

        print(
            "FOUND     :",
            state,
            "→",
            layer_name
        )

    else:

        print(
            "NOT FOUND :",
            state,
            "→",
            layer_name
        )


# ------------------------------------------------------------
# 8. TEST WFS GETFEATURE — ARUNACHAL PRADESH
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3. TESTING WFS GETFEATURE")
print("-" * 70)

test_layer = "lineament:AR_LN50K_0506"

print("\nTest layer:")
print(test_layer)


getfeature_params = {
    "service": "WFS",
    "version": "1.1.0",
    "request": "GetFeature",
    "typeName": test_layer,
    "outputFormat": "application/json",
    "srsName": "EPSG:4326",
    "bbox": (
        f"{MINX},{MINY},{MAXX},{MAXY},"
        "EPSG:4326"
    ),
    "maxFeatures": 100,
}


try:

    feature_response = requests.get(
        WFS_URL,
        params=getfeature_params,
        timeout=120
    )

    print(
        "\nHTTP status:",
        feature_response.status_code
    )

    print(
        "Response size:",
        len(feature_response.content),
        "bytes"
    )

    print(
        "Content type:",
        feature_response.headers.get(
            "Content-Type"
        )
    )

    if feature_response.status_code != 200:

        print("\nServer response:")
        print(feature_response.text[:5000])

        raise RuntimeError(
            "WFS GetFeature request failed."
        )


    # --------------------------------------------------------
    # SAVE RESPONSE
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / "raw_data"
        / "lineaments"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    geojson_file = (
        output_dir
        / "test_arunachal_lineaments.geojson"
    )

    with open(
        geojson_file,
        "wb"
    ) as f:

        f.write(feature_response.content)

    print(
        "\nWFS response saved to:"
    )

    print(geojson_file)


    # --------------------------------------------------------
    # PRINT FIRST PART OF RESPONSE
    # --------------------------------------------------------

    print("\nFirst part of response:")
    print("-" * 70)

    print(
        feature_response.text[:5000]
    )

    print("-" * 70)


except Exception as e:

    print("\nERROR during GetFeature:")
    print(e)

    raise


# ------------------------------------------------------------
# 9. TRY READING AS GEOJSON
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("4. CHECKING WHETHER RESPONSE IS VALID GEOJSON")
print("-" * 70)

try:

    import geopandas as gpd

    lineaments = gpd.read_file(
        geojson_file
    )

    print(
        "GeoJSON successfully loaded."
    )

    print(
        "Number of lineament features:",
        len(lineaments)
    )

    print(
        "CRS:",
        lineaments.crs
    )

    print(
        "Geometry types:"
    )

    print(
        lineaments.geometry
        .geom_type
        .value_counts()
    )

    print(
        "\nColumns:"
    )

    print(
        list(lineaments.columns)
    )

    if len(lineaments) > 0:

        print(
            "\nFirst feature:"
        )

        print(
            lineaments.iloc[0]
        )

except Exception as e:

    print(
        "\nCould not read response as GeoJSON."
    )

    print("Reason:")
    print(e)


# ------------------------------------------------------------
# 10. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 40B TEST COMPLETED")
print("=" * 70)

print(
    "\nSend me the complete terminal output."
)

print(
    "\nDo NOT start the full NER extraction yet."
)

print(
    "We first need to confirm that actual line geometries "
    "are available through WFS."
)

print("=" * 70)