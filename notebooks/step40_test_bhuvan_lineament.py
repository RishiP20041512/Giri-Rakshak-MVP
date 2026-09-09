# ============================================================
# STEP 40A — TEST BHUVAN LINEAMENT WMS
# ============================================================

import requests
from pathlib import Path


# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# BHUVAN WMS
# ------------------------------------------------------------

WMS_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"


# ------------------------------------------------------------
# NER LINEAMENT LAYERS
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
# TEST POINT
# ------------------------------------------------------------
#
# This is one of the NER training points previously used
# during the geomorphology/LULC testing.
#
# ------------------------------------------------------------

TEST_LON = 92.63415
TEST_LAT = 27.45


# ------------------------------------------------------------
# TEST PARAMETERS
# ------------------------------------------------------------

BBOX_SIZE = 0.02

MINX = TEST_LON - BBOX_SIZE
MAXX = TEST_LON + BBOX_SIZE
MINY = TEST_LAT - BBOX_SIZE
MAXY = TEST_LAT + BBOX_SIZE

WIDTH = 101
HEIGHT = 101


# ------------------------------------------------------------
# PRINT HEADER
# ------------------------------------------------------------

print("=" * 70)
print("STEP 40A — BHUVAN LINEAMENT WMS TEST")
print("=" * 70)

print("\nWMS URL:")
print(WMS_URL)

print("\nTest point:")
print("Longitude:", TEST_LON)
print("Latitude :", TEST_LAT)


# ------------------------------------------------------------
# 1. TEST GETCAPABILITIES
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("1. TESTING WMS GETCAPABILITIES")
print("-" * 70)

capabilities_params = {
    "service": "WMS",
    "request": "GetCapabilities",
    "version": "1.1.1",
}

try:

    response = requests.get(
        WMS_URL,
        params=capabilities_params,
        timeout=60
    )

    print("HTTP status:", response.status_code)
    print("Response size:", len(response.content), "bytes")

    if response.status_code != 200:

        raise RuntimeError(
            "Bhuvan GetCapabilities request failed."
        )

    capabilities_text = response.text

    print("GetCapabilities request successful.")

except Exception as e:

    print("\nERROR while accessing Bhuvan:")
    print(e)

    raise


# ------------------------------------------------------------
# 2. CHECK NER LAYERS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("2. CHECKING NER LINEAMENT LAYERS")
print("-" * 70)

for state, layer_name in LINEAMENT_LAYERS.items():

    if layer_name in capabilities_text:

        print("FOUND :", state, "→", layer_name)

    else:

        print("NOT FOUND :", state, "→", layer_name)


# ------------------------------------------------------------
# 3. TEST GETMAP
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3. TESTING GETMAP")
print("-" * 70)

test_layer = "lineament:AR_LN50K_0506"

getmap_params = {
    "service": "WMS",
    "version": "1.1.1",
    "request": "GetMap",
    "layers": test_layer,
    "styles": "",
    "srs": "EPSG:4326",
    "bbox": f"{MINX},{MINY},{MAXX},{MAXY}",
    "width": WIDTH,
    "height": HEIGHT,
    "format": "image/png",
    "transparent": "true",
}


try:

    map_response = requests.get(
        WMS_URL,
        params=getmap_params,
        timeout=60
    )

    print("HTTP status:", map_response.status_code)
    print(
        "Response size:",
        len(map_response.content),
        "bytes"
    )
    print(
        "Content type:",
        map_response.headers.get("Content-Type")
    )

    if map_response.status_code != 200:

        raise RuntimeError(
            "Bhuvan GetMap request failed."
        )

    output_dir = PROJECT_ROOT / "raw_data" / "lineaments"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "test_arunachal_lineament.png"
    )

    with open(output_file, "wb") as f:

        f.write(map_response.content)

    print("\nTest map saved to:")
    print(output_file)

except Exception as e:

    print("\nERROR during GetMap:")
    print(e)

    raise


# ------------------------------------------------------------
# 4. TEST GETFEATUREINFO
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("4. TESTING GETFEATUREINFO")
print("-" * 70)

getfeatureinfo_params = {
    "service": "WMS",
    "version": "1.1.1",
    "request": "GetFeatureInfo",
    "layers": test_layer,
    "query_layers": test_layer,
    "styles": "",
    "srs": "EPSG:4326",
    "bbox": f"{MINX},{MINY},{MAXX},{MAXY}",
    "width": WIDTH,
    "height": HEIGHT,
    "x": WIDTH // 2,
    "y": HEIGHT // 2,
    "info_format": "text/html",
    "feature_count": 10,
}


try:

    info_response = requests.get(
        WMS_URL,
        params=getfeatureinfo_params,
        timeout=60
    )

    print(
        "HTTP status:",
        info_response.status_code
    )

    print(
        "Content type:",
        info_response.headers.get("Content-Type")
    )

    print("\nResponse:")
    print("-" * 70)

    print(
        info_response.text[:5000]
    )

    print("-" * 70)

except Exception as e:

    print("\nERROR during GetFeatureInfo:")
    print(e)

    raise


# ------------------------------------------------------------
# 5. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 40A TEST COMPLETED")
print("=" * 70)

print(
    "\nIf GetCapabilities, GetMap and GetFeatureInfo "
    "work correctly, we will determine the best way "
    "to obtain lineament geometries for density."
)

print("=" * 70)