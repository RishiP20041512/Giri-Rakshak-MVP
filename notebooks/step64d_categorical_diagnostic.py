"""
STEP 64D-A — BHUVAN CATEGORICAL LAYER DIAGNOSTIC

Purpose
-------
Inspect Bhuvan WMS categorical layers before creating
full-resolution categorical rasters.

Factors:
1. Geomorphology -> geomorph_origin
2. LUCC -> Level_I

This diagnostic does NOT create the final categorical rasters.

It:
- downloads WMS GetCapabilities
- checks known geomorphology and LUCC layers
- requests legend graphics
- requests small diagnostic GetMap images
- retries slow Bhuvan requests
"""

from pathlib import Path
import requests
import time


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

OUT_DIR = (
    PROJECT
    / "raw_data"
    / "categorical_diagnostic"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# BHUVAN WMS
# ============================================================

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)


# ============================================================
# GEOMORPHOLOGY LAYERS
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
        "geomorphology:MZ_GM50K_0506",

    "NL":
        "geomorphology:NL_GM50K_0506",

    "SK":
        "geomorphology:SK_GM50K_0506",

    "TR":
        "geomorphology:TR_GM50K_0506",
}


# ============================================================
# LUCC LAYERS
# ============================================================

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
# DIAGNOSTIC BOUNDING BOXES
# ============================================================

TEST_BBOXES = {

    "AR":
        "91.5,26.5,94.5,28.5",

    "AS":
        "89.5,25.0,94.0,28.0",

    "MN":
        "93.0,23.8,94.5,25.8",

    "ML":
        "89.5,25.0,92.0,26.5",

    "MZ":
        "92.0,21.8,93.5,24.5",

    "NL":
        "93.3,25.0,95.5,27.0",

    "SK":
        "88.0,26.8,88.9,28.2",

    "TR":
        "91.0,22.8,92.5,24.0",
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
# ROBUST GET REQUEST
# ============================================================

def request_with_retry(
    url,
    timeout=(30, 180),
    attempts=3
):

    last_error = None

    for attempt in range(
        1,
        attempts + 1
    ):

        print(
            f"Attempt {attempt}/{attempts}"
        )

        try:

            response = session.get(
                url,
                timeout=timeout
            )

            print(
                f"HTTP {response.status_code}"
            )

            response.raise_for_status()

            return response

        except requests.exceptions.RequestException as e:

            last_error = e

            print(
                f"Request failed: {e}"
            )

            if attempt < attempts:

                print(
                    "Waiting 5 seconds before retry..."
                )

                time.sleep(5)

    raise last_error


# ============================================================
# SAVE TEXT
# ============================================================

def save_text(
    filename,
    text
):

    path = (
        OUT_DIR
        / filename
    )

    path.write_text(
        text,
        encoding="utf-8"
    )

    print(
        f"Saved: {path}"
    )


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print(
    "STEP 64D-A — BHUVAN CATEGORICAL DIAGNOSTIC"
)
print("=" * 70)

print()
print(
    f"Output directory:\n{OUT_DIR}"
)


# ============================================================
# 1. GETCAPABILITIES
# ============================================================

print()
print(
    "[1] Downloading WMS GetCapabilities"
)

print(
    "-" * 70
)

cap_url = (
    WMS_URL
    + "?SERVICE=WMS"
    + "&VERSION=1.1.1"
    + "&REQUEST=GetCapabilities"
)

try:

    response = request_with_retry(
        cap_url,
        timeout=(30, 180),
        attempts=3
    )

    capabilities = response.text

    save_text(
        "bhuvan_wms_getcapabilities.xml",
        capabilities
    )

except Exception as e:

    print()
    print(
        "WARNING: GetCapabilities request failed."
    )

    print(
        f"Reason: {e}"
    )

    print()
    print(
        "The diagnostic will continue using the "
        "known layer names."
    )

    capabilities = ""


# ============================================================
# 2. CHECK KNOWN LAYERS
# ============================================================

print()
print(
    "[2] Checking known Bhuvan layers"
)

print(
    "-" * 70
)


print()
print("GEOMORPHOLOGY")
print("-" * 70)

for state, layer in GEOMORPH_LAYERS.items():

    if capabilities:

        found = (
            layer in capabilities
        )

        status = (
            "FOUND"
            if found
            else "NOT FOUND"
        )

    else:

        status = (
            "NOT CHECKED "
            "(GetCapabilities unavailable)"
        )

    print(
        f"{state:3s} | "
        f"{status:35s} | "
        f"{layer}"
    )


print()
print("LUCC")
print("-" * 70)

for state, layer in LUCC_LAYERS.items():

    if capabilities:

        found = (
            layer in capabilities
        )

        status = (
            "FOUND"
            if found
            else "NOT FOUND"
        )

    else:

        status = (
            "NOT CHECKED "
            "(GetCapabilities unavailable)"
        )

    print(
        f"{state:3s} | "
        f"{status:35s} | "
        f"{layer}"
    )


# ============================================================
# 3. GET LEGEND GRAPHICS
# ============================================================

print()
print(
    "[3] Requesting legend graphics"
)

print(
    "-" * 70
)


def request_legend(
    state,
    layer,
    factor
):

    print()
    print(
        f"{state} — {factor}"
    )

    url = (
        WMS_URL
        + "?SERVICE=WMS"
        + "&VERSION=1.1.1"
        + "&REQUEST=GetLegendGraphic"
        + "&FORMAT=image/png"
        + "&LAYER="
        + layer
    )

    try:

        response = request_with_retry(
            url,
            timeout=(30, 120),
            attempts=2
        )

        path = (
            OUT_DIR
            / f"{state}_{factor}_legend.png"
        )

        path.write_bytes(
            response.content
        )

        print(
            f"Saved: {path}"
        )

        print(
            f"Bytes: {len(response.content):,}"
        )

        return True

    except Exception as e:

        print(
            f"Legend failed: {e}"
        )

        return False


# Geomorphology legends

for state, layer in GEOMORPH_LAYERS.items():

    request_legend(
        state,
        layer,
        "geomorphology"
    )


# LUCC legends

for state, layer in LUCC_LAYERS.items():

    request_legend(
        state,
        layer,
        "lulc"
    )


# ============================================================
# 4. GETMAP DIAGNOSTIC IMAGES
# ============================================================

print()
print(
    "[4] Requesting small diagnostic GetMap images"
)

print(
    "-" * 70
)


def request_getmap(
    state,
    layer,
    factor
):

    bbox = (
        TEST_BBOXES[state]
    )

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
        + bbox
        + "&WIDTH=1000"
        + "&HEIGHT=1000"
        + "&FORMAT=image/png"
        + "&TRANSPARENT=true"
    )

    print()
    print(
        f"{state} — {factor}"
    )

    try:

        response = request_with_retry(
            url,
            timeout=(30, 180),
            attempts=2
        )

        path = (
            OUT_DIR
            / f"{state}_{factor}_getmap.png"
        )

        path.write_bytes(
            response.content
        )

        print(
            f"Saved: {path}"
        )

        print(
            f"Bytes: {len(response.content):,}"
        )

        return True

    except Exception as e:

        print(
            f"GetMap failed: {e}"
        )

        return False


# Geomorphology GetMap

for state, layer in GEOMORPH_LAYERS.items():

    request_getmap(
        state,
        layer,
        "geomorphology"
    )


# LUCC GetMap

for state, layer in LUCC_LAYERS.items():

    request_getmap(
        state,
        layer,
        "lulc"
    )


# ============================================================
# 5. SUMMARY
# ============================================================

print()
print("=" * 70)
print(
    "STEP 64D-A COMPLETED"
)
print("=" * 70)

print()
print(
    "Diagnostic files are located at:"
)

print(
    OUT_DIR
)

print()
print(
    "IMPORTANT:"
)

print(
    "Do NOT create the final categorical rasters yet."
)

print(
    "The legend and GetMap outputs must be inspected "
    "before assigning categorical class codes."
)

print()
print(
    "Next step:"
)

print(
    "Use the diagnostic outputs to establish the "
    "Bhuvan class/color mapping and validate it "
    "against the 1,071 training points."
)