"""
STEP 64D-A — BHUVAN CATEGORICAL LAYER DIAGNOSTIC

Purpose
-------
Inspect Bhuvan WMS categorical layers before creating
full-resolution categorical rasters.

Layers:
1. Geomorphology 1:50,000
2. LUCC SIS-DP Phase 2

This script does NOT modify existing project data.
"""

from pathlib import Path
import requests
import xml.etree.ElementTree as ET


# ============================================================
# PROJECT
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
# LAYERS
# ============================================================

GEOMORPH_LAYERS = {
    "AR": "geomorphology:AR_GM50K_0506",
    "AS": "geomorphology:AS_GM50K_0506",
    "MN": "geomorphology:MN_GM50K_0506",
    "ML": "geomorphology:ML_GM50K_0506",
    "MZ": "geomorphology:MZ_GM50K_0506",
    "NL": "geomorphology:NL_GM50K_0506",
    "SK": "geomorphology:SK_GM50K_0506",
    "TR": "geomorphology:TR_GM50K_0506",
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
# HELPER
# ============================================================

def get_text(url):

    response = requests.get(
        url,
        timeout=60
    )

    print(
        f"HTTP {response.status_code}: {url}"
    )

    response.raise_for_status()

    return response.text


def save_text(
    filename,
    text
):

    path = OUT_DIR / filename

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

print("=" * 70)
print("STEP 64D-A — BHUVAN CATEGORICAL DIAGNOSTIC")
print("=" * 70)


# ============================================================
# 1. GETCAPABILITIES
# ============================================================

print("\n[1] Downloading WMS GetCapabilities")
print("-" * 70)

cap_url = (
    WMS_URL
    + "?SERVICE=WMS"
    + "&VERSION=1.1.1"
    + "&REQUEST=GetCapabilities"
)

capabilities = get_text(
    cap_url
)

save_text(
    "bhuvan_wms_getcapabilities.xml",
    capabilities
)


# ============================================================
# 2. PARSE REQUESTED LAYERS
# ============================================================

print("\n[2] Checking requested layers")
print("-" * 70)

all_layers = (
    list(GEOMORPH_LAYERS.items())
    + list(LUCC_LAYERS.items())
)

for state, layer in all_layers:

    found = (
        layer in capabilities
    )

    status = (
        "FOUND"
        if found
        else "NOT FOUND"
    )

    print(
        f"{state:3s} | "
        f"{status:10s} | "
        f"{layer}"
    )


# ============================================================
# 3. GET LEGEND GRAPHIC
# ============================================================

print("\n[3] Requesting legend information")
print("-" * 70)

for state, layer in GEOMORPH_LAYERS.items():

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

        response = requests.get(
            url,
            timeout=60
        )

        print(
            f"{state} geomorphology: "
            f"HTTP {response.status_code}, "
            f"{len(response.content):,} bytes"
        )

        if response.ok:

            path = (
                OUT_DIR
                / f"{state}_geomorphology_legend.png"
            )

            path.write_bytes(
                response.content
            )

            print(
                f"  Saved: {path}"
            )

    except Exception as e:

        print(
            f"  ERROR: {e}"
        )


for state, layer in LUCC_LAYERS.items():

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

        response = requests.get(
            url,
            timeout=60
        )

        print(
            f"{state} LUCC: "
            f"HTTP {response.status_code}, "
            f"{len(response.content):,} bytes"
        )

        if response.ok:

            path = (
                OUT_DIR
                / f"{state}_lulc_legend.png"
            )

            path.write_bytes(
                response.content
            )

            print(
                f"  Saved: {path}"
            )

    except Exception as e:

        print(
            f"  ERROR: {e}"
        )


# ============================================================
# 4. GETMAP TEST IMAGES
# ============================================================

print("\n[4] Creating small GetMap diagnostic images")
print("-" * 70)

# Approximate NER state/test bounding boxes.
# These are ONLY diagnostic requests.

TEST_BBOXES = {
    "AR": "91.5,26.5,94.5,28.5",
    "AS": "89.5,25.0,94.0,28.0",
    "MN": "93.0,23.8,94.5,25.8",
    "ML": "89.5,25.0,92.0,26.5",
    "MZ": "92.0,21.8,93.5,24.5",
    "NL": "93.3,25.0,95.5,27.0",
    "SK": "88.0,26.8,88.9,28.2",
    "TR": "91.0,22.8,92.5,24.0",
}


def request_getmap(
    state,
    layer,
    suffix
):

    bbox = TEST_BBOXES[state]

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

    try:

        response = requests.get(
            url,
            timeout=90
        )

        print(
            f"{state} {suffix}: "
            f"HTTP {response.status_code}, "
            f"{len(response.content):,} bytes"
        )

        if response.ok:

            path = (
                OUT_DIR
                / f"{state}_{suffix}_getmap.png"
            )

            path.write_bytes(
                response.content
            )

            print(
                f"  Saved: {path}"
            )

    except Exception as e:

        print(
            f"{state} {suffix}: ERROR {e}"
        )


for state, layer in GEOMORPH_LAYERS.items():

    request_getmap(
        state,
        layer,
        "geomorphology"
    )


for state, layer in LUCC_LAYERS.items():

    request_getmap(
        state,
        layer,
        "lulc"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 64D-A COMPLETED")
print("=" * 70)

print("\nOutput directory:")
print(OUT_DIR)

print("\nImportant:")
print(
    "Do NOT create categorical rasters yet."
)

print(
    "The legends/GetMap outputs must be inspected "
    "before assigning numeric class codes."
)