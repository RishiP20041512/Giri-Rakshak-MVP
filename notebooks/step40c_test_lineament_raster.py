# ============================================================
# STEP 40C — TEST BHUVAN LINEAMENT RASTER EXTRACTION
# ============================================================
#
# Purpose:
#   Download a georeferenced raster from the official Bhuvan
#   Lineament WMS and inspect the pixel structure.
#
#   This is a TEST ONLY.
#
#   We will NOT calculate the final lineament density until
#   the raster representation has been verified.
#
# ============================================================

import requests
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "raw_data"
    / "lineaments"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. BHUVAN WMS
# ------------------------------------------------------------

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)


# ------------------------------------------------------------
# 3. TEST LAYER
# ------------------------------------------------------------

TEST_LAYER = (
    "lineament:AR_LN50K_0506"
)


# ------------------------------------------------------------
# 4. TEST EXTENT
# ------------------------------------------------------------
#
# Small area around our previous test point.
#
# ------------------------------------------------------------

CENTER_LON = 92.63415
CENTER_LAT = 27.45

HALF_SIZE = 0.10

MINX = CENTER_LON - HALF_SIZE
MAXX = CENTER_LON + HALF_SIZE

MINY = CENTER_LAT - HALF_SIZE
MAXY = CENTER_LAT + HALF_SIZE


# ------------------------------------------------------------
# 5. RASTER SIZE
# ------------------------------------------------------------
#
# 1000 x 1000 gives us enough detail for this test while
# keeping the response manageable.
#
# ------------------------------------------------------------

WIDTH = 1000
HEIGHT = 1000


# ------------------------------------------------------------
# 6. OUTPUT FILES
# ------------------------------------------------------------

PNG_FILE = (
    OUTPUT_DIR
    / "test_arunachal_lineament_1000.png"
)

TIF_FILE = (
    OUTPUT_DIR
    / "test_arunachal_lineament_1000.tif"
)


# ------------------------------------------------------------
# 7. HEADER
# ------------------------------------------------------------

print("=" * 70)
print("STEP 40C — BHUVAN LINEAMENT RASTER TEST")
print("=" * 70)

print("\nLayer:")
print(TEST_LAYER)

print("\nBounding box:")
print(
    f"MINX = {MINX}"
)
print(
    f"MINY = {MINY}"
)
print(
    f"MAXX = {MAXX}"
)
print(
    f"MAXY = {MAXY}"
)

print("\nRaster size:")
print(
    WIDTH,
    "x",
    HEIGHT
)


# ------------------------------------------------------------
# 8. GETMAP
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("1. REQUESTING BHUVAN WMS")
print("-" * 70)

params = {
    "service": "WMS",
    "version": "1.1.1",
    "request": "GetMap",
    "layers": TEST_LAYER,
    "styles": "",
    "srs": "EPSG:4326",
    "bbox": (
        f"{MINX},{MINY},{MAXX},{MAXY}"
    ),
    "width": WIDTH,
    "height": HEIGHT,
    "format": "image/png",
    "transparent": "true",
}


response = requests.get(
    WMS_URL,
    params=params,
    timeout=120
)


print(
    "HTTP status:",
    response.status_code
)

print(
    "Content type:",
    response.headers.get("Content-Type")
)

print(
    "Response size:",
    len(response.content),
    "bytes"
)


if response.status_code != 200:

    print("\nServer response:")
    print(response.text[:5000])

    raise RuntimeError(
        "Bhuvan WMS GetMap request failed."
    )


# ------------------------------------------------------------
# 9. SAVE PNG
# ------------------------------------------------------------

with open(
    PNG_FILE,
    "wb"
) as f:

    f.write(response.content)


print("\nPNG saved:")
print(PNG_FILE)


# ------------------------------------------------------------
# 10. OPEN PNG
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("2. INSPECTING PNG")
print("-" * 70)

image = Image.open(
    PNG_FILE
)

print(
    "Image mode:",
    image.mode
)

print(
    "Image size:",
    image.size
)


# ------------------------------------------------------------
# 11. CONVERT TO NUMPY
# ------------------------------------------------------------

array = np.array(
    image
)

print(
    "Array shape:",
    array.shape
)

print(
    "Array dtype:",
    array.dtype
)


# ------------------------------------------------------------
# 12. PIXEL VALUE ANALYSIS
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("3. PIXEL VALUE ANALYSIS")
print("-" * 70)


if array.ndim == 2:

    unique_values, counts = np.unique(
        array,
        return_counts=True
    )

    print(
        "Number of unique pixel values:",
        len(unique_values)
    )

    print("\nMost common pixel values:")

    order = np.argsort(
        counts
    )[::-1]

    for i in order[:20]:

        print(
            "Value:",
            unique_values[i],
            "Count:",
            counts[i]
        )


elif array.ndim == 3:

    print(
        "Number of channels:",
        array.shape[2]
    )

    for channel in range(
        array.shape[2]
    ):

        print(
            "\nChannel:",
            channel
        )

        channel_data = array[:, :, channel]

        unique_values, counts = np.unique(
            channel_data,
            return_counts=True
        )

        order = np.argsort(
            counts
        )[::-1]

        print(
            "Unique values:",
            len(unique_values)
        )

        for i in order[:15]:

            print(
                "Value:",
                unique_values[i],
                "Count:",
                counts[i]
            )


# ------------------------------------------------------------
# 13. ALPHA CHANNEL ANALYSIS
# ------------------------------------------------------------

if array.ndim == 3 and array.shape[2] == 4:

    print("\n" + "-" * 70)
    print("4. ALPHA CHANNEL ANALYSIS")
    print("-" * 70)

    alpha = array[:, :, 3]

    transparent = np.sum(
        alpha == 0
    )

    nontransparent = np.sum(
        alpha > 0
    )

    print(
        "Transparent pixels:",
        transparent
    )

    print(
        "Non-transparent pixels:",
        nontransparent
    )

    print(
        "Total pixels:",
        alpha.size
    )

    print(
        "Non-transparent percentage:",
        100 * nontransparent / alpha.size
    )


# ------------------------------------------------------------
# 14. SAVE GEOTIFF
# ------------------------------------------------------------
#
# The PNG itself has no spatial reference.
# We therefore create a GeoTIFF using the known WMS extent.
#
# This is only a TEST raster.
#
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("5. CREATING TEST GEOTIFF")
print("-" * 70)


transform = from_bounds(
    MINX,
    MINY,
    MAXX,
    MAXY,
    WIDTH,
    HEIGHT
)


if array.ndim == 2:

    band_count = 1

elif array.ndim == 3:

    band_count = array.shape[2]

else:

    raise ValueError(
        "Unexpected image dimensions."
    )


with rasterio.open(
    TIF_FILE,
    "w",
    driver="GTiff",
    height=HEIGHT,
    width=WIDTH,
    count=band_count,
    dtype=array.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:

    if array.ndim == 2:

        dst.write(
            array,
            1
        )

    else:

        for band in range(
            band_count
        ):

            dst.write(
                array[:, :, band],
                band + 1
            )


print(
    "GeoTIFF saved:"
)

print(
    TIF_FILE
)


# ------------------------------------------------------------
# 15. REOPEN GEOTIFF
# ------------------------------------------------------------

print("\n" + "-" * 70)
print("6. VERIFYING GEOTIFF")
print("-" * 70)

with rasterio.open(
    TIF_FILE
) as src:

    print(
        "CRS:",
        src.crs
    )

    print(
        "Width:",
        src.width
    )

    print(
        "Height:",
        src.height
    )

    print(
        "Bands:",
        src.count
    )

    print(
        "Bounds:",
        src.bounds
    )

    print(
        "Resolution:",
        src.res
    )


# ------------------------------------------------------------
# 16. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 40C TEST COMPLETED")
print("=" * 70)

print("\nFiles created:")

print(
    PNG_FILE
)

print(
    TIF_FILE
)

print(
    "\nIMPORTANT:"
)

print(
    "This is only a test raster."
)

print(
    "Do NOT use the pixel values as lineament density yet."
)

print(
    "\nSend me the COMPLETE terminal output."
)

print("=" * 70)