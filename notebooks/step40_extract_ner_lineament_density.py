# ============================================================
# STEP 40 — NER LINEAMENT DENSITY
# ============================================================
#
# Source:
#   Bhuvan / NRSC NGLM Lineament 1:50,000 (2005–06)
#
# Method:
#   1. Read NER training points
#   2. Assign each point to its NER state
#   3. Normalize state names to handle Unicode accents
#   4. Request the corresponding Bhuvan Lineament WMS
#   5. Use the alpha channel to identify rendered lineament
#      pixels
#   6. Reproject the raster to EPSG:3857
#   7. Calculate local lineament-density index
#   8. Extract density values at the 1,071 training points
#   9. Save the final CSV
#
# IMPORTANT:
#   Bhuvan WFS is disabled.
#   Therefore this is a RASTER-DERIVED lineament density
#   from the official Bhuvan WMS representation.
#
# Output:
#   processed/ner_lineament_density.csv
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import io
import time
import unicodedata
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import geopandas as gpd

from PIL import Image

import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import (
    calculate_default_transform,
    reproject
)
from rasterio.enums import Resampling

from scipy.ndimage import uniform_filter


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POINTS_FILE = (
    PROJECT_ROOT
    / "processed"
    / "ner_training_points.geojson"
)

BOUNDARY_FILE = (
    PROJECT_ROOT
    / "raw_data"
    / "boundaries"
    / "geoBoundaries-IND-ADM1.geojson"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "raw_data"
    / "lineaments"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "ner_lineament_density.csv"
)


# ============================================================
# 3. BHUVAN WMS
# ============================================================

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)


# ============================================================
# 4. STATE → BHUVAN LINEAMENT LAYER
# ============================================================

LINEAMENT_LAYERS = {

    "ARUNACHAL PRADESH":
        "lineament:AR_LN50K_0506",

    "ASSAM":
        "lineament:AS_LN50K_0506",

    "MANIPUR":
        "lineament:MN_LN50K_0506",

    "MEGHALAYA":
        "lineament:ML_LN50K_0506",

    "MIZORAM":
        "lineament:MZ_LN50K_0506",

    "NAGALAND":
        "lineament:NL_LN50K_0506",

    "SIKKIM":
        "lineament:SK_LN50K_0506",

    "TRIPURA":
        "lineament:TR_LN50K_0506",
}


# ============================================================
# 5. RASTER / DENSITY SETTINGS
# ============================================================

# WMS output dimensions
WMS_WIDTH = 2000
WMS_HEIGHT = 2000

# Target working pixel size
PIXEL_SIZE_M = 100.0

# Local density radius
WINDOW_RADIUS_M = 1000.0

# Moving-window dimension
WINDOW_PIXELS = int(
    (2 * WINDOW_RADIUS_M)
    / PIXEL_SIZE_M
) + 1

# Make sure window is odd
if WINDOW_PIXELS % 2 == 0:
    WINDOW_PIXELS += 1


# ============================================================
# 6. NORMALIZE STATE NAMES
# ============================================================

def normalize_state_name(name):
    """
    Convert state names into a consistent form.

    Example:

        Arunāchal Pradesh
        →
        ARUNACHAL PRADESH

        Nāgāland
        →
        NAGALAND

        Meghālaya
        →
        MEGHALAYA
    """

    if pd.isna(name):
        return ""

    name = str(name)

    # Decompose accented Unicode characters
    name = unicodedata.normalize(
        "NFKD",
        name
    )

    # Remove combining accent marks
    name = "".join(
        char
        for char in name
        if not unicodedata.combining(char)
    )

    # Uppercase and remove extra spaces
    return name.upper().strip()


# ============================================================
# 7. DOWNLOAD BHUVAN WMS RASTER
# ============================================================

def download_wms_raster(
    layer_name,
    bbox,
    width,
    height
):
    """
    Download a transparent RGBA PNG from Bhuvan WMS.
    """

    minx, miny, maxx, maxy = bbox

    params = {

        "service": "WMS",

        "version": "1.1.1",

        "request": "GetMap",

        "layers": layer_name,

        "styles": "",

        "srs": "EPSG:4326",

        "bbox": (
            f"{minx},{miny},"
            f"{maxx},{maxy}"
        ),

        "width": width,

        "height": height,

        "format": "image/png",

        "transparent": "true",
    }

    response = requests.get(
        WMS_URL,
        params=params,
        timeout=180
    )

    if response.status_code != 200:

        raise RuntimeError(
            "Bhuvan WMS request failed. "
            f"HTTP status: {response.status_code}"
        )

    content_type = (
        response
        .headers
        .get(
            "Content-Type",
            ""
        )
        .lower()
    )

    if "image/png" not in content_type:

        raise RuntimeError(
            "Bhuvan did not return PNG. "
            f"Content-Type: {content_type}"
        )

    image = Image.open(
        io.BytesIO(
            response.content
        )
    ).convert("RGBA")

    return np.array(
        image
    )


# ============================================================
# 8. HEADER
# ============================================================

print("=" * 70)
print("STEP 40 — NER LINEAMENT DENSITY")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nTraining points:")
print(POINTS_FILE)

print("\nNER boundary:")
print(BOUNDARY_FILE)

print("\nBhuvan WMS:")
print(WMS_URL)

print("\nTarget pixel size:")
print(
    PIXEL_SIZE_M,
    "metres"
)

print("\nDensity window radius:")
print(
    WINDOW_RADIUS_M,
    "metres"
)

print("\nMoving window:")
print(
    WINDOW_PIXELS,
    "x",
    WINDOW_PIXELS,
    "pixels"
)


# ============================================================
# 9. CHECK INPUT FILES
# ============================================================

print("\n" + "-" * 70)
print("1. CHECKING INPUT FILES")
print("-" * 70)

if not POINTS_FILE.exists():

    raise FileNotFoundError(
        f"Training points not found:\n{POINTS_FILE}"
    )

if not BOUNDARY_FILE.exists():

    raise FileNotFoundError(
        f"NER boundary not found:\n{BOUNDARY_FILE}"
    )

print("Training points file found.")

print("NER boundary file found.")


# ============================================================
# 10. READ TRAINING POINTS
# ============================================================

print("\nReading training points...")

points = gpd.read_file(
    POINTS_FILE
)

print(
    "Training points:",
    len(points)
)

print(
    "CRS:",
    points.crs
)

if points.empty:

    raise ValueError(
        "Training points file is empty."
    )


# ============================================================
# 11. READ NER BOUNDARY
# ============================================================

print("\nReading NER boundary...")

boundary = gpd.read_file(
    BOUNDARY_FILE
)

print(
    "Boundary features:",
    len(boundary)
)

print(
    "Boundary CRS:",
    boundary.crs
)


# ============================================================
# 12. REPROJECT TO EPSG:4326
# ============================================================

points_4326 = points.to_crs(
    "EPSG:4326"
)

boundary_4326 = boundary.to_crs(
    "EPSG:4326"
)


# ============================================================
# 13. ASSIGN STATE TO EACH TRAINING POINT
# ============================================================

print("\n" + "-" * 70)
print("2. ASSIGNING STATES TO TRAINING POINTS")
print("-" * 70)

state_join = gpd.sjoin(

    points_4326,

    boundary_4326[
        [
            "shapeName",
            "geometry"
        ]
    ],

    how="left",

    predicate="within"
)


# ------------------------------------------------------------
# Remove duplicate spatial-join rows
# ------------------------------------------------------------

state_join = state_join[
    ~state_join.index.duplicated(
        keep="first"
    )
].copy()


# ------------------------------------------------------------
# Attach original state name
# ------------------------------------------------------------

points_4326["state"] = (

    state_join[
        "shapeName"
    ]
    .reindex(
        points_4326.index
    )
    .values
)


# ------------------------------------------------------------
# Normalize Unicode state names
# ------------------------------------------------------------

points_4326[
    "state_normalized"
] = (

    points_4326[
        "state"
    ]
    .apply(
        normalize_state_name
    )
)


# ============================================================
# 14. PRINT STATE DISTRIBUTION
# ============================================================

print("\nOriginal state names:")

print(
    points_4326[
        "state"
    ]
    .value_counts(
        dropna=False
    )
)


print("\nNormalized state names:")

print(
    points_4326[
        "state_normalized"
    ]
    .value_counts(
        dropna=False
    )
)


# ============================================================
# 15. CHECK BHUVAN WMS CAPABILITIES
# ============================================================

print("\n" + "-" * 70)
print("3. CHECKING BHUVAN LINEAMENT LAYERS")
print("-" * 70)

capabilities_params = {

    "service": "WMS",

    "request": "GetCapabilities",

    "version": "1.1.1",
}


cap_response = requests.get(

    WMS_URL,

    params=capabilities_params,

    timeout=120
)


print(
    "GetCapabilities HTTP status:",
    cap_response.status_code
)


if cap_response.status_code != 200:

    raise RuntimeError(
        "Could not retrieve Bhuvan WMS "
        "GetCapabilities."
    )


capabilities_text = (
    cap_response.text
)


# ------------------------------------------------------------
# Check each layer
# ------------------------------------------------------------

available_layers = {}


for state, layer in LINEAMENT_LAYERS.items():

    available = (
        layer
        in capabilities_text
    )

    available_layers[
        state
    ] = available

    if available:

        print(
            "AVAILABLE:",
            state,
            "→",
            layer
        )

    else:

        print(
            "UNAVAILABLE:",
            state,
            "→",
            layer
        )


# ============================================================
# 16. STATE-BY-STATE PROCESSING
# ============================================================

print("\n" + "-" * 70)
print("4. DOWNLOADING AND PROCESSING LINEAMENT RASTERS")
print("-" * 70)


state_rasters = {}


for state, layer in LINEAMENT_LAYERS.items():

    print("\n" + "=" * 60)

    print(
        "STATE:",
        state
    )

    print(
        "LAYER:",
        layer
    )

    print("=" * 60)


    # --------------------------------------------------------
    # Check layer availability
    # --------------------------------------------------------

    if not available_layers[state]:

        print(
            "Skipping — Bhuvan layer unavailable."
        )

        continue


    # --------------------------------------------------------
    # Get points in this state
    # --------------------------------------------------------

    mask = (
        points_4326[
            "state_normalized"
        ]
        == state
    )

    state_points = (
        points_4326[
            mask
        ]
    )


    print(
        "Training points in state:",
        len(state_points)
    )


    if len(state_points) == 0:

        print(
            "No training points in this state."
        )

        continue


    # --------------------------------------------------------
    # Calculate bounding box
    # --------------------------------------------------------

    minx, miny, maxx, maxy = (
        state_points
        .total_bounds
    )


    # Small geographic buffer
    buffer_deg = 0.05

    minx -= buffer_deg
    maxx += buffer_deg

    miny -= buffer_deg
    maxy += buffer_deg


    print(
        "\nWMS bounding box:"
    )

    print(
        "minx:",
        minx
    )

    print(
        "miny:",
        miny
    )

    print(
        "maxx:",
        maxx
    )

    print(
        "maxy:",
        maxy
    )


    # --------------------------------------------------------
    # Download WMS
    # --------------------------------------------------------

    print(
        "\nDownloading WMS raster..."
    )

    try:

        rgba = download_wms_raster(

            layer_name=layer,

            bbox=(
                minx,
                miny,
                maxx,
                maxy
            ),

            width=WMS_WIDTH,

            height=WMS_HEIGHT
        )

    except Exception as e:

        print(
            "\nERROR downloading layer:"
        )

        print(e)

        continue


    print(
        "Downloaded raster shape:",
        rgba.shape
    )


    # --------------------------------------------------------
    # Save WMS PNG
    # --------------------------------------------------------

    safe_state = (
        state
        .lower()
        .replace(
            " ",
            "_"
        )
    )

    png_file = (

        OUTPUT_DIR
        / f"{safe_state}_lineament_wms.png"
    )


    Image.fromarray(
        rgba
    ).save(
        png_file
    )


    print(
        "WMS PNG saved:"
    )

    print(
        png_file
    )


    # --------------------------------------------------------
    # Extract alpha channel
    # --------------------------------------------------------

    alpha = (
        rgba[
            :,
            :,
            3
        ]
    )


    lineament_mask = (
        alpha > 0
    )


    transparent_pixels = (
        np.sum(
            alpha == 0
        )
    )

    nontransparent_pixels = (
        np.sum(
            alpha > 0
        )
    )


    print(
        "\nTransparent pixels:",
        transparent_pixels
    )

    print(
        "Non-transparent pixels:",
        nontransparent_pixels
    )


    percentage = (
        100.0
        * np.mean(
            lineament_mask
        )
    )


    print(
        "Non-transparent percentage:",
        percentage
    )


    # --------------------------------------------------------
    # Create geographic transform
    # --------------------------------------------------------

    transform = from_bounds(

        minx,

        miny,

        maxx,

        maxy,

        WMS_WIDTH,

        WMS_HEIGHT
    )


    # --------------------------------------------------------
    # Convert mask to uint8
    # --------------------------------------------------------

    source = (
        lineament_mask
        .astype(
            np.uint8
        )
    )


    # --------------------------------------------------------
    # Reproject to EPSG:3857
    # --------------------------------------------------------

    print(
        "\nReprojecting lineament mask..."
    )


    (
        destination_transform,
        dest_width,
        dest_height
    ) = calculate_default_transform(

        "EPSG:4326",

        "EPSG:3857",

        WMS_WIDTH,

        WMS_HEIGHT,

        minx,

        miny,

        maxx,

        maxy
    )


    destination = np.zeros(

        (
            dest_height,
            dest_width
        ),

        dtype=np.uint8
    )


    reproject(

        source,

        destination,

        src_transform=transform,

        src_crs="EPSG:4326",

        dst_transform=destination_transform,

        dst_crs="EPSG:3857",

        resampling=Resampling.nearest
    )


    print(
        "Reprojected raster:",
        dest_width,
        "x",
        dest_height
    )


    # --------------------------------------------------------
    # Calculate local density
    # --------------------------------------------------------

    print(
        "\nCalculating local lineament density..."
    )


    density = uniform_filter(

        destination.astype(
            np.float32
        ),

        size=WINDOW_PIXELS,

        mode="constant",

        cval=0
    )


    # --------------------------------------------------------
    # Save density raster
    # --------------------------------------------------------

    density_file = (

        OUTPUT_DIR
        / f"{safe_state}_lineament_density.tif"
    )


    with rasterio.open(

        density_file,

        "w",

        driver="GTiff",

        height=dest_height,

        width=dest_width,

        count=1,

        dtype="float32",

        crs="EPSG:3857",

        transform=destination_transform,

        nodata=-9999.0

    ) as dst:

        dst.write(

            density.astype(
                np.float32
            ),

            1
        )


    print(
        "\nDensity raster saved:"
    )

    print(
        density_file
    )


    # --------------------------------------------------------
    # Store raster information
    # --------------------------------------------------------

    state_rasters[
        state
    ] = {

        "file":
            density_file,

        "transform":
            destination_transform,

        "width":
            dest_width,

        "height":
            dest_height,
    }


    time.sleep(1)


# ============================================================
# 17. EXTRACT DENSITY AT TRAINING POINTS
# ============================================================

print("\n" + "-" * 70)
print("5. EXTRACTING DENSITY AT TRAINING POINTS")
print("-" * 70)


density_values = np.full(

    len(points_4326),

    np.nan,

    dtype=float
)


for state, info in state_rasters.items():

    print(
        "\nProcessing:",
        state
    )


    # --------------------------------------------------------
    # Select points in this state
    # --------------------------------------------------------

    mask = (

        points_4326[
            "state_normalized"
        ]

        == state
    )


    indices = (

        points_4326
        .index[
            mask
        ]
        .tolist()
    )


    if len(indices) == 0:

        continue


    # --------------------------------------------------------
    # Open density raster
    # --------------------------------------------------------

    with rasterio.open(

        info["file"]

    ) as src:

        state_points = (

            points_4326
            .loc[
                indices
            ]
            .to_crs(
                "EPSG:3857"
            )
        )


        coordinates = [

            (
                geometry.x,
                geometry.y
            )

            for geometry
            in state_points.geometry
        ]


        values = list(

            src.sample(
                coordinates
            )
        )


    # --------------------------------------------------------
    # Store values
    # --------------------------------------------------------

    for idx, value in zip(

        indices,

        values

    ):

        position = (
            points_4326
            .index
            .get_loc(idx)
        )


        density_values[
            position
        ] = float(
            value[0]
        )


    print(
        "Points processed:",
        len(indices)
    )


# ============================================================
# 18. CREATE FINAL OUTPUT DATAFRAME
# ============================================================

print("\n" + "-" * 70)
print("6. CREATING FINAL OUTPUT")
print("-" * 70)


output = pd.DataFrame({

    "lat":
        points_4326
        .geometry
        .y
        .values,

    "lon":
        points_4326
        .geometry
        .x
        .values,
})


# ------------------------------------------------------------
# Preserve original metadata
# ------------------------------------------------------------

for column in [

    "date",

    "source",

    "region",

    "label",

]:

    if column in points.columns:

        output[column] = (
            points[column]
            .values
        )


# ------------------------------------------------------------
# Add lineament density
# ------------------------------------------------------------

output[
    "lineament_density"
] = density_values


# ============================================================
# 19. QUALITY CHECK
# ============================================================

print("\n" + "-" * 70)
print("7. LINEAMENT DENSITY QUALITY CHECK")
print("-" * 70)


valid = (

    output[
        "lineament_density"
    ]
    .notna()
    .sum()
)


missing = (

    output[
        "lineament_density"
    ]
    .isna()
    .sum()
)


print(
    "Total points:",
    len(output)
)

print(
    "Valid:",
    valid
)

print(
    "Missing:",
    missing
)


# ============================================================
# 20. DENSITY STATISTICS
# ============================================================

if valid > 0:

    values = (

        output[
            "lineament_density"
        ]
        .dropna()
    )


    print(
        "\nDensity statistics:"
    )


    print(
        "Minimum:",
        values.min()
    )


    print(
        "Maximum:",
        values.max()
    )


    print(
        "Mean:",
        values.mean()
    )


    print(
        "Median:",
        values.median()
    )


    print(
        "Standard deviation:",
        values.std()
    )


# ============================================================
# 21. SAVE CSV
# ============================================================

print("\n" + "-" * 70)
print("8. SAVING FINAL CSV")
print("-" * 70)


output.to_csv(

    OUTPUT_FILE,

    index=False
)


print(
    "Output saved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 22. VERIFY SAVED FILE
# ============================================================

print("\n" + "-" * 70)
print("9. VERIFYING SAVED CSV")
print("-" * 70)


if not OUTPUT_FILE.exists():

    raise RuntimeError(
        "Output CSV was not created."
    )


check = pd.read_csv(
    OUTPUT_FILE
)


print(
    "Rows:",
    len(check)
)


print(
    "Columns:",
    list(check.columns)
)


print(
    "\nFirst 5 rows:"
)


print(
    check.head()
)


# ============================================================
# 23. FINAL SANITY CHECK
# ============================================================

if len(check) != len(points):

    raise ValueError(

        "Final output row count does not "
        "match training-point count."
    )


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("STEP 40 COMPLETED")
print("=" * 70)

print(
    "\nTraining points:",
    len(points)
)

print(
    "Valid density values:",
    valid
)

print(
    "Missing density values:",
    missing
)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nMethodological note:"
)

print(
    "Lineament density was derived from the official "
    "Bhuvan NGLM 1:50,000 lineament WMS representation "
    "because the Bhuvan WFS service was disabled."
)

print("=" * 70)