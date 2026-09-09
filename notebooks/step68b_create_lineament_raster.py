# ============================================================
# STEP 68B — FINAL 250 m LINEAMENT DENSITY RASTER
# ============================================================
#
# Source:
#   Bhuvan / NRSC NGLM Lineament 1:50,000 (2005–06)
#
# Method:
#   1. Read official NER boundary
#   2. Check Bhuvan Lineament WMS availability
#   3. Download state-wise transparent WMS raster
#   4. Extract rendered lineament pixels
#   5. Reproject to EPSG:3857
#   6. Calculate local density using 1 km radius
#   7. Reproject density to final EPSG:6933 grid
#   8. Apply state boundary mask
#   9. Apply NER mask
#  10. Save final 250 m raster
#
# IMPORTANT:
#   Bhuvan WFS is disabled.
#   Therefore lineament density is derived from the
#   official Bhuvan WMS representation.
#
# Mizoram:
#   Official Bhuvan Mizoram lineament layer was unavailable.
#   Therefore Mizoram is preserved as NoData.
#
# Output:
#   processed/predictors/lineament_density_250m.tif
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
from rasterio.features import geometry_mask
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

GRID_FILE = (
    PROJECT_ROOT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

MASK_FILE = (
    PROJECT_ROOT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "processed"
    / "predictors"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "lineament_density_250m.tif"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "processed"
    / "step68b_lineament_raster_summary.csv"
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

    # Official layer unavailable
    "MIZORAM":
        None,

    "NAGALAND":
        "lineament:NL_LN50K_0506",

    "SIKKIM":
        "lineament:SK_LN50K_0506",

    "TRIPURA":
        "lineament:TR_LN50K_0506",
}


# ============================================================
# 5. WORKING RASTER SETTINGS
# ============================================================

WMS_WIDTH = 2000
WMS_HEIGHT = 2000

WORKING_RESOLUTION_M = 100.0

DENSITY_RADIUS_M = 1000.0

WINDOW_PIXELS = int(
    (2 * DENSITY_RADIUS_M)
    / WORKING_RESOLUTION_M
) + 1

if WINDOW_PIXELS % 2 == 0:
    WINDOW_PIXELS += 1


# ============================================================
# 6. RGB FILTER SETTINGS
# ============================================================

WHITE_THRESHOLD = 245
BLACK_THRESHOLD = 40


# ============================================================
# 7. NORMALIZE STATE NAMES
# ============================================================

def normalize_state_name(name):

    if pd.isna(name):
        return ""

    name = str(name)

    name = unicodedata.normalize(
        "NFKD",
        name
    )

    name = "".join(
        char
        for char in name
        if not unicodedata.combining(char)
    )

    return name.upper().strip()


# ============================================================
# 8. DOWNLOAD WMS RASTER
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

        "service":
            "WMS",

        "version":
            "1.1.1",

        "request":
            "GetMap",

        "layers":
            layer_name,

        "styles":
            "",

        "srs":
            "EPSG:4326",

        "bbox":
            f"{minx},{miny},{maxx},{maxy}",

        "width":
            width,

        "height":
            height,

        "format":
            "image/png",

        "transparent":
            "true",
    }

    last_error = None

    for attempt in range(1, 4):

        try:

            response = requests.get(
                WMS_URL,
                params=params,
                timeout=180
            )

            if response.status_code != 200:

                raise RuntimeError(
                    f"HTTP {response.status_code}"
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

        except Exception as e:

            last_error = e

            print(
                f"  Attempt {attempt}/3 failed: {e}"
            )

            if attempt < 3:
                time.sleep(3)

    raise RuntimeError(
        f"WMS download failed for {layer_name}: "
        f"{last_error}"
    )


# ============================================================
# 9. START
# ============================================================

print("=" * 70)
print("STEP 68B — FINAL 250 m LINEAMENT DENSITY RASTER")
print("=" * 70)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nBhuvan WMS:")
print(WMS_URL)

print("\nWorking resolution:")
print(
    WORKING_RESOLUTION_M,
    "m"
)

print("\nDensity radius:")
print(
    DENSITY_RADIUS_M,
    "m"
)

print("\nMoving window:")
print(
    WINDOW_PIXELS,
    "x",
    WINDOW_PIXELS
)


# ============================================================
# 10. CHECK INPUTS
# ============================================================

print("\n" + "-" * 70)
print("CHECKING INPUT FILES")
print("-" * 70)

for file_path in [
    POINTS_FILE,
    BOUNDARY_FILE,
    GRID_FILE,
    MASK_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    print(
        "FOUND:",
        file_path
    )


# ============================================================
# 11. LOAD FINAL GRID
# ============================================================

print("\n" + "-" * 70)
print("LOADING FINAL GRID")
print("-" * 70)

with rasterio.open(
    GRID_FILE
) as src:

    grid_profile = src.profile.copy()

    grid_transform = src.transform

    grid_crs = src.crs

    grid_width = src.width

    grid_height = src.height

    grid_bounds = src.bounds


print(
    "CRS:",
    grid_crs
)

print(
    "Size:",
    grid_width,
    "x",
    grid_height
)

print(
    "Resolution:",
    grid_transform.a,
    abs(grid_transform.e)
)

print(
    "Bounds:",
    grid_bounds
)


# ============================================================
# 12. CHECK FINAL GRID CRS
# ============================================================

if grid_crs.to_epsg() != 6933:

    raise ValueError(
        "Final grid is not EPSG:6933."
    )


if (
    abs(grid_transform.a - 250.0)
    > 0.01
):

    raise ValueError(
        "Final grid X resolution is not 250 m."
    )


if (
    abs(abs(grid_transform.e) - 250.0)
    > 0.01
):

    raise ValueError(
        "Final grid Y resolution is not 250 m."
    )


# ============================================================
# 13. LOAD NER MASK
# ============================================================

print("\n" + "-" * 70)
print("LOADING NER MASK")
print("-" * 70)

with rasterio.open(
    MASK_FILE
) as src:

    ner_mask = (
        src.read(1)
        > 0
    )

    mask_crs = src.crs

    mask_transform = src.transform

    mask_width = src.width

    mask_height = src.height


if mask_crs != grid_crs:

    raise ValueError(
        "NER mask CRS does not match final grid."
    )


if (
    mask_width != grid_width
    or mask_height != grid_height
):

    raise ValueError(
        "NER mask dimensions do not match final grid."
    )


print(
    "NER valid cells:",
    int(ner_mask.sum())
)


# ============================================================
# 14. LOAD TRAINING POINTS
# ============================================================

print("\n" + "-" * 70)
print("LOADING TRAINING POINTS")
print("-" * 70)

points = gpd.read_file(
    POINTS_FILE
)

points = points.to_crs(
    "EPSG:4326"
)

print(
    "Training points:",
    len(points)
)


# ============================================================
# 15. LOAD NER BOUNDARY
# ============================================================

print("\n" + "-" * 70)
print("LOADING NER BOUNDARY")
print("-" * 70)

boundary = gpd.read_file(
    BOUNDARY_FILE
)

boundary = boundary.to_crs(
    "EPSG:4326"
)

boundary["state_norm"] = (
    boundary["shapeName"]
    .apply(normalize_state_name)
)

print(
    "Boundary features:",
    len(boundary)
)


# ============================================================
# 16. CHECK BHUVAN CAPABILITIES
# ============================================================

print("\n" + "-" * 70)
print("CHECKING BHUVAN WMS CAPABILITIES")
print("-" * 70)

capabilities_response = requests.get(
    WMS_URL,
    params={
        "service":
            "WMS",

        "request":
            "GetCapabilities",

        "version":
            "1.1.1",
    },
    timeout=120
)

if capabilities_response.status_code != 200:

    raise RuntimeError(
        "Could not retrieve Bhuvan WMS capabilities."
    )

capabilities_text = (
    capabilities_response.text
)

available_layers = {}

for state, layer in LINEAMENT_LAYERS.items():

    if layer is None:

        available_layers[state] = False

        print(
            "UNAVAILABLE:",
            state,
            "→ intentionally skipped"
        )

    else:

        available = (
            layer
            in capabilities_text
        )

        available_layers[state] = available

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
# 17. CREATE EMPTY FINAL ARRAY
# ============================================================

final_density = np.full(
    (
        grid_height,
        grid_width
    ),
    np.nan,
    dtype=np.float32
)


# ============================================================
# 18. PROCESS EACH STATE
# ============================================================

for state, layer in LINEAMENT_LAYERS.items():

    print("\n" + "=" * 70)
    print("STATE:", state)
    print("=" * 70)


    # --------------------------------------------------------
    # Skip unavailable layer
    # --------------------------------------------------------

    if not available_layers[state]:

        print(
            "Skipping — lineament layer unavailable."
        )

        continue


    # --------------------------------------------------------
    # Get state geometry
    # --------------------------------------------------------

    state_geom = boundary[
        boundary["state_norm"] == state
    ].copy()

    if state_geom.empty:

        print(
            "WARNING: State boundary not found."
        )

        continue


    # --------------------------------------------------------
    # State bbox
    # --------------------------------------------------------

    minx, miny, maxx, maxy = (
        state_geom.total_bounds
    )

    buffer_deg = 0.05

    minx -= buffer_deg
    miny -= buffer_deg
    maxx += buffer_deg
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


    print(
        "Downloaded raster shape:",
        rgba.shape
    )


    # --------------------------------------------------------
    # Extract alpha
    # --------------------------------------------------------

    alpha = rgba[
        :,
        :,
        3
    ]

    rendered = (
        alpha > 0
    )


    # --------------------------------------------------------
    # Remove obvious background
    # --------------------------------------------------------

    rgb = rgba[
        :,
        :,
        :3
    ]

    white = np.all(
        rgb >= WHITE_THRESHOLD,
        axis=2
    )

    black = np.all(
        rgb <= BLACK_THRESHOLD,
        axis=2
    )


    # Keep only rendered pixels that are
    # neither obvious white background nor black
    rendered = (
        rendered
        & ~white
        & ~black
    )


    print(
        "Rendered lineament pixels:",
        int(rendered.sum())
    )


    if rendered.sum() == 0:

        print(
            "WARNING: No usable lineament pixels."
        )

        continue


    # ========================================================
    # 19. GEOGRAPHIC TRANSFORM
    # ========================================================

    src_transform = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        WMS_WIDTH,
        WMS_HEIGHT
    )


    # ========================================================
    # 20. CALCULATE EPSG:3857 WORKING GRID
    # ========================================================

    (
        work_transform,
        work_width,
        work_height
    ) = calculate_default_transform(

        "EPSG:4326",

        "EPSG:3857",

        WMS_WIDTH,

        WMS_HEIGHT,

        minx,
        miny,
        maxx,
        maxy,

        resolution=WORKING_RESOLUTION_M
    )


    print(
        "\nWorking raster:"
    )

    print(
        "Width:",
        work_width
    )

    print(
        "Height:",
        work_height
    )

    print(
        "Resolution:",
        abs(work_transform.a),
        abs(work_transform.e)
    )


    # ========================================================
    # 21. REPROJECT LINEAMENT MASK
    # ========================================================

    print(
        "\nReprojecting lineament mask..."
    )

    working_mask = np.zeros(
        (
            work_height,
            work_width
        ),
        dtype=np.uint8
    )


    reproject(

        rendered.astype(
            np.uint8
        ),

        working_mask,

        src_transform=src_transform,

        src_crs="EPSG:4326",

        dst_transform=work_transform,

        dst_crs="EPSG:3857",

        resampling=Resampling.nearest
    )


    # ========================================================
    # 22. CALCULATE LOCAL LINEAMENT DENSITY
    # ========================================================

    print(
        "\nCalculating local lineament density..."
    )

    density = uniform_filter(

        working_mask.astype(
            np.float32
        ),

        size=WINDOW_PIXELS,

        mode="constant",

        cval=0
    )


    print(
        "Working density min:",
        float(density.min())
    )

    print(
        "Working density max:",
        float(density.max())
    )


    # ========================================================
    # 23. REPROJECT STATE DENSITY TO FINAL GRID
    # ========================================================

    print(
        "\nReprojecting density to final 250 m grid..."
    )


    state_final = np.full(
        (
            grid_height,
            grid_width
        ),
        np.nan,
        dtype=np.float32
    )


    reproject(

        density,

        state_final,

        src_transform=work_transform,

        src_crs="EPSG:3857",

        dst_transform=grid_transform,

        dst_crs=grid_crs,

        src_nodata=None,

        dst_nodata=np.nan,

        resampling=Resampling.bilinear
    )


    # ========================================================
    # 24. IMPORTANT CRS FIX
    # ========================================================
    #
    # The final grid is EPSG:6933.
    # Therefore the state geometry MUST also be in EPSG:6933
    # before geometry_mask() is called.
    #
    # This is the correction to the previous version.
    # ========================================================

    print(
        "\nReprojecting state boundary to final grid CRS..."
    )

    state_geom_final = (
        state_geom
        .to_crs(grid_crs)
    )


    # ========================================================
    # 25. CREATE STATE MASK
    # ========================================================

    state_mask = geometry_mask(

        state_geom_final.geometry,

        out_shape=(
            grid_height,
            grid_width
        ),

        transform=grid_transform,

        invert=True
    )


    print(
        "State mask cells:",
        int(state_mask.sum())
    )


    # ========================================================
    # 26. INSERT STATE INTO FINAL MOSAIC
    # ========================================================

    valid_state = (
        state_mask
        & np.isfinite(state_final)
    )


    final_density[
        valid_state
    ] = (
        state_final[
            valid_state
        ]
    )


    print(
        "Final state cells:",
        int(valid_state.sum())
    )


    # Free memory
    del rgba
    del rendered
    del rgb
    del white
    del black
    del working_mask
    del density
    del state_final
    del state_mask

    time.sleep(1)


# ============================================================
# 27. APPLY NER MASK
# ============================================================

print("\n" + "-" * 70)
print("APPLYING NER MASK")
print("-" * 70)

final_density[
    ~ner_mask
] = np.nan


# ============================================================
# 28. FINAL STATISTICS
# ============================================================

valid_mask = np.isfinite(
    final_density
)

values = final_density[
    valid_mask
]


print("\n" + "=" * 70)
print("FINAL LINEAMENT RASTER STATISTICS")
print("=" * 70)

print(
    "Total grid cells:",
    grid_width * grid_height
)

print(
    "NER cells:",
    int(ner_mask.sum())
)

print(
    "Valid lineament cells:",
    int(valid_mask.sum())
)

print(
    "NoData cells:",
    int((~valid_mask).sum())
)


if len(values) > 0:

    print(
        "Minimum:",
        float(values.min())
    )

    print(
        "Maximum:",
        float(values.max())
    )

    print(
        "Mean:",
        float(values.mean())
    )

    print(
        "Median:",
        float(np.median(values))
    )

    print(
        "Standard deviation:",
        float(values.std())
    )

else:

    raise RuntimeError(
        "FINAL LINEAMENT RASTER HAS ZERO VALID CELLS."
    )


# ============================================================
# 29. SAVE FINAL RASTER
# ============================================================

print("\n" + "-" * 70)
print("SAVING FINAL RASTER")
print("-" * 70)


profile = grid_profile.copy()

profile.update(

    driver="GTiff",

    dtype="float32",

    count=1,

    nodata=-9999.0,

    compress="deflate",

    predictor=3
)


output_array = np.where(

    np.isfinite(final_density),

    final_density,

    -9999.0

).astype(
    np.float32
)


with rasterio.open(

    OUTPUT_FILE,

    "w",

    **profile

) as dst:

    dst.write(
        output_array,
        1
    )


print(
    "Raster saved:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 30. STATE AVAILABILITY SUMMARY
# ============================================================

summary = pd.DataFrame(

    [
        {
            "state":
                state,

            "bhuvan_layer":
                layer,

            "available":
                available_layers[state],

            "note":
                (
                    "Official layer unavailable"
                    if layer is None
                    else ""
                )
        }

        for state, layer
        in LINEAMENT_LAYERS.items()
    ]
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 31. VERIFY OUTPUT
# ============================================================

print("\n" + "-" * 70)
print("VERIFYING OUTPUT")
print("-" * 70)


if not OUTPUT_FILE.exists():

    raise RuntimeError(
        "Output raster was not created."
    )


with rasterio.open(
    OUTPUT_FILE
) as src:

    check = src.read(1)

    check_valid = (
        check != src.nodata
    )

    print(
        "Output CRS:",
        src.crs
    )

    print(
        "Output size:",
        src.width,
        "x",
        src.height
    )

    print(
        "Output resolution:",
        src.res
    )

    print(
        "Output valid cells:",
        int(check_valid.sum())
    )


# ============================================================
# 32. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("STEP 68B COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nOutput:"
)

print(
    OUTPUT_FILE
)

print(
    "\nQA summary:"
)

print(
    SUMMARY_FILE
)

print(
    "\nMizoram:"
)

print(
    "Mizoram remains NoData because the official "
    "Bhuvan lineament layer was unavailable."
)

print(
    "\nNext step:"
)

print(
    "Validate this raster against the existing "
    "ner_lineament_density.csv before using it "
    "in the final susceptibility model."
)

print("=" * 70)