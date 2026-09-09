from pathlib import Path
from io import BytesIO

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
from PIL import Image
import requests


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

BOUNDARY_FILE = (
    PROJECT
    / "raw_data"
    / "boundaries"
    / "geoBoundaries-IND-ADM1.geojson"
)

MASK_FILE = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

OUTPUT_FILE = (
    PROJECT
    / "processed"
    / "predictors"
    / "geomorph_origin_250m.tif"
)

AUDIT_FILE = (
    PROJECT
    / "processed"
    / "step68n_geomorphology_rebuild_audit.csv"
)


# ============================================================
# BHUVAN WMS
# ============================================================

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)

WMS_VERSION = "1.1.1"

LAYERS = {
    "Arunāchal Pradesh":
        "geomorphology:AR_GM50K_0506",

    "Assam":
        "geomorphology:AS_GM50K_0506",

    "Manipur":
        "geomorphology:MN_GM50K_0506",

    "Meghālaya":
        "geomorphology:ML_GM50K_0506",

    "Nāgāland":
        "geomorphology:NL_GM50K_0506",

    "Sikkim":
        "geomorphology:SK_GM50K_0506",

    "Tripura":
        "geomorphology:TR_GM50K_0506",
}


# ============================================================
# GEOMORPHOLOGICAL ORIGIN CODES
# ============================================================

ORIGIN_CODES = {
    "Denudational": 1,
    "Fluvial": 2,
    "Glacial": 3,
    "Lacustrine": 4,
    "Structural": 5,
    "Water Bodies": 6,
}


# ============================================================
# IMPORTANT
# ============================================================
#
# These are the validated RGB representatives established
# during the previous Bhuvan colour audit.
#
# They are used only as a starting point for classification.
# A tolerance is applied because WMS rendering can introduce
# small RGB variations.
#
# ============================================================

CLASS_COLORS = {

    # Structural
    "Structural Origin-Moderately Dissected Hills and Valleys":
        (255, 85, 0),

    "Structural Origin-Highly Dissected Hills and Valleys":
        (255, 85, 0),

    "Structural Origin-Low Dissected Hills and Valleys":
        (255, 85, 0),

    "Structural Origin-Moderately Dissected Upper Plateau":
        (255, 85, 0),

    "Structural Origin-Highly Dissected Upper Plateau":
        (255, 85, 0),

    "Structural Origin-Low Dissected Upper Plateau":
        (255, 85, 0),

    "Structural Origin-Moderately Dissected Lower Plateau":
        (255, 85, 0),

    "Structural Origin-Highly Dissected Lower Plateau":
        (255, 85, 0),

    # Fluvial
    "Fluvial Origin-Younger Alluvial Plain":
        (165, 200, 100),

    "Fluvial Origin-Older Flood Plain":
        (165, 200, 100),

    "Fluvial Origin-Active Flood Plain":
        (165, 200, 100),

    "Fluvial Origin-Piedmont Alluvial Plain":
        (165, 200, 100),

    # Glacial
    "Glacial Origin-Snow Cover":
        (190, 232, 255),

    "Glacial Origin-Glacial Terrain":
        (236, 129, 75),

    # Denudational
    "Denudational Origin-Moderately Dissected Lower Plateau":
        (245, 122, 122),

    "Denudational Origin-Highly Dissected Lower Plateau":
        (245, 122, 122),

    # Water
    "Water Bodies-River":
        (96, 153, 239),
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def origin_from_class(class_name):

    if class_name is None:
        return None

    if class_name.startswith(
        "Structural Origin"
    ):
        return ORIGIN_CODES["Structural"]

    if class_name.startswith(
        "Fluvial Origin"
    ):
        return ORIGIN_CODES["Fluvial"]

    if class_name.startswith(
        "Glacial Origin"
    ):
        return ORIGIN_CODES["Glacial"]

    if class_name.startswith(
        "Denudational Origin"
    ):
        return ORIGIN_CODES["Denudational"]

    if class_name.startswith(
        "Water Bodies"
    ):
        return ORIGIN_CODES["Water Bodies"]

    if class_name.startswith(
        "Lacustrine Origin"
    ):
        return ORIGIN_CODES["Lacustrine"]

    return None


def classify_image(
    image,
    tolerance=45
):

    arr = np.asarray(
        image
    )

    rgb = arr[:, :, :3].astype(
        np.int16
    )

    output = np.zeros(
        rgb.shape[:2],
        dtype=np.uint8
    )

    best_distance = np.full(
        rgb.shape[:2],
        np.inf,
        dtype=np.float32
    )

    for class_name, color in CLASS_COLORS.items():

        color_array = np.array(
            color,
            dtype=np.int16
        )

        distance = np.sqrt(
            np.sum(
                (
                    rgb
                    - color_array
                ) ** 2,
                axis=2
            )
        )

        origin = origin_from_class(
            class_name
        )

        if origin is None:
            continue

        better = (
            distance
            < best_distance
        )

        accepted = (
            distance
            <= tolerance
        )

        update = (
            better
            & accepted
        )

        output[update] = origin
        best_distance[update] = (
            distance[update]
        )

    return output


def get_wms_image(
    bbox,
    layer,
    width=3000,
    height=3000
):

    params = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "SRS": "EPSG:4326",
        "BBOX": ",".join(
            map(str, bbox)
        ),
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
    }

    response = requests.get(
        WMS_URL,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get(
            "Content-Type",
            ""
        )
        .lower()
    )

    if "image" not in content_type:

        raise RuntimeError(
            f"WMS returned non-image response "
            f"for {layer}: "
            f"{content_type}"
        )

    return Image.open(
        BytesIO(
            response.content
        )
    ).convert("RGBA")


# ============================================================
# START
# ============================================================

print("=" * 75)
print(
    "STEP 68N — REBUILD GEOMORPHOLOGY "
    "USING CORRECTED NER MASK"
)
print("=" * 75)


# ============================================================
# LOAD MASK
# ============================================================

with rasterio.open(
    MASK_FILE
) as src:

    mask_arr = src.read(1)

    final_crs = src.crs
    final_transform = src.transform
    final_width = src.width
    final_height = src.height
    final_profile = src.profile.copy()

print(
    f"\nFinal CRS: {final_crs}"
)

print(
    f"Final grid: "
    f"{final_width} x {final_height}"
)

print(
    f"Resolution: "
    f"{src.res}"
)

ner_cells = int(
    np.sum(
        mask_arr == 1
    )
)

print(
    f"NER cells: "
    f"{ner_cells:,}"
)


# ============================================================
# LOAD BOUNDARIES
# ============================================================

boundary = gpd.read_file(
    BOUNDARY_FILE
)

boundary = boundary[
    boundary["shapeName"].isin(
        list(LAYERS.keys())
    )
].copy()

print(
    f"\nAvailable geomorphology states: "
    f"{len(boundary)}"
)


# ============================================================
# INITIAL OUTPUT
# ============================================================

final_geomorph = np.zeros(
    (
        final_height,
        final_width
    ),
    dtype=np.uint8
)


# ============================================================
# PROCESS EACH STATE
# ============================================================

audit = []


for state, layer in LAYERS.items():

    print("\n" + "-" * 75)
    print(
        f"Processing: {state}"
    )
    print(
        f"WMS layer: {layer}"
    )

    state_boundary = boundary[
        boundary["shapeName"]
        == state
    ]

    if state_boundary.empty:

        print(
            "Boundary not found — skipped."
        )

        continue

    # --------------------------------------------------------
    # STATE BBOX IN WGS84
    # --------------------------------------------------------

    bounds = (
        state_boundary
        .total_bounds
    )

    minx, miny, maxx, maxy = bounds

    print(
        f"BBOX: "
        f"{minx:.6f}, "
        f"{miny:.6f}, "
        f"{maxx:.6f}, "
        f"{maxy:.6f}"
    )

    # Slightly enlarge bbox
    dx = (
        maxx - minx
    ) * 0.002

    dy = (
        maxy - miny
    ) * 0.002

    bbox = (
        minx - dx,
        miny - dy,
        maxx + dx,
        maxy + dy
    )

    # --------------------------------------------------------
    # DOWNLOAD WMS
    # --------------------------------------------------------

    try:

        image = get_wms_image(
            bbox,
            layer
        )

    except Exception as exc:

        print(
            f"WMS FAILED: {exc}"
        )

        audit.append({
            "state": state,
            "layer": layer,
            "status": "WMS_FAILED",
            "rendered_pixels": 0,
        })

        continue

    print(
        f"WMS image: "
        f"{image.size}"
    )

    # --------------------------------------------------------
    # CLASSIFY
    # --------------------------------------------------------

    classified = classify_image(
        image,
        tolerance=45
    )

    rendered_pixels = int(
        np.sum(
            classified > 0
        )
    )

    print(
        f"Classified pixels: "
        f"{rendered_pixels:,}"
    )

    # --------------------------------------------------------
    # CREATE TEMPORARY WGS84 RASTER
    # --------------------------------------------------------

    transform = rasterio.transform.from_bounds(
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        image.width,
        image.height
    )

    temp = np.zeros_like(
        classified,
        dtype=np.uint8
    )

    temp[:, :] = classified

    # --------------------------------------------------------
    # REPROJECT TO FINAL GRID
    # --------------------------------------------------------

    reprojected = np.zeros(
        (
            final_height,
            final_width
        ),
        dtype=np.uint8
    )

    reproject(
        source=temp,
        destination=reprojected,
        src_transform=transform,
        src_crs="EPSG:4326",
        dst_transform=final_transform,
        dst_crs=final_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0
    )

    # --------------------------------------------------------
    # STATE MASK
    # --------------------------------------------------------

    state_projected = (
        state_boundary
        .to_crs(final_crs)
    )

    state_mask = geometry_mask(
        state_projected.geometry,
        out_shape=(
            final_height,
            final_width
        ),
        transform=final_transform,
        invert=True
    )

    # --------------------------------------------------------
    # APPLY STATE MASK
    # --------------------------------------------------------

    valid_state = (
        state_mask
        &
        (reprojected > 0)
        &
        (mask_arr == 1)
    )

    final_geomorph[
        valid_state
    ] = reprojected[
        valid_state
    ]

    final_count = int(
        np.sum(
            valid_state
        )
    )

    print(
        f"Final valid state cells: "
        f"{final_count:,}"
    )

    audit.append({
        "state": state,
        "layer": layer,
        "status": "SUCCESS",
        "rendered_pixels": rendered_pixels,
        "final_valid_cells": final_count,
    })


# ============================================================
# APPLY NER MASK
# ============================================================

final_geomorph[
    mask_arr != 1
] = 0


# ============================================================
# SUMMARY
# ============================================================

valid_final = (
    final_geomorph > 0
)

valid_count = int(
    np.sum(
        valid_final
    )
)

coverage = (
    valid_count
    / ner_cells
    * 100
)


print("\n" + "=" * 75)
print("FINAL GEOMORPHOLOGY SUMMARY")
print("=" * 75)

print(
    f"NER cells: "
    f"{ner_cells:,}"
)

print(
    f"Valid geomorphology cells: "
    f"{valid_count:,}"
)

print(
    f"Coverage: "
    f"{coverage:.2f}%"
)


print(
    "\nOrigin counts:"
)

for code, name in [
    (1, "Denudational"),
    (2, "Fluvial"),
    (3, "Glacial"),
    (4, "Lacustrine"),
    (5, "Structural"),
    (6, "Water Bodies"),
]:

    count = int(
        np.sum(
            final_geomorph
            == code
        )
    )

    print(
        f"  {code} = {name}: "
        f"{count:,}"
    )


# ============================================================
# WRITE RASTER
# ============================================================

profile = final_profile.copy()

profile.update(
    driver="GTiff",
    dtype="uint8",
    count=1,
    nodata=0,
    compress="deflate",
    BIGTIFF="IF_SAFER"
)

profile.pop(
    "blockxsize",
    None
)

profile.pop(
    "blockysize",
    None
)


with rasterio.open(
    OUTPUT_FILE,
    "w",
    **profile
) as dst:

    dst.write(
        final_geomorph,
        1
    )


# ============================================================
# SAVE AUDIT
# ============================================================

audit_df = pd.DataFrame(
    audit
)

audit_df.to_csv(
    AUDIT_FILE,
    index=False
)


print("\n" + "=" * 75)
print("OUTPUT")
print("=" * 75)

print(
    f"Raster:\n{OUTPUT_FILE}"
)

print(
    f"Audit:\n{AUDIT_FILE}"
)

print("\n" + "=" * 75)
print("STEP 68N COMPLETE")
print("=" * 75)