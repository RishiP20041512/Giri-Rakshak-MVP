"""
STEP 64D-F2
Improved categorical raster reconstruction.

Factors:
    1. Geomorphology -> geomorph_origin
    2. LUCC -> Level_I

Method:
    - State-wise Bhuvan WMS GetMap
    - Higher WMS sampling density
    - Validated color palette matching
    - Transparent/white background = NoData
    - Nearest-neighbour reprojection to exact 250 m grid
    - State mosaicking
    - Final NER mask

Important:
    - No bilinear/cubic interpolation.
    - NoData is not artificially filled.
    - Mizoram geomorphology remains NoData because its official
      Bhuvan layer is unavailable.
"""

from pathlib import Path
import io
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image

import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask

import geopandas as gpd


# ================================================================
# PATHS
# ================================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

BOUNDARY = (
    PROJECT
    / "raw_data"
    / "boundaries"
    / "geoBoundaries-IND-ADM1.geojson"
)

GRID = (
    PROJECT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

MASK = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)

GEOM_PALETTE = (
    PROJECT
    / "processed"
    / "categorical_palette_validation"
    / "geomorphology_validated_palette.csv"
)

LULC_PALETTE = (
    PROJECT
    / "processed"
    / "categorical_palette_validation"
    / "lulc_validated_palette.csv"
)

OUT_DIR = (
    PROJECT
    / "processed"
    / "predictors"
)

WORK_DIR = (
    PROJECT
    / "processed"
    / "categorical_reconstruction_f2"
)

OUT_GEOM = (
    OUT_DIR
    / "geomorph_origin_250m.tif"
)

OUT_LULC = (
    OUT_DIR
    / "lulc_level1_250m.tif"
)

AUDIT_OUT = (
    WORK_DIR
    / "step64dF2_reconstruction_audit.csv"
)


# ================================================================
# BHUVAN WMS
# ================================================================

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)


# Geomorphology 1:50,000
GEOM_LAYERS = {

    "Arunachal Pradesh":
        "geomorphology:AR_GM50K_0506",

    "Assam":
        "geomorphology:AS_GM50K_0506",

    "Manipur":
        "geomorphology:MN_GM50K_0506",

    "Meghalaya":
        "geomorphology:ML_GM50K_0506",

    # Official layer unavailable
    "Mizoram":
        None,

    "Nagaland":
        "geomorphology:NL_GM50K_0506",

    "Sikkim":
        "geomorphology:SK_GM50K_0506",

    "Tripura":
        "geomorphology:TR_GM50K_0506",
}


# LUCC 10K
LULC_LAYERS = {

    "Arunachal Pradesh":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_AR",

    "Assam":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_AS",

    "Manipur":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_MN",

    "Meghalaya":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_ML",

    "Mizoram":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_MZ",

    "Nagaland":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_NL",

    "Sikkim":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_SK",

    "Tripura":
        "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_TR",
}


# ================================================================
# STATE NAME NORMALIZATION
# ================================================================

STATE_NAME_MAP = {

    "Arunāchal Pradesh":
        "Arunachal Pradesh",

    "Assam":
        "Assam",

    "Manipur":
        "Manipur",

    "Meghālaya":
        "Meghalaya",

    "Mizoram":
        "Mizoram",

    "Nāgāland":
        "Nagaland",

    "Sikkim":
        "Sikkim",

    "Tripura":
        "Tripura",
}


# ================================================================
# SETTINGS
# ================================================================

# Increased from previous 3000 x 3000.
IMAGE_SIZE = 3000

REQUEST_TIMEOUT = 180

# Broader RGB tolerance to handle WMS anti-aliasing.
COLOR_DISTANCE_MAX = 80.0

# Alpha threshold.
ALPHA_THRESHOLD = 80

# GeoTIFF NoData code.
NODATA = 0


# ================================================================
# CATEGORY CODES
# ================================================================

GEOM_ORIGIN_CODES = {

    "Denudational Origin": 1,

    "Fluvial Origin": 2,

    "Glacial Origin": 3,

    "Lacustrine Origin": 4,

    "Structural Origin": 5,

    "Water Bodies": 6,
}


LULC_CODES = {

    "Forest": 1,

    "Wastelands": 2,

    "Water Bodies": 3,

    "Agriculture": 4,

    "Others": 5,

    "Built-up": 6,

    "Grasslands / Grazing Lands": 7,
}


# ================================================================
# PRINT HELPER
# ================================================================

def print_header(text):

    print(
        "\n"
        + "=" * 70
    )

    print(text)

    print(
        "=" * 70
    )


# ================================================================
# LOAD GRID
# ================================================================

def load_grid():

    with rasterio.open(
        GRID
    ) as src:

        return {

            "width":
                src.width,

            "height":
                src.height,

            "transform":
                src.transform,

            "crs":
                src.crs,

            "profile":
                src.profile.copy(),
        }


# ================================================================
# LOAD BOUNDARY
# ================================================================

def load_boundary():

    gdf = gpd.read_file(
        BOUNDARY
    )

    if gdf.crs is None:

        raise RuntimeError(
            "Boundary CRS is missing."
        )

    gdf["state_norm"] = (
        gdf["shapeName"]
        .map(
            lambda x:
                STATE_NAME_MAP.get(
                    x,
                    x
                )
        )
    )

    states = list(
        GEOM_LAYERS.keys()
    )

    ner = gdf[
        gdf["state_norm"].isin(
            states
        )
    ].copy()

    return ner


# ================================================================
# LOAD PALETTE
# ================================================================

def load_palette(path):

    df = pd.read_csv(
        path
    )

    required = [
        "class",
        "mode_R",
        "mode_G",
        "mode_B",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"{path.name} is missing "
                f"column: {column}"
            )

    result = []

    for _, row in df.iterrows():

        rgb = np.array(
            [
                float(row["mode_R"]),
                float(row["mode_G"]),
                float(row["mode_B"]),
            ],
            dtype=np.float32,
        )

        result.append(
            {
                "class":
                    str(row["class"]),

                "rgb":
                    rgb,
            }
        )

    return result


# ================================================================
# DERIVE GEOMORPHOLOGY ORIGIN
# ================================================================

def geomorph_origin(
    class_name
):

    prefixes = [

        "Structural Origin",

        "Fluvial Origin",

        "Denudational Origin",

        "Glacial Origin",

        "Water Bodies",

        "Lacustrine Origin",
    ]

    for prefix in prefixes:

        if class_name.startswith(
            prefix
        ):

            return prefix

    return None


# ================================================================
# BUILD ORIGIN PALETTE
# ================================================================

def build_origin_palette(
    class_palette
):

    grouped = {}

    for item in class_palette:

        origin = geomorph_origin(
            item["class"]
        )

        if origin is None:

            continue

        if origin not in grouped:

            grouped[origin] = []

        grouped[
            origin
        ].append(
            item["rgb"]
        )

    output = {}

    for origin, colors in grouped.items():

        output[origin] = np.median(
            np.vstack(colors),
            axis=0
        )

    return output


# ================================================================
# REQUEST BHUVAN WMS
# ================================================================

def request_wms(
    layer,
    bbox,
    width,
    height
):

    minx, miny, maxx, maxy = bbox

    params = {

        "SERVICE":
            "WMS",

        "VERSION":
            "1.1.1",

        "REQUEST":
            "GetMap",

        "LAYERS":
            layer,

        "STYLES":
            "",

        "SRS":
            "EPSG:4326",

        "BBOX":
            (
                f"{minx},"
                f"{miny},"
                f"{maxx},"
                f"{maxy}"
            ),

        "WIDTH":
            width,

        "HEIGHT":
            height,

        "FORMAT":
            "image/png",

        "TRANSPARENT":
            "TRUE",
    }

    last_error = None

    for attempt in range(
        1,
        4
    ):

        try:

            response = requests.get(
                WMS_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            image = Image.open(
                io.BytesIO(
                    response.content
                )
            ).convert(
                "RGBA"
            )

            arr = np.asarray(
                image
            )

            expected_shape = (
                height,
                width,
                4
            )

            if arr.shape != expected_shape:

                raise RuntimeError(
                    "Unexpected WMS image shape: "
                    f"{arr.shape}"
                )

            return arr

        except Exception as exc:

            last_error = exc

            print(
                f"  WMS attempt "
                f"{attempt}/3 failed: "
                f"{exc}"
            )

            time.sleep(4)

    raise RuntimeError(
        "WMS request failed: "
        f"{last_error}"
    )


# ================================================================
# CLASSIFY RENDERED COLORS
# ================================================================

def classify_pixels(
    rgba,
    palette,
    max_distance
):

    rgb = (
        rgba[:, :, :3]
        .astype(np.float32)
    )

    alpha = (
        rgba[:, :, 3]
    )

    height, width, _ = (
        rgb.shape
    )

    output = np.zeros(
        (
            height,
            width
        ),
        dtype=np.uint16
    )

    palette_rgb = np.vstack(
        [
            item["rgb"]
            for item in palette
        ]
    )

    # ------------------------------------------------------------
    # Valid rendered pixels
    # ------------------------------------------------------------

    valid = (
        alpha
        > ALPHA_THRESHOLD
    )

    # Remove white background.
    white = np.all(
        rgb >= 245,
        axis=2
    )

    valid &= ~white

    flat_rgb = (
        rgb.reshape(
            -1,
            3
        )
    )

    flat_valid = (
        valid.reshape(
            -1
        )
    )

    indices = np.flatnonzero(
        flat_valid
    )

    output_flat = (
        output.reshape(
            -1
        )
    )

    chunk_size = 500_000

    # ------------------------------------------------------------
    # Nearest validated palette color
    # ------------------------------------------------------------

    for start in range(
        0,
        len(indices),
        chunk_size
    ):

        idx = indices[
            start:
            start + chunk_size
        ]

        pixels = (
            flat_rgb[idx]
        )

        difference = (
            pixels[:, None, :]
            -
            palette_rgb[None, :, :]
        )

        distance = np.sqrt(
            np.sum(
                difference ** 2,
                axis=2
            )
        )

        nearest = np.argmin(
            distance,
            axis=1
        )

        nearest_distance = (
            distance[
                np.arange(
                    len(idx)
                ),
                nearest
            ]
        )

        accepted = (
            nearest_distance
            <= max_distance
        )

        output_flat[
            idx[accepted]
        ] = (
            nearest[accepted]
            + 1
        )

    return output


# ================================================================
# WRITE RAW GEOTIFF
# ================================================================

def write_raw(
    path,
    data,
    bbox
):

    height, width = (
        data.shape
    )

    transform = from_bounds(
        bbox[0],
        bbox[1],
        bbox[2],
        bbox[3],
        width,
        height
    )

    profile = {

        "driver":
            "GTiff",

        "height":
            height,

        "width":
            width,

        "count":
            1,

        "dtype":
            "uint16",

        "crs":
            "EPSG:4326",

        "transform":
            transform,

        "nodata":
            NODATA,

        "compress":
            "lzw",
    }

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            data,
            1
        )


# ================================================================
# REPROJECT TO FINAL GRID
# ================================================================

def reproject_state(
    source,
    grid
):

    destination = np.zeros(
        (
            grid["height"],
            grid["width"]
        ),
        dtype=np.uint16
    )

    with rasterio.open(
        source
    ) as src:

        reproject(

            source=
                rasterio.band(
                    src,
                    1
                ),

            destination=
                destination,

            src_transform=
                src.transform,

            src_crs=
                src.crs,

            src_nodata=
                NODATA,

            dst_transform=
                grid["transform"],

            dst_crs=
                grid["crs"],

            dst_nodata=
                NODATA,

            resampling=
                Resampling.nearest,
        )

    return destination


# ================================================================
# WRITE FINAL RASTER
# ================================================================

def write_final(
    path,
    data,
    grid
):

    profile = (
        grid["profile"]
        .copy()
    )

    profile.update(

        driver="GTiff",

        dtype="uint16",

        count=1,

        nodata=NODATA,

        compress="lzw",

        tiled=True,

        blockxsize=256,

        blockysize=256,

        BIGTIFF="IF_SAFER",
    )

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            data,
            1
        )


# ================================================================
# PROCESS ONE FACTOR
# ================================================================

def process_factor(
    factor_name,
    layers,
    boundary,
    palette,
    grid
):

    print_header(
        factor_name.upper()
    )

    mosaic = np.zeros(
        (
            grid["height"],
            grid["width"]
        ),
        dtype=np.uint16
    )

    audit = []

    for state in layers:

        print(
            f"\n[{state}]"
        )

        layer = layers[
            state
        ]

        # --------------------------------------------------------
        # Missing source
        # --------------------------------------------------------

        if layer is None:

            print(
                "  Source layer unavailable."
            )

            print(
                "  State remains NoData."
            )

            audit.append(
                {
                    "factor":
                        factor_name,

                    "state":
                        state,

                    "status":
                        "SOURCE_UNAVAILABLE",

                    "recognized_pixels":
                        0,

                    "total_pixels":
                        0,

                    "recognized_percent":
                        0.0,

                    "final_valid_cells":
                        0,
                }
            )

            continue

        # --------------------------------------------------------
        # State geometry
        # --------------------------------------------------------

        state_geom = boundary[
            boundary[
                "state_norm"
            ]
            == state
        ]

        if state_geom.empty:

            print(
                "  ERROR: state geometry "
                "not found."
            )

            audit.append(
                {
                    "factor":
                        factor_name,

                    "state":
                        state,

                    "status":
                        "GEOMETRY_MISSING",

                    "recognized_pixels":
                        0,

                    "total_pixels":
                        0,

                    "recognized_percent":
                        0.0,

                    "final_valid_cells":
                        0,
                }
            )

            continue

        geometry = (
            state_geom.geometry
            .union_all()
        )

        bbox = geometry.bounds

        print(
            "  BBOX:",
            tuple(
                round(
                    x,
                    6
                )
                for x in bbox
            )
        )

        print(
            "  Layer:",
            layer
        )

        # --------------------------------------------------------
        # Download WMS
        # --------------------------------------------------------

        rgba = request_wms(
            layer,
            bbox,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        print(
            "  Image received:",
            rgba.shape
        )

        # --------------------------------------------------------
        # Classify colors
        # --------------------------------------------------------

        classified = (
            classify_pixels(
                rgba,
                palette,
                COLOR_DISTANCE_MAX
            )
        )

        recognized = int(
            np.sum(
                classified > 0
            )
        )

        total = (
            classified.size
        )

        recognized_percent = (
            recognized
            /
            total
            *
            100.0
        )

        print(
            f"  Recognized pixels: "
            f"{recognized:,}/"
            f"{total:,} "
            f"({recognized_percent:.2f}%)"
        )

        # --------------------------------------------------------
        # Temporary raw raster
        # --------------------------------------------------------

        raw_path = (
            WORK_DIR
            /
            (
                f"{factor_name}_"
                f"{state.replace(' ', '_')}_"
                f"raw.tif"
            )
        )

        write_raw(
            raw_path,
            classified,
            bbox
        )

        # --------------------------------------------------------
        # Reproject
        # --------------------------------------------------------

        state_grid = (
            reproject_state(
                raw_path,
                grid
            )
        )

        # --------------------------------------------------------
        # State mask
        # --------------------------------------------------------

        state_projected = (
            state_geom
            .to_crs(
                grid["crs"]
            )
        )

        state_mask = (
            geometry_mask(
                state_projected.geometry,
                out_shape=(
                    grid["height"],
                    grid["width"]
                ),
                transform=
                    grid["transform"],
                invert=True
            )
        )

        state_grid[
            ~state_mask
        ] = NODATA

        valid = (
            state_grid > 0
        )

        valid_count = int(
            np.sum(valid)
        )

        print(
            "  Final valid cells:",
            f"{valid_count:,}"
        )

        # --------------------------------------------------------
        # Mosaic
        # --------------------------------------------------------

        fill = (
            valid
            &
            (
                mosaic
                == NODATA
            )
        )

        mosaic[fill] = (
            state_grid[fill]
        )

        audit.append(
            {
                "factor":
                    factor_name,

                "state":
                    state,

                "status":
                    "OK",

                "recognized_pixels":
                    recognized,

                "total_pixels":
                    total,

                "recognized_percent":
                    recognized_percent,

                "final_valid_cells":
                    valid_count,
            }
        )

    return (
        mosaic,
        audit
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "STEP 64D-F2 — IMPROVED "
        "CATEGORICAL RECONSTRUCTION"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # 1. GRID
    # ============================================================

    print_header(
        "[1] Loading final grid"
    )

    grid = load_grid()

    print(
        "CRS:",
        grid["crs"]
    )

    print(
        "Width:",
        grid["width"]
    )

    print(
        "Height:",
        grid["height"]
    )

    print(
        "Resolution:",
        grid["transform"].a,
        "x",
        abs(
            grid["transform"].e
        ),
        "m"
    )

    # ============================================================
    # 2. BOUNDARY
    # ============================================================

    print_header(
        "[2] Loading NER boundary"
    )

    boundary = load_boundary()

    print(
        "NER states:",
        len(boundary)
    )

    for state in sorted(
        boundary[
            "state_norm"
        ].unique()
    ):

        print(
            " ",
            state
        )

    # ============================================================
    # 3. PALETTES
    # ============================================================

    print_header(
        "[3] Loading validated palettes"
    )

    geom_class_palette = (
        load_palette(
            GEOM_PALETTE
        )
    )

    lulc_palette = (
        load_palette(
            LULC_PALETTE
        )
    )

    # ------------------------------------------------------------
    # Build geomorphology origin palette
    # ------------------------------------------------------------

    origin_palette = (
        build_origin_palette(
            geom_class_palette
        )
    )

    geom_palette = []

    for origin, rgb in (
        origin_palette.items()
    ):

        geom_palette.append(
            {
                "class":
                    origin,

                "rgb":
                    rgb,
            }
        )

    print(
        "\nGeomorphology origin colors:"
    )

    for item in geom_palette:

        rgb_tuple = tuple(
            np.round(
                item["rgb"]
            ).astype(int)
        )

        print(
            f"  "
            f"{item['class']:25s}"
            f" -> "
            f"{rgb_tuple}"
        )

    # ------------------------------------------------------------
    # LUCC palette
    # ------------------------------------------------------------

    print(
        "\nLUCC colors:"
    )

    for item in lulc_palette:

        rgb_tuple = tuple(
            np.round(
                item["rgb"]
            ).astype(int)
        )

        print(
            f"  "
            f"{item['class']:30s}"
            f" -> "
            f"{rgb_tuple}"
        )

    # ============================================================
    # 4. GEOMORPHOLOGY
    # ============================================================

    geom_raster, geom_audit = (
        process_factor(

            "geomorph_origin",

            GEOM_LAYERS,

            boundary,

            geom_palette,

            grid
        )
    )

    # ============================================================
    # 5. LUCC
    # ============================================================

    lulc_raster, lulc_audit = (
        process_factor(

            "lulc_level1",

            LULC_LAYERS,

            boundary,

            lulc_palette,

            grid
        )
    )

    # ============================================================
    # 6. NER MASK
    # ============================================================

    print_header(
        "[4] Applying NER mask"
    )

    with rasterio.open(
        MASK
    ) as src:

        ner_mask = (
            src.read(1)
        )

    ner_valid = (
        ner_mask > 0
    )

    geom_raster[
        ~ner_valid
    ] = NODATA

    lulc_raster[
        ~ner_valid
    ] = NODATA

    # ============================================================
    # 7. WRITE FINAL RASTERS
    # ============================================================

    print_header(
        "[5] Writing final rasters"
    )

    write_final(
        OUT_GEOM,
        geom_raster,
        grid
    )

    print(
        "Written:",
        OUT_GEOM
    )

    write_final(
        OUT_LULC,
        lulc_raster,
        grid
    )

    print(
        "Written:",
        OUT_LULC
    )

    # ============================================================
    # 8. AUDIT
    # ============================================================

    audit = pd.DataFrame(
        geom_audit
        +
        lulc_audit
    )

    audit.to_csv(
        AUDIT_OUT,
        index=False
    )

    # ============================================================
    # 9. FINAL STATISTICS
    # ============================================================

    print_header(
        "[6] Final raster statistics"
    )

    for name, data in [

        (
            "geomorph_origin",
            geom_raster
        ),

        (
            "lulc_level1",
            lulc_raster
        ),
    ]:

        valid = (
            data > 0
        )

        print(
            f"\n{name}"
        )

        print(
            "  Valid cells:",
            f"{int(np.sum(valid)):,}"
        )

        print(
            "  NoData cells:",
            f"{int(np.sum(~valid)):,}"
        )

        values, counts = (
            np.unique(
                data[valid],
                return_counts=True
            )
        )

        for value, count in zip(
            values,
            counts
        ):

            print(
                f"    code "
                f"{int(value)}: "
                f"{int(count):,}"
            )

    # ============================================================
    # 10. PROPERTY CHECK
    # ============================================================

    print_header(
        "[7] Raster property check"
    )

    for path in [

        OUT_GEOM,

        OUT_LULC,
    ]:

        with rasterio.open(
            path
        ) as src:

            print(
                f"\n{path.name}"
            )

            print(
                "  CRS:",
                src.crs
            )

            print(
                "  Size:",
                src.width,
                "x",
                src.height
            )

            print(
                "  Resolution:",
                src.res
            )

            print(
                "  Data type:",
                src.dtypes[0]
            )

            print(
                "  NoData:",
                src.nodata
            )

    # ============================================================
    # COMPLETE
    # ============================================================

    print_header(
        "STEP 64D-F2 COMPLETED"
    )

    print(
        "\nOutputs:"
    )

    print(
        OUT_GEOM
    )

    print(
        OUT_LULC
    )

    print(
        AUDIT_OUT
    )

    print(
        "\nNext:"
    )

    print(
        "Run categorical point validation."
    )


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":

    main()