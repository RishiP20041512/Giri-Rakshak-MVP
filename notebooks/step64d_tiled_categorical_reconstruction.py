"""
STEP 64D-F3
Tiled categorical reconstruction from Bhuvan WMS.

Purpose:
    Improve categorical raster coverage by requesting smaller
    geographic WMS tiles instead of one very large state image.

Factors:
    - Geomorphology -> geomorph_origin
    - LUCC -> Level_I

Final grid:
    EPSG:6933
    250 m x 250 m

Rules:
    - WMS rendered categorical data only
    - nearest validated RGB palette
    - nearest-neighbour reprojection
    - no categorical interpolation
    - no artificial filling of NoData
    - Mizoram geomorphology remains NoData
"""

from pathlib import Path
import io
import time

import numpy as np
import pandas as pd
import requests
from PIL import Image

import geopandas as gpd

import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask


# ================================================================
# PROJECT PATHS
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
    / "categorical_tiled_reconstruction"
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
    / "step64dF3_tiled_reconstruction_audit.csv"
)


# ================================================================
# BHUVAN
# ================================================================

WMS_URL = (
    "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
)


GEOM_LAYERS = {

    "Arunachal Pradesh":
        "geomorphology:AR_GM50K_0506",

    "Assam":
        "geomorphology:AS_GM50K_0506",

    "Manipur":
        "geomorphology:MN_GM50K_0506",

    "Meghalaya":
        "geomorphology:ML_GM50K_0506",

    "Mizoram":
        None,

    "Nagaland":
        "geomorphology:NL_GM50K_0506",

    "Sikkim":
        "geomorphology:SK_GM50K_0506",

    "Tripura":
        "geomorphology:TR_GM50K_0506",
}


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

# Keep the known-working WMS image size.
WMS_IMAGE_SIZE = 3000

REQUEST_TIMEOUT = 180

MAX_RETRIES = 3

# Keep the validated tolerance from the original reconstruction.
COLOR_DISTANCE_MAX = 80.0

ALPHA_THRESHOLD = 80

NODATA = 0

# Geographic tile size in degrees.
# Smaller than whole-state request.
TILE_SIZE_DEG = 1.0


# ================================================================
# CATEGORY CODES
# ================================================================

GEOM_CODES = {

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
# HELPERS
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

    return gdf[
        gdf["state_norm"].isin(
            states
        )
    ].copy()


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

    for col in required:

        if col not in df.columns:

            raise RuntimeError(
                f"Missing column {col} "
                f"in {path.name}"
            )

    result = []

    for _, row in df.iterrows():

        result.append(
            {
                "class":
                    str(row["class"]),

                "rgb":
                    np.array(
                        [
                            float(row["mode_R"]),
                            float(row["mode_G"]),
                            float(row["mode_B"]),
                        ],
                        dtype=np.float32,
                    ),
            }
        )

    return result


def get_origin(
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


def build_origin_palette(
    class_palette
):

    grouped = {}

    for item in class_palette:

        origin = get_origin(
            item["class"]
        )

        if origin is None:

            continue

        grouped.setdefault(
            origin,
            []
        ).append(
            item["rgb"]
        )

    result = {}

    for origin, colors in (
        grouped.items()
    ):

        result[origin] = np.median(
            np.vstack(colors),
            axis=0
        )

    return [
        {
            "class":
                origin,

            "rgb":
                rgb,
        }

        for origin, rgb
        in result.items()
    ]


def generate_tiles(
    bbox,
    tile_size
):

    minx, miny, maxx, maxy = bbox

    tiles = []

    x = minx

    while x < maxx:

        x2 = min(
            x + tile_size,
            maxx
        )

        y = miny

        while y < maxy:

            y2 = min(
                y + tile_size,
                maxy
            )

            tiles.append(
                (
                    x,
                    y,
                    x2,
                    y2
                )
            )

            y = y2

        x = x2

    return tiles


def request_wms(
    layer,
    bbox
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
            WMS_IMAGE_SIZE,

        "HEIGHT":
            WMS_IMAGE_SIZE,

        "FORMAT":
            "image/png",

        "TRANSPARENT":
            "TRUE",
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.get(
                WMS_URL,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            image = Image.open(
                io.BytesIO(
                    response.content
                )
            ).convert(
                "RGBA"
            )

            array = np.asarray(
                image
            )

            expected = (
                WMS_IMAGE_SIZE,
                WMS_IMAGE_SIZE,
                4
            )

            if array.shape != expected:

                raise RuntimeError(
                    f"Unexpected shape "
                    f"{array.shape}"
                )

            return array

        except Exception as exc:

            last_error = exc

            print(
                f"    Attempt "
                f"{attempt}/"
                f"{MAX_RETRIES} failed: "
                f"{exc}"
            )

            time.sleep(4)

    raise RuntimeError(
        "WMS request failed: "
        f"{last_error}"
    )


def classify_colors(
    rgba,
    palette
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

    for start in range(
        0,
        len(indices),
        chunk_size
    ):

        idx = indices[
            start:
            start + chunk_size
        ]

        pixels = flat_rgb[idx]

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
            <= COLOR_DISTANCE_MAX
        )

        output_flat[
            idx[accepted]
        ] = (
            nearest[accepted]
            + 1
        )

    return output


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


def reproject_to_grid(
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
# PROCESS FACTOR
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

    audit_rows = []

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

            audit_rows.append(
                {
                    "factor":
                        factor_name,

                    "state":
                        state,

                    "tile":
                        "ALL",

                    "status":
                        "SOURCE_UNAVAILABLE",

                    "recognized_pixels":
                        0,

                    "total_pixels":
                        0,

                    "recognized_percent":
                        0,

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
                "  State geometry not found."
            )

            continue

        geometry = (
            state_geom.geometry
            .union_all()
        )

        state_bbox = (
            geometry.bounds
        )

        tiles = generate_tiles(
            state_bbox,
            TILE_SIZE_DEG
        )

        print(
            "  State tiles:",
            len(tiles)
        )

        # --------------------------------------------------------
        # Process tiles
        # --------------------------------------------------------

        for tile_number, tile_bbox in enumerate(
            tiles,
            start=1
        ):

            print(
                f"\n  Tile "
                f"{tile_number}/"
                f"{len(tiles)}"
            )

            print(
                "    BBOX:",
                tuple(
                    round(
                        x,
                        6
                    )
                    for x in tile_bbox
                )
            )

            try:

                rgba = request_wms(
                    layer,
                    tile_bbox
                )

            except Exception as exc:

                print(
                    "    FAILED:",
                    exc
                )

                audit_rows.append(
                    {
                        "factor":
                            factor_name,

                        "state":
                            state,

                        "tile":
                            tile_number,

                        "status":
                            "WMS_FAILED",

                        "recognized_pixels":
                            0,

                        "total_pixels":
                            0,

                        "recognized_percent":
                            0,

                        "final_valid_cells":
                            0,
                    }
                )

                continue

            classified = (
                classify_colors(
                    rgba,
                    palette
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

            percent = (
                recognized
                /
                total
                *
                100
            )

            print(
                f"    Recognized: "
                f"{recognized:,}/"
                f"{total:,} "
                f"({percent:.2f}%)"
            )

            raw_path = (
                WORK_DIR
                /
                (
                    f"{factor_name}_"
                    f"{state.replace(' ', '_')}_"
                    f"tile_{tile_number}.tif"
                )
            )

            write_raw(
                raw_path,
                classified,
                tile_bbox
            )

            tile_grid = (
                reproject_to_grid(
                    raw_path,
                    grid
                )
            )

            # ----------------------------------------------------
            # State clipping
            # ----------------------------------------------------

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

            tile_grid[
                ~state_mask
            ] = NODATA

            valid = (
                tile_grid > 0
            )

            valid_count = int(
                np.sum(valid)
            )

            print(
                "    Final valid cells:",
                f"{valid_count:,}"
            )

            # ----------------------------------------------------
            # Merge into mosaic
            # ----------------------------------------------------

            fill = (
                valid
                &
                (
                    mosaic
                    == NODATA
                )
            )

            mosaic[fill] = (
                tile_grid[fill]
            )

            audit_rows.append(
                {
                    "factor":
                        factor_name,

                    "state":
                        state,

                    "tile":
                        tile_number,

                    "status":
                        "OK",

                    "recognized_pixels":
                        recognized,

                    "total_pixels":
                        total,

                    "recognized_percent":
                        percent,

                    "final_valid_cells":
                        valid_count,
                }
            )

    return (
        mosaic,
        audit_rows
    )


# ================================================================
# MAIN
# ================================================================

def main():

    print_header(
        "STEP 64D-F3 — TILED "
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
    # GRID
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
        "Size:",
        grid["width"],
        "x",
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
    # BOUNDARY
    # ============================================================

    print_header(
        "[2] Loading NER boundary"
    )

    boundary = load_boundary()

    print(
        "NER states:",
        len(boundary)
    )

    # ============================================================
    # PALETTES
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

    geom_palette = (
        build_origin_palette(
            geom_class_palette
        )
    )

    print(
        "\nGeomorphology origin palette:"
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

    print(
        "\nLUCC palette:"
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
    # GEOMORPHOLOGY
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
    # LUCC
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
    # FINAL NER MASK
    # ============================================================

    print_header(
        "[4] Applying final NER mask"
    )

    with rasterio.open(
        MASK
    ) as src:

        ner_mask = src.read(
            1
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
    # WRITE
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
    # AUDIT
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
    # FINAL STATISTICS
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

        values, counts = np.unique(
            data[valid],
            return_counts=True
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
    # PROPERTY CHECK
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

    print_header(
        "STEP 64D-F3 COMPLETED"
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
        "Run Step 64D-F validation."
    )


if __name__ == "__main__":

    main()