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
    / "categorical_classlevel_reconstruction"
)

OUT_GEOM = (
    OUT_DIR
    / "geomorph_origin_250m.tif"
)

OUT_LULC = (
    OUT_DIR
    / "lulc_level1_250m.tif"
)

AUDIT = (
    WORK_DIR
    / "step64dF6_reconstruction_audit.csv"
)


# ================================================================
# BHUVAN WMS
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

WMS_SIZE = 3000

TIMEOUT = 180

RETRIES = 3

COLOR_TOLERANCE = 70.0

ALPHA_THRESHOLD = 80

TILE_SIZE_DEG = 1.0

NODATA = 0


# ================================================================
# CODES
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

    return gdf


# ================================================================
# LOAD CLASS PALETTE
# ================================================================

def load_palette(
    path
):

    df = pd.read_csv(
        path
    )

    required = [
        "class",
        "mode_R",
        "mode_G",
        "mode_B",
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    palette = []

    for _, row in df.iterrows():

        class_name = str(
            row["class"]
        )

        rgb = np.array(
            [
                float(row["mode_R"]),
                float(row["mode_G"]),
                float(row["mode_B"]),
            ],
            dtype=np.float32
        )

        palette.append(
            {
                "class":
                    class_name,

                "rgb":
                    rgb,
            }
        )

    return palette


# ================================================================
# GEOMORPH CLASS -> ORIGIN
# ================================================================

def geomorph_origin(
    class_name
):

    if class_name.startswith(
        "Structural Origin"
    ):
        return "Structural Origin"

    if class_name.startswith(
        "Fluvial Origin"
    ):
        return "Fluvial Origin"

    if class_name.startswith(
        "Denudational Origin"
    ):
        return "Denudational Origin"

    if class_name.startswith(
        "Glacial Origin"
    ):
        return "Glacial Origin"

    if class_name.startswith(
        "Lacustrine Origin"
    ):
        return "Lacustrine Origin"

    if class_name.startswith(
        "Water Bodies"
    ):
        return "Water Bodies"

    return None


# ================================================================
# BUILD CLASS PALETTE
# ================================================================

def build_geom_class_palette(
    palette
):

    result = []

    for item in palette:

        origin = geomorph_origin(
            item["class"]
        )

        if origin is None:

            continue

        code = (
            GEOM_ORIGIN_CODES[
                origin
            ]
        )

        result.append(
            {
                "class":
                    item["class"],

                "origin":
                    origin,

                "code":
                    code,

                "rgb":
                    item["rgb"],
            }
        )

    return result


def build_lulc_palette(
    palette
):

    result = []

    for item in palette:

        class_name = item[
            "class"
        ]

        if class_name not in (
            LULC_CODES
        ):

            continue

        result.append(
            {
                "class":
                    class_name,

                "code":
                    LULC_CODES[
                        class_name
                    ],

                "rgb":
                    item["rgb"],
            }
        )

    return result


# ================================================================
# TILE GENERATION
# ================================================================

def generate_tiles(
    bbox
):

    minx, miny, maxx, maxy = bbox

    tiles = []

    x = minx

    while x < maxx:

        x2 = min(
            x + TILE_SIZE_DEG,
            maxx
        )

        y = miny

        while y < maxy:

            y2 = min(
                y + TILE_SIZE_DEG,
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


# ================================================================
# WMS REQUEST
# ================================================================

def request_wms(
    layer,
    bbox
):

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
            ",".join(
                str(x)
                for x in bbox
            ),

        "WIDTH":
            WMS_SIZE,

        "HEIGHT":
            WMS_SIZE,

        "FORMAT":
            "image/png",

        "TRANSPARENT":
            "TRUE",
    }

    last_error = None

    for attempt in range(
        1,
        RETRIES + 1
    ):

        try:

            response = requests.get(
                WMS_URL,
                params=params,
                timeout=TIMEOUT
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

            if array.shape != (
                WMS_SIZE,
                WMS_SIZE,
                4
            ):

                raise RuntimeError(
                    f"Unexpected WMS shape "
                    f"{array.shape}"
                )

            return array

        except Exception as exc:

            last_error = exc

            print(
                f"      Retry "
                f"{attempt}/{RETRIES}: "
                f"{exc}"
            )

            time.sleep(3)

    raise RuntimeError(
        f"WMS failed: {last_error}"
    )


# ================================================================
# CLASSIFY WMS RGB
# ================================================================

def classify_geomorphology(
    rgba,
    palette
):

    rgb = rgba[
        :, :, :3
    ].astype(
        np.float32
    )

    alpha = rgba[
        :, :, 3
    ]

    output = np.zeros(
        rgb.shape[:2],
        dtype=np.uint16
    )

    # ------------------------------------------------------------
    # Valid image pixels
    # ------------------------------------------------------------

    valid = (
        alpha
        >= ALPHA_THRESHOLD
    )

    # Reject white/background
    white = np.all(
        rgb >= 245,
        axis=2
    )

    valid &= ~white

    # Reject black text/borders
    black = np.all(
        rgb <= 40,
        axis=2
    )

    valid &= ~black

    if not np.any(valid):

        return output

    pixels = rgb[
        valid
    ]

    palette_rgb = np.vstack(
        [
            p["rgb"]
            for p in palette
        ]
    )

    differences = (
        pixels[:, None, :]
        -
        palette_rgb[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=2
        )
    )

    nearest = np.argmin(
        distances,
        axis=1
    )

    nearest_distance = (
        distances[
            np.arange(
                len(pixels)
            ),
            nearest
        ]
    )

    accepted = (
        nearest_distance
        <= COLOR_TOLERANCE
    )

    valid_indices = np.flatnonzero(
        valid
    )

    output_flat = (
        output.reshape(-1)
    )

    output_flat[
        valid_indices[accepted]
    ] = np.array(
        [
            palette[i]["code"]
            for i in nearest[accepted]
        ],
        dtype=np.uint16
    )

    return output


def classify_lulc(
    rgba,
    palette
):

    rgb = rgba[
        :, :, :3
    ].astype(
        np.float32
    )

    alpha = rgba[
        :, :, 3
    ]

    output = np.zeros(
        rgb.shape[:2],
        dtype=np.uint16
    )

    valid = (
        alpha
        >= ALPHA_THRESHOLD
    )

    white = np.all(
        rgb >= 245,
        axis=2
    )

    valid &= ~white

    black = np.all(
        rgb <= 40,
        axis=2
    )

    valid &= ~black

    if not np.any(valid):

        return output

    pixels = rgb[
        valid
    ]

    palette_rgb = np.vstack(
        [
            p["rgb"]
            for p in palette
        ]
    )

    differences = (
        pixels[:, None, :]
        -
        palette_rgb[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=2
        )
    )

    nearest = np.argmin(
        distances,
        axis=1
    )

    nearest_distance = (
        distances[
            np.arange(
                len(pixels)
            ),
            nearest
        ]
    )

    accepted = (
        nearest_distance
        <= COLOR_TOLERANCE
    )

    valid_indices = np.flatnonzero(
        valid
    )

    output_flat = (
        output.reshape(-1)
    )

    output_flat[
        valid_indices[accepted]
    ] = np.array(
        [
            palette[i]["code"]
            for i in nearest[accepted]
        ],
        dtype=np.uint16
    )

    return output


# ================================================================
# WRITE RAW TILE
# ================================================================

def write_tile(
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
# REPROJECT TILE TO FINAL GRID
# ================================================================

def reproject_tile(
    path,
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
        path
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
# PROCESS FACTOR
# ================================================================

def process_factor(
    factor_name,
    layers,
    palette,
    boundary,
    grid
):

    mosaic = np.zeros(
        (
            grid["height"],
            grid["width"]
        ),
        dtype=np.uint16
    )

    audit_rows = []

    for state, layer in (
        layers.items()
    ):

        print(
            f"\n[{state}]"
        )

        state_geom = boundary[
            boundary[
                "state_norm"
            ]
            == state
        ]

        if state_geom.empty:

            print(
                "  State geometry missing."
            )

            continue

        if layer is None:

            print(
                "  Source unavailable."
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

        geometry = (
            state_geom.geometry
            .union_all()
        )

        bbox = geometry.bounds

        tiles = generate_tiles(
            bbox
        )

        print(
            "  Tiles:",
            len(tiles)
        )

        state_projected = (
            state_geom
            .to_crs(
                grid["crs"]
            )
        )

        state_mask = geometry_mask(

            state_projected.geometry,

            out_shape=(
                grid["height"],
                grid["width"]
            ),

            transform=
                grid["transform"],

            invert=True
        )

        for number, tile_bbox in enumerate(
            tiles,
            start=1
        ):

            print(
                f"  Tile "
                f"{number}/"
                f"{len(tiles)}"
            )

            try:

                rgba = request_wms(
                    layer,
                    tile_bbox
                )

                if factor_name == (
                    "geomorph_origin"
                ):

                    classified = (
                        classify_geomorphology(
                            rgba,
                            palette
                        )
                    )

                else:

                    classified = (
                        classify_lulc(
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

                tile_path = (
                    WORK_DIR
                    /
                    (
                        f"{factor_name}_"
                        f"{state.replace(' ', '_')}_"
                        f"{number}.tif"
                    )
                )

                write_tile(
                    tile_path,
                    classified,
                    tile_bbox
                )

                projected = (
                    reproject_tile(
                        tile_path,
                        grid
                    )
                )

                projected[
                    ~state_mask
                ] = NODATA

                valid = (
                    projected > 0
                )

                valid_count = int(
                    np.sum(valid)
                )

                fill = (
                    valid
                    &
                    (
                        mosaic
                        == NODATA
                    )
                )

                mosaic[
                    fill
                ] = projected[
                    fill
                ]

                audit_rows.append(
                    {
                        "factor":
                            factor_name,

                        "state":
                            state,

                        "tile":
                            number,

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
                            number,

                        "status":
                            "FAILED",

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

    return mosaic, audit_rows


# ================================================================
# WRITE FINAL
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
# MAIN
# ================================================================

def main():

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 64D-F6 — CLASS-LEVEL "
        "CATEGORICAL RECONSTRUCTION"
    )

    print(
        "=" * 70
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # GRID
    # ------------------------------------------------------------

    print(
        "\n[1] Loading final grid"
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

    # ------------------------------------------------------------
    # BOUNDARY
    # ------------------------------------------------------------

    print(
        "\n[2] Loading boundary"
    )

    boundary = load_boundary()

    ner_states = list(
        GEOM_LAYERS.keys()
    )

    boundary = boundary[
        boundary[
            "state_norm"
        ].isin(
            ner_states
        )
    ]

    print(
        "NER states:",
        len(boundary)
    )

    # ------------------------------------------------------------
    # PALETTES
    # ------------------------------------------------------------

    print(
        "\n[3] Loading class-level palettes"
    )

    geom_palette_raw = (
        load_palette(
            GEOM_PALETTE
        )
    )

    lulc_palette_raw = (
        load_palette(
            LULC_PALETTE
        )
    )

    geom_palette = (
        build_geom_class_palette(
            geom_palette_raw
        )
    )

    lulc_palette = (
        build_lulc_palette(
            lulc_palette_raw
        )
    )

    print(
        "Geomorphology class colors:",
        len(geom_palette)
    )

    print(
        "LUCC class colors:",
        len(lulc_palette)
    )

    # ------------------------------------------------------------
    # GEOMORPHOLOGY
    # ------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GEOMORPH_ORIGIN"
    )

    print(
        "=" * 70
    )

    geom_raster, geom_audit = (
        process_factor(

            "geomorph_origin",

            GEOM_LAYERS,

            geom_palette,

            boundary,

            grid
        )
    )

    # ------------------------------------------------------------
    # LUCC
    # ------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LULC_LEVEL1"
    )

    print(
        "=" * 70
    )

    lulc_raster, lulc_audit = (
        process_factor(

            "lulc_level1",

            LULC_LAYERS,

            lulc_palette,

            boundary,

            grid
        )
    )

    # ------------------------------------------------------------
    # FINAL NER MASK
    # ------------------------------------------------------------

    print(
        "\n[4] Applying NER mask"
    )

    with rasterio.open(
        MASK
    ) as src:

        ner_mask = src.read(
            1
        )

    valid_ner = (
        ner_mask > 0
    )

    geom_raster[
        ~valid_ner
    ] = NODATA

    lulc_raster[
        ~valid_ner
    ] = NODATA

    # ------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------

    print(
        "\n[5] Writing final rasters"
    )

    write_final(
        OUT_GEOM,
        geom_raster,
        grid
    )

    write_final(
        OUT_LULC,
        lulc_raster,
        grid
    )

    print(
        "Written:",
        OUT_GEOM
    )

    print(
        "Written:",
        OUT_LULC
    )

    # ------------------------------------------------------------
    # AUDIT
    # ------------------------------------------------------------

    audit = pd.DataFrame(
        geom_audit
        +
        lulc_audit
    )

    audit.to_csv(
        AUDIT,
        index=False
    )

    # ------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL STATISTICS"
    )

    print(
        "=" * 70
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
                f"  Code "
                f"{int(value)}: "
                f"{int(count):,}"
            )

    # ------------------------------------------------------------
    # PROPERTY CHECK
    # ------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RASTER PROPERTY CHECK"
    )

    print(
        "=" * 70
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
                "  NoData:",
                src.nodata
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "STEP 64D-F6 COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nNext:"
    )

    print(
        "Run step64d_validate_categorical_rasters.py"
    )


if __name__ == "__main__":

    main()