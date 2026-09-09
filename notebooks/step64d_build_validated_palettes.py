"""
STEP 64D-D — BUILD VALIDATED BHUVAN COLOR PALETTES

Purpose
-------
Build robust RGB color palettes from the training-point
validation results.

Because WMS rendering introduces anti-aliasing and
transparency, exact RGB equality is NOT used.

Instead:
    1. collect observed RGB colors
    2. group them by known Bhuvan class
    3. identify representative colors
    4. calculate class/color confidence
    5. create palettes for later raster classification

This script does NOT create final categorical rasters.
"""


from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

PROCESSED = (
    PROJECT
    / "processed"
)


# ============================================================
# INPUTS
# ============================================================

GEOMORPH_INPUT = (
    PROCESSED
    / "step64d_geomorphology_color_validation.csv"
)

LUCC_INPUT = (
    PROCESSED
    / "step64d_lulc_color_validation.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUT_DIR = (
    PROCESSED
    / "categorical_palette_validation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print()
print("=" * 70)
print(
    "STEP 64D-D — BUILD VALIDATED BHUVAN PALETTES"
)
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

print()
print("[1] Loading validation data")


geomorph = pd.read_csv(
    GEOMORPH_INPUT
)

lulc = pd.read_csv(
    LUCC_INPUT
)


print(
    f"Geomorphology observations: "
    f"{len(geomorph)}"
)

print(
    f"LUCC observations: "
    f"{len(lulc)}"
)


# ============================================================
# RGB DISTANCE
# ============================================================

def rgb_distance(
    rgb1,
    rgb2
):

    return float(
        np.sqrt(
            np.sum(
                (
                    np.array(rgb1)
                    - np.array(rgb2)
                )
                ** 2
            )
        )
    )


# ============================================================
# REMOVE BACKGROUND
# ============================================================

def prepare(
    data,
    class_column
):

    data = data.copy()

    # --------------------------------------------------------
    # Valid alpha
    # --------------------------------------------------------

    data = data[
        data["A"] > 200
    ].copy()

    # --------------------------------------------------------
    # Remove white background
    # --------------------------------------------------------

    data = data[
        ~(
            (data["R"] >= 250)
            &
            (data["G"] >= 250)
            &
            (data["B"] >= 250)
        )
    ].copy()

    # --------------------------------------------------------
    # Remove black text/border pixels
    #
    # Black pixels are frequently map labels/text rather
    # than categorical fill colors.
    # --------------------------------------------------------

    data = data[
        ~(
            (data["R"] <= 10)
            &
            (data["G"] <= 10)
            &
            (data["B"] <= 10)
        )
    ].copy()

    # --------------------------------------------------------
    # Require known class
    # --------------------------------------------------------

    data = data[
        data[class_column]
        .notna()
    ].copy()

    return data


# ============================================================
# REPRESENTATIVE COLORS
# ============================================================

def build_palette(
    data,
    class_column,
    factor_name
):

    data = prepare(
        data,
        class_column
    )

    print()
    print(
        f"{factor_name}: usable observations = "
        f"{len(data)}"
    )

    classes = (
        data[class_column]
        .dropna()
        .unique()
        .tolist()
    )

    rows = []

    for class_name in classes:

        subset = data[
            data[class_column]
            == class_name
        ].copy()

        if subset.empty:
            continue

        # ----------------------------------------------------
        # Count exact RGB values
        # ----------------------------------------------------

        colors = (
            subset
            .groupby(
                [
                    "R",
                    "G",
                    "B"
                ]
            )
            .size()
            .reset_index(
                name="count"
            )
            .sort_values(
                "count",
                ascending=False
            )
        )

        # ----------------------------------------------------
        # Calculate weighted mean RGB
        # ----------------------------------------------------

        weights = (
            colors["count"]
            .values
        )

        rgb_values = (
            colors[
                [
                    "R",
                    "G",
                    "B"
                ]
            ]
            .values
        )

        weighted_mean = (
            np.average(
                rgb_values,
                axis=0,
                weights=weights
            )
        )

        # ----------------------------------------------------
        # Most common exact color
        # ----------------------------------------------------

        mode_rgb = (
            colors.iloc[0]
            [
                [
                    "R",
                    "G",
                    "B"
                ]
            ]
            .astype(int)
            .tolist()
        )

        # ----------------------------------------------------
        # Median RGB
        # ----------------------------------------------------

        median_rgb = (
            subset[
                [
                    "R",
                    "G",
                    "B"
                ]
            ]
            .median()
            .round()
            .astype(int)
            .tolist()
        )

        # ----------------------------------------------------
        # Save representatives
        # ----------------------------------------------------

        rows.append({

            "factor":
                factor_name,

            "class":
                class_name,

            "observations":
                len(subset),

            "unique_rgb_values":
                len(colors),

            "mode_R":
                mode_rgb[0],

            "mode_G":
                mode_rgb[1],

            "mode_B":
                mode_rgb[2],

            "median_R":
                median_rgb[0],

            "median_G":
                median_rgb[1],

            "median_B":
                median_rgb[2],

            "mean_R":
                round(
                    weighted_mean[0],
                    3
                ),

            "mean_G":
                round(
                    weighted_mean[1],
                    3
                ),

            "mean_B":
                round(
                    weighted_mean[2],
                    3
                )

        })

    palette = pd.DataFrame(
        rows
    )

    return palette


# ============================================================
# GEOMORPHOLOGY PALETTE
# ============================================================

print()
print("=" * 70)

print(
    "[2] Building geomorphology palette"
)

print(
    "=" * 70
)


geomorph_palette = build_palette(
    geomorph,
    "geomorphology",
    "geomorphology"
)


geomorph_palette_file = (
    OUT_DIR
    / "geomorphology_validated_palette.csv"
)


geomorph_palette.to_csv(
    geomorph_palette_file,
    index=False
)


print()
print(
    "Geomorphology palette:"
)

print(
    geomorph_palette.to_string(
        index=False
    )
)


# ============================================================
# LUCC PALETTE
# ============================================================

print()
print("=" * 70)

print(
    "[3] Building LUCC palette"
)

print(
    "=" * 70
)


lulc_palette = build_palette(
    lulc,
    "Level_I",
    "lulc"
)


lulc_palette_file = (
    OUT_DIR
    / "lulc_validated_palette.csv"
)


lulc_palette.to_csv(
    lulc_palette_file,
    index=False
)


print()
print(
    "LUCC palette:"
)

print(
    lulc_palette.to_string(
        index=False
    )
)


# ============================================================
# GEOMORPH ORIGIN GROUPING
# ============================================================

print()
print("=" * 70)

print(
    "[4] Deriving geomorph_origin"
)

print(
    "=" * 70
)


def get_origin(
    value
):

    if pd.isna(value):
        return np.nan

    value = str(
        value
    )

    prefixes = [

        "Structural Origin",

        "Fluvial Origin",

        "Denudational Origin",

        "Glacial Origin",

        "Water Bodies",

        "Lacustrine Origin"

    ]

    for prefix in prefixes:

        if value.startswith(
            prefix
        ):

            return prefix

    return np.nan


geomorph_palette[
    "geomorph_origin"
] = (
    geomorph_palette[
        "class"
    ]
    .apply(
        get_origin
    )
)


origin_palette = (
    geomorph_palette
    .groupby(
        "geomorph_origin"
    )
    .agg(
        observations=(
            "observations",
            "sum"
        )
    )
    .reset_index()
)


origin_file = (
    OUT_DIR
    / "geomorph_origin_class_summary.csv"
)


origin_palette.to_csv(
    origin_file,
    index=False
)


print(
    origin_palette.to_string(
        index=False
    )
)


# ============================================================
# REQUIRED FINAL MODEL CATEGORIES
# ============================================================

print()
print("=" * 70)

print(
    "[5] Checking required modeling categories"
)

print(
    "=" * 70
)


required_origins = {

    "Structural Origin",

    "Fluvial Origin",

    "Denudational Origin",

    "Glacial Origin",

    "Water Bodies",

    "Lacustrine Origin"

}


available_origins = set(
    origin_palette[
        "geomorph_origin"
    ]
    .dropna()
)


print()
print(
    "Required geomorphology origins:"
)

for origin in sorted(
    required_origins
):

    status = (
        "AVAILABLE"
        if origin in available_origins
        else "MISSING"
    )

    print(
        f"{origin:30s} "
        f"{status}"
    )


required_lulc = {

    "Forest",

    "Built-up",

    "Agriculture",

    "Wastelands",

    "Water Bodies",

    "Others",

    "Grasslands / Grazing Lands"

}


available_lulc = set(
    lulc_palette[
        "class"
    ]
    .dropna()
)


print()
print(
    "Required LUCC Level_I classes:"
)

for cls in sorted(
    required_lulc
):

    status = (
        "AVAILABLE"
        if cls in available_lulc
        else "MISSING"
    )

    print(
        f"{cls:30s} "
        f"{status}"
    )


# ============================================================
# SAVE MASTER PALETTE SUMMARY
# ============================================================

summary = pd.concat(
    [
        geomorph_palette[
            [
                "factor",
                "class",
                "observations",
                "mode_R",
                "mode_G",
                "mode_B",
                "median_R",
                "median_G",
                "median_B",
                "mean_R",
                "mean_G",
                "mean_B"
            ]
        ],
        lulc_palette[
            [
                "factor",
                "class",
                "observations",
                "mode_R",
                "mode_G",
                "mode_B",
                "median_R",
                "median_G",
                "median_B",
                "mean_R",
                "mean_G",
                "mean_B"
            ]
        ]
    ],
    ignore_index=True
)


summary_file = (
    PROCESSED
    / "step64d_validated_color_palettes.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)

print(
    "STEP 64D-D COMPLETED"
)

print(
    "=" * 70
)

print()
print(
    "Outputs:"
)

print(
    geomorph_palette_file
)

print(
    lulc_palette_file
)

print(
    origin_file
)

print(
    summary_file
)

print()
print(
    "IMPORTANT:"
)

print(
    "These palettes are validated from training-point "
    "observations."
)

print(
    "They are NOT yet used to classify the full NER."
)

print()
print(
    "Next step: spatially reconstruct the categorical "
    "rasters and validate them at the training points."
)