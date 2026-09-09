"""
STEP 64D-B — INSPECT BHUVAN CATEGORICAL LEGENDS

Purpose
-------
Read the downloaded Bhuvan legend PNG files and identify
the dominant non-background colors.

This is a diagnostic step only.

It does NOT create final categorical rasters.
"""

from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd


# ============================================================
# PROJECT
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

DIAG_DIR = (
    PROJECT
    / "raw_data"
    / "categorical_diagnostic"
)

OUT_DIR = (
    PROJECT
    / "processed"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FILES
# ============================================================

legend_files = sorted(
    DIAG_DIR.glob("*_legend.png")
)

print()
print("=" * 70)
print(
    "STEP 64D-B — BHUVAN LEGEND INSPECTION"
)
print("=" * 70)

print()
print(
    f"Diagnostic directory:"
)

print(
    DIAG_DIR
)

print()
print(
    f"Legend files found: {len(legend_files)}"
)

for path in legend_files:

    print(
        "  ",
        path.name
    )


# ============================================================
# COLOR ANALYSIS
# ============================================================

all_rows = []


def analyze_image(path):

    print()
    print(
        "-" * 70
    )

    print(
        f"Analyzing: {path.name}"
    )

    image = Image.open(path).convert(
        "RGBA"
    )

    arr = np.array(
        image
    )

    height, width = arr.shape[:2]

    print(
        f"Image size: {width} x {height}"
    )

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    pixels = arr.reshape(
        -1,
        4
    )

    df = pd.DataFrame(
        pixels,
        columns=[
            "R",
            "G",
            "B",
            "A"
        ]
    )

    # --------------------------------------------------------
    # Remove fully transparent pixels
    # --------------------------------------------------------

    df = df[
        df["A"] > 0
    ].copy()

    print(
        f"Non-transparent pixels: {len(df):,}"
    )

    if len(df) == 0:

        print(
            "No non-transparent pixels."
        )

        return

    # --------------------------------------------------------
    # Count RGBA colors
    # --------------------------------------------------------

    counts = (
        df
        .groupby(
            [
                "R",
                "G",
                "B",
                "A"
            ]
        )
        .size()
        .reset_index(
            name="pixel_count"
        )
        .sort_values(
            "pixel_count",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Percentage
    # --------------------------------------------------------

    counts["percent"] = (
        counts["pixel_count"]
        / len(df)
        * 100
    )

    # --------------------------------------------------------
    # Keep reasonably common colors
    #
    # We keep the top 100 colors because legends can contain
    # anti-aliased text and borders.
    # --------------------------------------------------------

    counts = counts.head(
        100
    ).copy()

    counts.insert(
        0,
        "file",
        path.name
    )

    counts.insert(
        1,
        "rank",
        range(
            1,
            len(counts) + 1
        )
    )

    all_rows.append(
        counts
    )

    # --------------------------------------------------------
    # Print top colors
    # --------------------------------------------------------

    print()
    print(
        "Top colors:"
    )

    print()

    for _, row in counts.head(20).iterrows():

        print(
            f"#{int(row['R']):02X}"
            f"{int(row['G']):02X}"
            f"{int(row['B']):02X}"
            f"{int(row['A']):02X}"
            f"  "
            f"{int(row['pixel_count']):8d} pixels  "
            f"{row['percent']:7.3f}%"
        )


# ============================================================
# PROCESS ALL LEGENDS
# ============================================================

for path in legend_files:

    try:

        analyze_image(
            path
        )

    except Exception as e:

        print()
        print(
            f"ERROR processing {path.name}:"
        )

        print(
            e
        )


# ============================================================
# SAVE AUDIT TABLE
# ============================================================

print()
print("=" * 70)
print(
    "SAVING COLOR AUDIT"
)
print("=" * 70)


if all_rows:

    color_audit = pd.concat(
        all_rows,
        ignore_index=True
    )

else:

    color_audit = pd.DataFrame(
        columns=[
            "file",
            "rank",
            "R",
            "G",
            "B",
            "A",
            "pixel_count",
            "percent"
        ]
    )


output = (
    OUT_DIR
    / "step64d_bhuvan_legend_color_audit.csv"
)

color_audit.to_csv(
    output,
    index=False
)

print()
print(
    f"Saved:"
)

print(
    output
)


# ============================================================
# SUMMARY BY FILE
# ============================================================

print()
print(
    "=" * 70
)

print(
    "LEGEND SUMMARY"
)

print(
    "=" * 70
)

if not color_audit.empty:

    summary = (
        color_audit
        .groupby("file")
        .size()
        .reset_index(
            name="colors_recorded"
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print(
    "=" * 70
)

print(
    "STEP 64D-B COMPLETED"
)

print(
    "=" * 70
)

print()
print(
    "IMPORTANT:"
)

print(
    "The detected colors are diagnostic only."
)

print(
    "Do NOT assign class numbers based only on color."
)

print()
print(
    "Next we will connect the rendered colors "
    "to actual Bhuvan categorical class names "
    "and validate them against the existing "
    "training-point GetFeatureInfo results."
)