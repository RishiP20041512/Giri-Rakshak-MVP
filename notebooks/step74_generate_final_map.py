from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import rasterio
import geopandas as gpd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "processed"
BOUNDARIES = ROOT / "raw_data" / "boundaries"

CLASS_RASTER = (
    PROCESSED /
    "step71_8factor_susceptibility_classes.tif"
)

BOUNDARY_FILE = (
    BOUNDARIES /
    "geoBoundaries-IND-ADM1.geojson"
)

OUTPUT_PNG = (
    PROCESSED /
    "step74_final_ner_susceptibility_map.png"
)

OUTPUT_PDF = (
    PROCESSED /
    "step74_final_ner_susceptibility_map.pdf"
)


# ============================================================
# SETTINGS
# ============================================================

CLASS_NAMES = {
    1: "Very Low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very High",
}

# Conventional susceptibility colour progression
CLASS_COLORS = [
    "#2ca25f",   # Very Low
    "#99d8c9",   # Low
    "#fee391",   # Moderate
    "#fe9929",   # High
    "#de2d26",   # Very High
]


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 74 — GENERATE FINAL NER SUSCEPTIBILITY MAP")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

if not CLASS_RASTER.exists():
    raise FileNotFoundError(
        f"Class raster not found:\n{CLASS_RASTER}"
    )

if not BOUNDARY_FILE.exists():
    raise FileNotFoundError(
        f"Boundary file not found:\n{BOUNDARY_FILE}"
    )


# ============================================================
# READ SUSCEPTIBILITY RASTER
# ============================================================

print("\nReading susceptibility raster...")

with rasterio.open(CLASS_RASTER) as src:

    raster = src.read(1)

    raster_crs = src.crs
    transform = src.transform

    bounds = src.bounds

    nodata = src.nodata

    width = src.width
    height = src.height

print(f"Raster size : {width} x {height}")
print(f"CRS         : {raster_crs}")
print(f"NoData      : {nodata}")


# ============================================================
# CREATE MASK
# ============================================================

valid = np.isin(
    raster,
    [1, 2, 3, 4, 5]
)

display_raster = np.ma.masked_where(
    ~valid,
    raster
)


# ============================================================
# READ NER BOUNDARIES
# ============================================================

print("\nReading Northeast India boundaries...")

gdf = gpd.read_file(
    BOUNDARY_FILE
)

print(
    f"Boundary features: {len(gdf)}"
)

print(
    f"Boundary CRS: {gdf.crs}"
)


# ============================================================
# REPROJECT BOUNDARIES
# ============================================================

if gdf.crs != raster_crs:

    print(
        "Reprojecting boundaries to raster CRS..."
    )

    gdf = gdf.to_crs(
        raster_crs
    )


# ============================================================
# RASTER EXTENT
# ============================================================

extent = [
    bounds.left,
    bounds.right,
    bounds.bottom,
    bounds.top,
]


# ============================================================
# CREATE FIGURE
# ============================================================

print("\nCreating map...")

fig, ax = plt.subplots(
    figsize=(12, 10),
    dpi=200
)


# ============================================================
# SUSCEPTIBILITY MAP
# ============================================================

cmap = ListedColormap(
    CLASS_COLORS
)

cmap.set_bad(
    alpha=0
)

ax.imshow(
    display_raster,
    cmap=cmap,
    interpolation="nearest",
    extent=extent,
    origin="upper",
    vmin=1,
    vmax=5,
)


# ============================================================
# STATE BOUNDARIES
# ============================================================

gdf.boundary.plot(
    ax=ax,
    linewidth=0.5,
    edgecolor="black",
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Landslide Susceptibility Map of Northeast India",
    fontsize=18,
    fontweight="bold",
    pad=15,
)

ax.text(
    0.5,
    1.01,
    "8-Factor Random Forest Model | 250 m spatial resolution",
    transform=ax.transAxes,
    ha="center",
    fontsize=11,
)


# ============================================================
# LEGEND
# ============================================================

legend_handles = []

for class_id in range(1, 6):

    legend_handles.append(
        Patch(
            facecolor=CLASS_COLORS[class_id - 1],
            edgecolor="black",
            label=(
                f"{class_id} — "
                f"{CLASS_NAMES[class_id]}"
            ),
        )
    )


legend = ax.legend(
    handles=legend_handles,
    title="Susceptibility",
    loc="lower left",
    frameon=True,
    fontsize=10,
    title_fontsize=11,
)


# ============================================================
# NORTH ARROW
# ============================================================

ax.annotate(
    "N",
    xy=(0.94, 0.91),
    xytext=(0.94, 0.79),
    xycoords="axes fraction",
    textcoords="axes fraction",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=1.5,
    ),
)


# ============================================================
# SCALE BAR
# ============================================================

# Approximately 200 km scale bar.
# EPSG:6933 is metre-based.

x_min = bounds.left
x_max = bounds.right

y_min = bounds.bottom
y_max = bounds.top

scale_length_m = 200_000

scale_x = (
    x_min +
    0.06 * (x_max - x_min)
)

scale_y = (
    y_min +
    0.05 * (y_max - y_min)
)

ax.plot(
    [
        scale_x,
        scale_x + scale_length_m
    ],
    [
        scale_y,
        scale_y
    ],
    linewidth=4,
    color="black",
)

ax.plot(
    [
        scale_x,
        scale_x
    ],
    [
        scale_y - 3000,
        scale_y + 3000
    ],
    linewidth=2,
    color="black",
)

ax.plot(
    [
        scale_x + scale_length_m,
        scale_x + scale_length_m
    ],
    [
        scale_y - 3000,
        scale_y + 3000
    ],
    linewidth=2,
    color="black",
)

ax.text(
    scale_x +
    scale_length_m / 2,
    scale_y +
    10000,
    "200 km",
    ha="center",
    va="bottom",
    fontsize=10,
    fontweight="bold",
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Easting (m)",
    fontsize=10,
)

ax.set_ylabel(
    "Northing (m)",
    fontsize=10,
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    linewidth=0.3,
    alpha=0.4,
)


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.5,
    0.015,
    (
        "Factors: Elevation, Slope, Rainfall, Soil Moisture, "
        "NDVI, Distance to Road, Lineament Density and "
        "Geomorphological Origin. "
        "NoData indicates unavailable predictor information."
    ),
    ha="center",
    fontsize=8,
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[0, 0.035, 1, 1]
)


# ============================================================
# SAVE PNG
# ============================================================

print("\nSaving PNG...")

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# SAVE PDF
# ============================================================

print("Saving PDF...")

plt.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
)


# ============================================================
# CLOSE
# ============================================================

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("STEP 74 COMPLETE")
print("=" * 70)

print("\nPNG map:")
print(OUTPUT_PNG)

print("\nPDF map:")
print(OUTPUT_PDF)

print("\nThe final susceptibility map has been generated.")

print("=" * 70)