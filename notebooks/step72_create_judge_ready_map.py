import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


# ============================================================
# STEP 72 — JUDGE-READY FINAL SUSCEPTIBILITY MAP
# ============================================================

print("=" * 75)
print("STEP 72 — JUDGE-READY 8-FACTOR SUSCEPTIBILITY MAP")
print("=" * 75)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_TIF = os.path.join(
    BASE,
    "processed",
    "step71_final_8factor_susceptibility.tif"
)

BOUNDARY = os.path.join(
    BASE,
    "raw_data",
    "boundaries",
    "ner_8states.geojson"
)

OUTPUT_CLASSIFIED = os.path.join(
    BASE,
    "processed",
    "step72_final_susceptibility_classes.tif"
)

OUTPUT_PNG = os.path.join(
    BASE,
    "processed",
    "step72_judge_ready_ner_susceptibility_map.png"
)


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking input files...")

if not os.path.exists(INPUT_TIF):
    raise FileNotFoundError(
        f"Susceptibility raster not found:\n{INPUT_TIF}"
    )

if not os.path.exists(BOUNDARY):
    raise FileNotFoundError(
        f"NER boundary not found:\n{BOUNDARY}"
    )

print("OK: susceptibility raster")
print("OK: NER 8-state boundary")


# ============================================================
# LOAD SUSCEPTIBILITY
# ============================================================

print("\nLoading susceptibility raster...")

with rasterio.open(INPUT_TIF) as src:

    susceptibility = src.read(1)

    profile = src.profile.copy()

    transform = src.transform
    crs = src.crs
    width = src.width
    height = src.height
    nodata = src.nodata

print(f"CRS        : {crs}")
print(f"Dimensions : {width} x {height}")
print(f"Resolution : {transform.a} m")


# ============================================================
# VALID MASK
# ============================================================

valid = (
    np.isfinite(susceptibility) &
    (susceptibility != nodata) &
    (susceptibility >= 0) &
    (susceptibility <= 1)
)

values = susceptibility[valid]

print("\nSusceptibility statistics:")
print(f"Valid cells : {values.size:,}")
print(f"Minimum     : {values.min():.6f}")
print(f"Maximum     : {values.max():.6f}")
print(f"Mean        : {values.mean():.6f}")
print(f"Median      : {np.median(values):.6f}")


# ============================================================
# CLASSIFICATION
# ============================================================
#
# Probability-based classes:
#
# 0.00 - 0.20  LOW
# 0.20 - 0.40  MODERATE
# 0.40 - 0.60  HIGH
# 0.60 - 1.00  VERY HIGH
#
# These are presentation classes, NOT retraining thresholds.
# ============================================================

classes = np.zeros_like(susceptibility, dtype=np.uint8)

classes[valid & (susceptibility < 0.20)] = 1
classes[valid & (susceptibility >= 0.20) &
        (susceptibility < 0.40)] = 2
classes[valid & (susceptibility >= 0.40) &
        (susceptibility < 0.60)] = 3
classes[valid & (susceptibility >= 0.60)] = 4

classes[~valid] = 0


# ============================================================
# CLASS COUNTS
# ============================================================

print("\nSusceptibility classes:")

class_names = {
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very High"
}

for code, name in class_names.items():

    count = np.sum(classes == code)

    percentage = (
        count / values.size * 100
        if values.size > 0 else 0
    )

    print(
        f"{name:12s}: "
        f"{count:,} cells "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SAVE CLASSIFIED GEOTIFF
# ============================================================

print("\nSaving classified GeoTIFF...")

profile.update(
    dtype="uint8",
    count=1,
    nodata=0,
    compress="deflate"
)

with rasterio.open(
    OUTPUT_CLASSIFIED,
    "w",
    **profile
) as dst:

    dst.write(classes, 1)

print("Saved:")
print(OUTPUT_CLASSIFIED)


# ============================================================
# LOAD STATE BOUNDARIES
# ============================================================

print("\nLoading NER state boundaries...")

states = gpd.read_file(BOUNDARY)

print(f"States loaded: {len(states)}")

# Convert to raster CRS
states = states.to_crs(crs)


# ============================================================
# GET RASTER EXTENT
# ============================================================

left = transform.c
top = transform.f
right = left + width * transform.a
bottom = top + height * transform.e

extent = [
    left,
    right,
    bottom,
    top
]


# ============================================================
# PREPARE DISPLAY ARRAY
# ============================================================

display_classes = classes.astype(float)

display_classes[classes == 0] = np.nan


# ============================================================
# CREATE FIGURE
# ============================================================

print("\nCreating presentation map...")

fig, ax = plt.subplots(
    figsize=(14, 11),
    dpi=150
)


# ============================================================
# CLASS COLORS
# ============================================================

cmap = ListedColormap([
    "#2ca25f",   # Low
    "#fee08b",   # Moderate
    "#f46d43",   # High
    "#d73027"    # Very High
])

norm = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5, 4.5],
    cmap.N
)


# ============================================================
# DRAW SUSCEPTIBILITY
# ============================================================

ax.imshow(
    display_classes,
    extent=extent,
    origin="upper",
    cmap=cmap,
    norm=norm,
    interpolation="nearest"
)


# ============================================================
# DRAW STATE BOUNDARIES
# ============================================================

states.boundary.plot(
    ax=ax,
    linewidth=0.8,
    edgecolor="black"
)


# ============================================================
# LABEL STATES
# ============================================================

name_column = None

possible_names = [
    "shapeName",
    "NAME_1",
    "name",
    "NAME",
    "State",
    "state"
]

for col in possible_names:

    if col in states.columns:
        name_column = col
        break


if name_column is not None:

    for _, row in states.iterrows():

        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        point = geom.representative_point()

        ax.text(
            point.x,
            point.y,
            str(row[name_column]),
            fontsize=7,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(
                facecolor="white",
                alpha=0.65,
                edgecolor="none",
                pad=1.5
            )
        )


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "North-East India Landslide Susceptibility Map",
    fontsize=20,
    fontweight="bold",
    pad=18
)

ax.text(
    0.5,
    1.01,
    "8-Factor Random Forest Model | 250 m Grid",
    transform=ax.transAxes,
    ha="center",
    fontsize=11
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Patch(
        facecolor="#2ca25f",
        edgecolor="black",
        label="Low (0.00–0.20)"
    ),

    Patch(
        facecolor="#fee08b",
        edgecolor="black",
        label="Moderate (0.20–0.40)"
    ),

    Patch(
        facecolor="#f46d43",
        edgecolor="black",
        label="High (0.40–0.60)"
    ),

    Patch(
        facecolor="#d73027",
        edgecolor="black",
        label="Very High (0.60–1.00)"
    )
]

ax.legend(
    handles=legend_elements,
    title="Susceptibility",
    loc="lower left",
    frameon=True,
    fontsize=10,
    title_fontsize=11
)


# ============================================================
# NORTH ARROW
# ============================================================

ax.annotate(
    "N",
    xy=(0.95, 0.90),
    xycoords="axes fraction",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold"
)

ax.annotate(
    "",
    xy=(0.95, 0.875),
    xytext=(0.95, 0.80),
    xycoords="axes fraction",
    arrowprops=dict(
        facecolor="black",
        edgecolor="black",
        width=3,
        headwidth=12
    )
)


# ============================================================
# AXIS
# ============================================================

ax.set_xlabel(
    "Projected Easting (m) — EPSG:6933",
    fontsize=10
)

ax.set_ylabel(
    "Projected Northing (m) — EPSG:6933",
    fontsize=10
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    True,
    linestyle="--",
    linewidth=0.4,
    alpha=0.4
)


# ============================================================
# FOOTNOTE
# ============================================================

fig.text(
    0.5,
    0.015,
    "Source: Copernicus DEM, CHIRPS, NASA SMAP, Sentinel-2, "
    "GRIP4, Bhuvan | Model: Random Forest (8 factors)",
    ha="center",
    fontsize=8
)


# ============================================================
# SAVE PNG
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 75)
print("STEP 72 COMPLETE")
print("=" * 75)

print("\nFinal outputs:")

print(
    f"\nClassified GeoTIFF:\n"
    f"{OUTPUT_CLASSIFIED}"
)

print(
    f"\nJudge-ready PNG:\n"
    f"{OUTPUT_PNG}"
)

print("\nNoData areas were preserved.")
print("No synthetic predictor values were added.")