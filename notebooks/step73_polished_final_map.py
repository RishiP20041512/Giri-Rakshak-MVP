import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RASTER = os.path.join(
    BASE, "processed",
    "step71_final_8factor_susceptibility.tif"
)

BOUNDARY = os.path.join(
    BASE, "raw_data", "boundaries",
    "ner_8states.geojson"
)

OUTPUT = os.path.join(
    BASE, "processed",
    "step73_FINAL_JUDGE_MAP.png"
)

with rasterio.open(RASTER) as src:
    data = src.read(1)
    transform = src.transform
    crs = src.crs
    nodata = src.nodata

states = gpd.read_file(BOUNDARY).to_crs(crs)

valid = (
    np.isfinite(data) &
    (data != nodata) &
    (data >= 0) &
    (data <= 1)
)

# ------------------------------------------------------------
# CLASSIFICATION
# ------------------------------------------------------------

classes = np.zeros_like(data, dtype=np.uint8)

classes[valid & (data < 0.20)] = 1
classes[valid & (data >= 0.20) & (data < 0.40)] = 2
classes[valid & (data >= 0.40) & (data < 0.60)] = 3
classes[valid & (data >= 0.60)] = 4

display = classes.astype(float)
display[classes == 0] = np.nan

# ------------------------------------------------------------
# EXTENT
# ------------------------------------------------------------

left = transform.c
top = transform.f
right = left + data.shape[1] * transform.a
bottom = top + data.shape[0] * transform.e

extent = [left, right, bottom, top]

# ------------------------------------------------------------
# MAP
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(14, 11), dpi=150)

cmap = ListedColormap([
    "#2ca25f",
    "#fee08b",
    "#f46d43",
    "#d73027"
])

norm = BoundaryNorm(
    [0.5, 1.5, 2.5, 3.5, 4.5],
    4
)

ax.imshow(
    display,
    extent=extent,
    origin="upper",
    cmap=cmap,
    norm=norm,
    interpolation="nearest"
)

# State boundaries
states.boundary.plot(
    ax=ax,
    color="black",
    linewidth=0.8
)

# ------------------------------------------------------------
# STATE LABELS
# ------------------------------------------------------------

name_col = "shapeName"

for _, row in states.iterrows():

    if row.geometry is None or row.geometry.is_empty:
        continue

    p = row.geometry.representative_point()

    name = str(row[name_col])

    # Remove accents for cleaner presentation
    name = (
        name.replace("ā", "a")
            .replace("ī", "i")
            .replace("ṅ", "n")
    )

    ax.text(
        p.x,
        p.y,
        name,
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            facecolor="white",
            alpha=0.65,
            edgecolor="none",
            pad=1.5
        )
    )

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

ax.set_title(
    "North-East India Landslide Susceptibility",
    fontsize=21,
    fontweight="bold",
    pad=20
)

ax.text(
    0.5,
    1.015,
    "8-Factor Random Forest Model • 250 m Spatial Resolution",
    transform=ax.transAxes,
    ha="center",
    fontsize=11
)

# ------------------------------------------------------------
# LEGEND
# ------------------------------------------------------------

legend = [
    Patch(
        facecolor="#2ca25f",
        edgecolor="black",
        label="Low  (<0.20)"
    ),
    Patch(
        facecolor="#fee08b",
        edgecolor="black",
        label="Moderate  (0.20–0.39)"
    ),
    Patch(
        facecolor="#f46d43",
        edgecolor="black",
        label="High  (0.40–0.59)"
    ),
    Patch(
        facecolor="#d73027",
        edgecolor="black",
        label="Very High  (≥0.60)"
    )
]

ax.legend(
    handles=legend,
    title="Susceptibility Probability",
    loc="lower left",
    fontsize=10,
    title_fontsize=11,
    framealpha=0.95
)

# ------------------------------------------------------------
# NORTH ARROW
# ------------------------------------------------------------

ax.annotate(
    "N",
    xy=(0.95, 0.91),
    xycoords="axes fraction",
    ha="center",
    fontsize=16,
    fontweight="bold"
)

ax.annotate(
    "",
    xy=(0.95, 0.88),
    xytext=(0.95, 0.80),
    xycoords="axes fraction",
    arrowprops=dict(
        width=3,
        headwidth=12
    )
)

# ------------------------------------------------------------
# AXES
# ------------------------------------------------------------

ax.set_xlabel("Easting (m) — EPSG:6933", fontsize=9)
ax.set_ylabel("Northing (m) — EPSG:6933", fontsize=9)

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

fig.text(
    0.5,
    0.025,
    "Sources: Copernicus DEM • CHIRPS • NASA SMAP • Sentinel-2 • "
    "GRIP4 • Bhuvan | Random Forest susceptibility model",
    ha="center",
    fontsize=8
)

fig.text(
    0.5,
    0.008,
    "White areas indicate NoData where required real predictor coverage was unavailable.",
    ha="center",
    fontsize=8
)

plt.tight_layout(rect=[0, 0.045, 1, 0.97])

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.close()

print("=" * 70)
print("STEP 73 COMPLETE")
print("=" * 70)
print("\nFINAL JUDGE MAP:")
print(OUTPUT)