import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


# ============================================================
# FILES
# ============================================================

class_file = (
    "processed/susceptibility_class.tif"
)

points_file = (
    "processed/training_features_complete.geojson"
)

output_file = (
    "processed/susceptibility_map.png"
)


# ============================================================
# READ SUSCEPTIBILITY RASTER
# ============================================================

print("Reading susceptibility raster...")

with rasterio.open(class_file) as src:

    susceptibility = src.read(1)

    bounds = src.bounds

    crs = src.crs


print("CRS:", crs)
print("Raster size:", susceptibility.shape)


# ============================================================
# READ LANDSLIDE POINTS
# ============================================================

print("Reading training points...")

points = gpd.read_file(points_file)

points = points.to_crs(crs)

landslides = points[
    points["label"] == 1
]


print(
    "Landslide points:",
    len(landslides)
)


# ============================================================
# CREATE MASKED RASTER
# ============================================================

masked = susceptibility.astype(float)

masked[masked == 255] = float("nan")


# ============================================================
# CREATE MAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 10)
)


# Low / Moderate / High
cmap = ListedColormap([
    "green",
    "yellow",
    "red"
])


ax.imshow(
    masked,
    extent=[
        bounds.left,
        bounds.right,
        bounds.bottom,
        bounds.top
    ],
    origin="upper",
    cmap=cmap,
    vmin=1,
    vmax=3,
    interpolation="nearest"
)


# ============================================================
# PLOT LANDSLIDE LOCATIONS
# ============================================================

landslides.plot(
    ax=ax,
    marker="*",
    markersize=80,
    color="blue",
    edgecolor="white",
    linewidth=0.8,
    label="Known landslide"
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Patch(
        facecolor="green",
        label="Low susceptibility"
    ),

    Patch(
        facecolor="yellow",
        label="Moderate susceptibility"
    ),

    Patch(
        facecolor="red",
        label="High susceptibility"
    )
]


ax.legend(
    handles=legend_elements,
    loc="upper right",
    title="Susceptibility"
)


# ============================================================
# LABELS
# ============================================================

ax.set_title(
    "Giri Rakshak — Landslide Susceptibility Map",
    fontsize=16,
    fontweight="bold"
)

ax.set_xlabel(
    "Longitude"
)

ax.set_ylabel(
    "Latitude"
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    linestyle="--",
    alpha=0.4
)


# ============================================================
# SAVE
# ============================================================

plt.tight_layout()

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n========================================")
print("STEP 28 COMPLETE")
print("========================================")

print(
    "Saved:",
    output_file
)