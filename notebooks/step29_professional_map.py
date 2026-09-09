import json
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from shapely.geometry import LineString, Point


# ============================================================
# FILES
# ============================================================

raster_file = "processed/susceptibility_class.tif"

points_file = "processed/training_features_complete.geojson"

osm_file = "raw_data/roads_villages/osm_data.json"

output_file = "processed/Giri_Rakshak_Final_Map.png"


# ============================================================
# READ SUSCEPTIBILITY RASTER
# ============================================================

print("Reading susceptibility map...")

with rasterio.open(raster_file) as src:

    susceptibility = src.read(1)

    bounds = src.bounds

    transform = src.transform

    crs = src.crs


# ============================================================
# READ TRAINING POINTS
# ============================================================

print("Reading landslide inventory...")

points = gpd.read_file(points_file)

points = points.to_crs(crs)

landslides = points[
    points["label"] == 1
]


# ============================================================
# READ OSM
# ============================================================

print("Reading OpenStreetMap data...")

with open(
    osm_file,
    encoding="utf-8"
) as f:

    osm = json.load(f)


roads = []
rivers = []
places = []


for element in osm.get("elements", []):

    tags = element.get("tags", {})

    # --------------------------------------------------------
    # ROADS
    # --------------------------------------------------------

    if (
        "highway" in tags
        and "geometry" in element
    ):

        coords = [
            (p["lon"], p["lat"])
            for p in element["geometry"]
        ]

        if len(coords) >= 2:

            roads.append(
                LineString(coords)
            )


    # --------------------------------------------------------
    # WATERWAYS
    # --------------------------------------------------------

    if (
        "waterway" in tags
        and "geometry" in element
    ):

        coords = [
            (p["lon"], p["lat"])
            for p in element["geometry"]
        ]

        if len(coords) >= 2:

            rivers.append(
                LineString(coords)
            )


    # --------------------------------------------------------
    # VILLAGES / TOWNS / CITIES
    # --------------------------------------------------------

    if (
        "place" in tags
        and "lat" in element
        and "lon" in element
    ):

        places.append(
            {
                "name": tags.get(
                    "name",
                    "Unnamed"
                ),
                "geometry": Point(
                    element["lon"],
                    element["lat"]
                )
            }
        )


# ============================================================
# CREATE GEODATAFRAMES
# ============================================================

roads_gdf = gpd.GeoDataFrame(
    geometry=roads,
    crs="EPSG:4326"
)

rivers_gdf = gpd.GeoDataFrame(
    geometry=rivers,
    crs="EPSG:4326"
)

places_gdf = gpd.GeoDataFrame(
    places,
    crs="EPSG:4326"
)


roads_gdf = roads_gdf.to_crs(crs)
rivers_gdf = rivers_gdf.to_crs(crs)
places_gdf = places_gdf.to_crs(crs)


print(
    "Roads:",
    len(roads_gdf)
)

print(
    "Rivers:",
    len(rivers_gdf)
)

print(
    "Places:",
    len(places_gdf)
)


# ============================================================
# MASK NODATA
# ============================================================

masked = susceptibility.astype(float)

masked[
    susceptibility == 255
] = float("nan")


# ============================================================
# AREA STATISTICS
# ============================================================

low_count = (
    (susceptibility == 1).sum()
)

moderate_count = (
    (susceptibility == 2).sum()
)

high_count = (
    (susceptibility == 3).sum()
)

total = (
    low_count
    + moderate_count
    + high_count
)


low_pct = (
    low_count / total * 100
)

moderate_pct = (
    moderate_count / total * 100
)

high_pct = (
    high_count / total * 100
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(14, 11)
)


# ============================================================
# SUSCEPTIBILITY
# ============================================================

cmap = ListedColormap(
    [
        "green",
        "yellow",
        "red"
    ]
)


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
# RIVERS
# ============================================================

if len(rivers_gdf) > 0:

    rivers_gdf.plot(
        ax=ax,
        linewidth=0.8,
        alpha=0.8
    )


# ============================================================
# ROADS
# ============================================================

if len(roads_gdf) > 0:

    roads_gdf.plot(
        ax=ax,
        linewidth=0.35,
        alpha=0.65
    )


# ============================================================
# LANDSLIDE LOCATIONS
# ============================================================

landslides.plot(
    ax=ax,
    marker="*",
    markersize=110,
    color="blue",
    edgecolor="white",
    linewidth=0.8
)


# ============================================================
# VILLAGES
# ============================================================

if len(places_gdf) > 0:

    places_gdf.plot(
        ax=ax,
        marker="o",
        markersize=18,
        color="black",
        alpha=0.8
    )


# ============================================================
# LABEL SELECTED PLACES
# ============================================================

for _, row in places_gdf.iterrows():

    name = row.get(
        "name",
        ""
    )

    if name and name != "Unnamed":

        ax.annotate(
            name,
            xy=(
                row.geometry.x,
                row.geometry.y
            ),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6
        )


# ============================================================
# SCALE BAR
# ============================================================

# The map is currently in EPSG:4326 (degrees).
# A physically accurate scale bar requires a projected CRS.
# We will add the proper kilometre scale bar after projection.

# ============================================================
# NORTH ARROW
# ============================================================

ax.annotate(
    "N",
    xy=(0.06, 0.91),
    xycoords="axes fraction",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold"
)

ax.annotate(
    "↑",
    xy=(0.06, 0.86),
    xycoords="axes fraction",
    ha="center",
    va="center",
    fontsize=30
)


# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Patch(
        facecolor="green",
        label=f"Low ({low_pct:.2f}%)"
    ),

    Patch(
        facecolor="yellow",
        label=f"Moderate ({moderate_pct:.2f}%)"
    ),

    Patch(
        facecolor="red",
        label=f"High ({high_pct:.2f}%)"
    ),

    Line2D(
        [0],
        [0],
        marker="*",
        linestyle="None",
        markersize=12,
        label="Known landslide"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markersize=6,
        label="Village / settlement"
    )

]


ax.legend(
    handles=legend_elements,
    loc="lower right",
    title="Landslide Susceptibility",
    framealpha=0.95
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "GIRI RAKSHAK\n"
    "AI-Based Landslide Susceptibility Map — Dibang Valley",
    fontsize=19,
    fontweight="bold",
    pad=18
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Longitude (°E)",
    fontsize=11
)

ax.set_ylabel(
    "Latitude (°N)",
    fontsize=11
)


# ============================================================
# GRID
# ============================================================

ax.grid(
    linestyle="--",
    alpha=0.3
)


# ============================================================
# INFORMATION BOX
# ============================================================

info = (
    "Model: Random Forest\n"
    f"Mapped area: {total:,} pixels\n"
    f"High susceptibility: {high_pct:.2f}%\n"
    f"Known landslides: {len(landslides)}\n"
    "Sources: DEM • Sentinel-2 • GPM • SMAP • OSM"
)


ax.text(
    0.015,
    0.015,
    info,
    transform=ax.transAxes,
    fontsize=8,
    verticalalignment="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        alpha=0.85
    )
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
print("STEP 29B COMPLETE")
print("========================================")

print(
    "Saved:",
    output_file
)

print(
    f"Low: {low_pct:.2f}%"
)

print(
    f"Moderate: {moderate_pct:.2f}%"
)

print(
    f"High: {high_pct:.2f}%"
)