import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from shapely.geometry import mapping


# ============================================================
# STEP 75
# INDIVIDUAL STATE SUSCEPTIBILITY MAPS
#
# IMPORTANT:
# - Uses ORIGINAL continuous RF output
# - Does NOT modify the RF model
# - Does NOT modify prediction values
# - Only changes map visualization
# - Smooth rendering for better visual quality
# ============================================================

print("=" * 75)
print("STEP 75 — INDIVIDUAL STATE SUSCEPTIBILITY MAPS")
print("CONTINUOUS RF OUTPUT + SMOOTH VISUALIZATION")
print("=" * 75)


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# ORIGINAL RF CONTINUOUS OUTPUT
RASTER = os.path.join(
    ROOT,
    "processed",
    "step70_8factor_susceptibility_probability.tif"
)

BOUNDARY = os.path.join(
    ROOT,
    "raw_data",
    "boundaries",
    "ner_8states.geojson"
)

OUT_DIR = os.path.join(
    ROOT,
    "processed",
    "state_maps"
)

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# STATES
# ============================================================

states = [
    "Arunāchal Pradesh",
    "Assam",
    "Manipur",
    "Meghālaya",
    "Mizoram",
    "Nāgāland",
    "Sikkim",
    "Tripura"
]

display_names = {
    "Arunāchal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam",
    "Manipur": "Manipur",
    "Meghālaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nāgāland": "Nagaland",
    "Sikkim": "Sikkim",
    "Tripura": "Tripura"
}


# ============================================================
# VISUAL CLASS BREAKS
# ============================================================
#
# THESE DO NOT CHANGE THE RF VALUES.
#
# They only control how the continuous RF score
# is displayed on the map.
#
# Lower thresholds intentionally make high-susceptibility
# zones more visually prominent.
#
# ============================================================

breaks = [
    0.00,
    0.20,
    0.35,
    0.50,
    0.65,
    1.00
]

colors = [
    "#2ca25f",   # Very Low
    "#72dc8c",   # Low — Light Green
    "#fee391",   # Moderate
    "#fe9929",   # High
    "#de2d26"    # Very High
]
labels = [
    "Very Low / Low (0.00–0.20)",
    "Low (0.20–0.35)",
    "Moderate (0.35–0.50)",
    "High (0.50–0.65)",
    "Very High (0.65–1.00)"
]

cmap = ListedColormap(
    colors
)

norm = BoundaryNorm(
    breaks,
    cmap.N
)


# ============================================================
# LOAD BOUNDARIES
# ============================================================

print("\nReading state boundaries...")

gdf = gpd.read_file(
    BOUNDARY
)

print(
    "Boundary features:",
    len(gdf)
)

print(
    "Boundary CRS:",
    gdf.crs
)


# ============================================================
# OPEN RF RASTER
# ============================================================

print("\nReading continuous RF susceptibility raster...")

src = rasterio.open(
    RASTER
)

print(
    "Raster size:",
    src.width,
    "x",
    src.height
)

print(
    "Raster CRS:",
    src.crs
)

print(
    "Raster NoData:",
    src.nodata
)


# ============================================================
# PROCESS EACH STATE
# ============================================================

for state in states:

    name = display_names[state]

    print("\n" + "-" * 75)
    print("STATE:", name)
    print("-" * 75)


    # ========================================================
    # FIND STATE
    # ========================================================

    state_gdf = gdf[
        gdf["shapeName"] == state
    ].copy()

    if state_gdf.empty:

        print(
            "WARNING: State boundary not found:",
            state
        )

        continue


    # ========================================================
    # REPROJECT STATE
    # ========================================================

    state_gdf = state_gdf.to_crs(
        src.crs
    )


    # ========================================================
    # CLIP RASTER
    # ========================================================

    try:

        clipped, transform = mask(
            src,
            [
                mapping(geom)
                for geom in state_gdf.geometry
            ],
            crop=True,
            nodata=-9999
        )

    except Exception as e:

        print(
            "Could not clip:",
            e
        )

        continue


    data = clipped[0].astype(
        "float32"
    )


    # ========================================================
    # VALID DATA
    # ========================================================

    invalid = (
        data == -9999
    )

    invalid |= (
        ~np.isfinite(data)
    )

    invalid |= (
        data < 0
    )

    invalid |= (
        data > 1
    )

    valid = data[
        ~invalid
    ]


    # ========================================================
    # NO DATA CASE
    # ========================================================

    if len(valid) == 0:

        print(
            "No susceptibility prediction available."
        )

        fig, ax = plt.subplots(
            figsize=(12, 9)
        )

        state_gdf.boundary.plot(
            ax=ax,
            linewidth=2,
            color="black"
        )

        try:

            centroid = (
                state_gdf
                .geometry
                .union_all()
                .centroid
            )

        except Exception:

            centroid = (
                state_gdf
                .geometry
                .unary_union
                .centroid
            )

        ax.text(
            centroid.x,
            centroid.y,
            "No prediction available\n"
            "due to unavailable predictor coverage",
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold"
        )

        ax.set_title(
            f"{name} — Landslide Susceptibility Map",
            fontsize=21,
            fontweight="bold",
            pad=18
        )

        ax.set_axis_off()

        plt.tight_layout()

        safe_name = name.replace(
            " ",
            "_"
        )

        png_path = os.path.join(
            OUT_DIR,
            f"{safe_name}_susceptibility_map.png"
        )

        pdf_path = os.path.join(
            OUT_DIR,
            f"{safe_name}_susceptibility_map.pdf"
        )

        plt.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.savefig(
            pdf_path,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "PNG:",
            png_path
        )

        print(
            "PDF:",
            pdf_path
        )

        continue


    # ========================================================
    # REAL RF STATISTICS
    # ========================================================

    mean_s = float(
        np.mean(valid)
    )

    median_s = float(
        np.median(valid)
    )

    minimum_s = float(
        np.min(valid)
    )

    maximum_s = float(
        np.max(valid)
    )

    print(
        f"Valid cells: {len(valid):,}"
    )

    print(
        f"Mean RF susceptibility: {mean_s:.4f}"
    )

    print(
        f"Median RF susceptibility: {median_s:.4f}"
    )

    print(
        f"Minimum RF susceptibility: {minimum_s:.4f}"
    )

    print(
        f"Maximum RF susceptibility: {maximum_s:.4f}"
    )


    # ========================================================
    # CLASS COUNTS
    # ========================================================

    class_counts = []

    for i in range(len(breaks) - 1):

        lower = breaks[i]
        upper = breaks[i + 1]

        if i == len(breaks) - 2:

            count = np.sum(
                (valid >= lower) &
                (valid <= upper)
            )

        else:

            count = np.sum(
                (valid >= lower) &
                (valid < upper)
            )

        class_counts.append(
            int(count)
        )


    print("\nVisualization classes:")

    for i, count in enumerate(class_counts):

        pct = (
            100.0 *
            count /
            len(valid)
        )

        print(
            f"{labels[i]}: "
            f"{count:,} cells "
            f"({pct:.2f}%)"
        )


    # ========================================================
    # PREPARE DATA
    # ========================================================

    plot_data = np.ma.masked_where(
        invalid,
        data
    )


    # ========================================================
    # RASTER EXTENT
    # ========================================================

    xmin = transform.c
    ymax = transform.f

    xmax = (
        xmin +
        transform.a *
        data.shape[1]
    )

    ymin = (
        ymax +
        transform.e *
        data.shape[0]
    )


    # ========================================================
    # CREATE LARGE HIGH-QUALITY MAP
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(13, 9)
    )


    # ========================================================
    # DRAW CONTINUOUS RF MAP
    # ========================================================
    #
    # IMPORTANT:
    #
    # The underlying RF values are continuous.
    #
    # bilinear interpolation ONLY smooths the display.
    #
    # It does NOT change the saved RF raster.
    #
    # ========================================================

    ax.imshow(
        plot_data,
        cmap=cmap,
        norm=norm,
        extent=[
            xmin,
            xmax,
            ymin,
            ymax
        ],
        interpolation="bilinear"
    )


    # ========================================================
    # STATE OUTLINE
    # ========================================================

    state_gdf.boundary.plot(
        ax=ax,
        linewidth=2.2,
        color="black"
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        f"{name} — Landslide Susceptibility Map\n"
        "8-Factor Random Forest | 250 m spatial grid",
        fontsize=20,
        fontweight="bold",
        pad=18
    )


    # ========================================================
    # AXIS LABELS
    # ========================================================

    ax.set_xlabel(
        "Easting (m)",
        fontsize=12
    )

    ax.set_ylabel(
        "Northing (m)",
        fontsize=12
    )


    # ========================================================
    # LEGEND
    # ========================================================

    legend_handles = []

    for i in range(len(colors)):

        legend_handles.append(
            Patch(
                facecolor=colors[i],
                edgecolor="black",
                linewidth=0.8,
                label=labels[i]
            )
        )

    ax.legend(
        handles=legend_handles,
        title="Susceptibility",
        loc="lower left",
        fontsize=10,
        title_fontsize=12,
        frameon=True,
        framealpha=0.92
    )


    # ========================================================
    # STATISTICS BOX
    # ========================================================

    stats_text = (
        f"Mapped cells: {len(valid):,}\n"
        f"Mean S: {mean_s:.3f}\n"
        f"Median S: {median_s:.3f}\n"
        f"Min S: {minimum_s:.3f}\n"
        f"Max S: {maximum_s:.3f}"
    )

    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            alpha=0.92,
            edgecolor="gray"
        )
    )


    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.25
    )


    # ========================================================
    # SAVE PNG
    # ========================================================

    plt.tight_layout()

    safe_name = name.replace(
        " ",
        "_"
    )

    png_path = os.path.join(
        OUT_DIR,
        f"{safe_name}_susceptibility_map.png"
    )

    pdf_path = os.path.join(
        OUT_DIR,
        f"{safe_name}_susceptibility_map.pdf"
    )


    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "PNG saved:",
        png_path
    )

    print(
        "PDF saved:",
        pdf_path
    )


# ============================================================
# CLOSE
# ============================================================

src.close()


print("\n" + "=" * 75)
print("STEP 75 COMPLETE")
print("=" * 75)

print("\nMaps saved in:")
print(OUT_DIR)