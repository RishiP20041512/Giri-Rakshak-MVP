from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

BOUNDARY_FILE = (
    PROJECT
    / "raw_data"
    / "boundaries"
    / "geoBoundaries-IND-ADM1.geojson"
)

GRID_FILE = (
    PROJECT
    / "processed"
    / "step64a_final_ner_grid.tif"
)

OUTPUT_MASK = (
    PROJECT
    / "processed"
    / "step64a_final_ner_mask.tif"
)


# ============================================================
# NER STATES
# ============================================================

NER_STATES = [
    "Arunāchal Pradesh",
    "Assam",
    "Manipur",
    "Meghālaya",
    "Mizoram",
    "Nāgāland",
    "Sikkim",
    "Tripura",
]


print("=" * 75)
print("STEP 68H — REBUILD CORRECT NER MASK")
print("=" * 75)


# ============================================================
# LOAD BOUNDARIES
# ============================================================

print("\nLoading India ADM1 boundary...")

gdf = gpd.read_file(
    BOUNDARY_FILE
)

print(
    f"Total administrative units: "
    f"{len(gdf)}"
)

print(
    f"Boundary CRS: "
    f"{gdf.crs}"
)

print(
    "\nAvailable state names:"
)

print(
    gdf["shapeName"]
    .sort_values()
    .to_string(index=False)
)


# ============================================================
# FILTER NER
# ============================================================

ner = gdf[
    gdf["shapeName"].isin(
        NER_STATES
    )
].copy()


print("\n" + "=" * 75)
print("NER STATE SELECTION")
print("=" * 75)

print(
    f"Requested NER states: "
    f"{len(NER_STATES)}"
)

print(
    f"Matched states: "
    f"{len(ner)}"
)

print("\nMatched states:")

for state in ner["shapeName"]:
    print(
        f"  {state}"
    )


# ============================================================
# VERIFY ALL 8 STATES
# ============================================================

missing_states = sorted(
    set(NER_STATES)
    - set(ner["shapeName"])
)

if missing_states:

    raise ValueError(
        "The following NER states were not found:\n"
        + "\n".join(
            f"  - {s}"
            for s in missing_states
        )
    )


if len(ner) != 8:

    raise ValueError(
        f"Expected 8 NER states, "
        f"but found {len(ner)}."
    )


# ============================================================
# CALCULATE NER AREA
# ============================================================

ner_projected = ner.to_crs(
    "EPSG:6933"
)

area_km2 = (
    ner_projected.geometry.area.sum()
    / 1_000_000
)

print(
    f"\nNER area: "
    f"{area_km2:,.2f} km²"
)


# ============================================================
# LOAD FINAL GRID
# ============================================================

print("\n" + "=" * 75)
print("LOADING FINAL GRID")
print("=" * 75)

with rasterio.open(
    GRID_FILE
) as grid:

    grid_crs = grid.crs
    transform = grid.transform
    width = grid.width
    height = grid.height
    resolution = grid.res
    profile = grid.profile.copy()

    print(
        f"Grid CRS: "
        f"{grid_crs}"
    )

    print(
        f"Grid dimensions: "
        f"{width} x {height}"
    )

    print(
        f"Grid resolution: "
        f"{resolution}"
    )

    print(
        f"Grid bounds: "
        f"{grid.bounds}"
    )


# ============================================================
# REPROJECT NER GEOMETRY
# ============================================================

print("\nReprojecting NER boundaries...")

ner_projected = ner.to_crs(
    grid_crs
)


# ============================================================
# RASTERIZE NER
# ============================================================

print("\nRasterizing NER states...")

shapes = [
    (geom, 1)
    for geom in
    ner_projected.geometry
    if geom is not None
    and not geom.is_empty
]


mask = rasterize(
    shapes=shapes,
    out_shape=(
        height,
        width
    ),
    transform=transform,
    fill=0,
    default_value=1,
    dtype="uint8",
    all_touched=False
)


# ============================================================
# STATISTICS
# ============================================================

ner_cells = int(
    (mask == 1).sum()
)

outside_cells = int(
    (mask == 0).sum()
)

cell_area_km2 = (
    resolution[0]
    * resolution[1]
    / 1_000_000
)

raster_area_km2 = (
    ner_cells
    * cell_area_km2
)

print("\n" + "=" * 75)
print("MASK STATISTICS")
print("=" * 75)

print(
    f"NER cells: "
    f"{ner_cells:,}"
)

print(
    f"Outside cells: "
    f"{outside_cells:,}"
)

print(
    f"250 m cell area: "
    f"{cell_area_km2:.4f} km²"
)

print(
    f"Rasterized NER area: "
    f"{raster_area_km2:,.2f} km²"
)

print(
    f"Vector NER area: "
    f"{area_km2:,.2f} km²"
)

area_difference = (
    raster_area_km2
    - area_km2
)

area_difference_percent = (
    area_difference
    / area_km2
    * 100
)

print(
    f"Area difference: "
    f"{area_difference:,.2f} km²"
)

print(
    f"Area difference: "
    f"{area_difference_percent:.3f}%"
)


# ============================================================
# WRITE MASK
# ============================================================

print("\n" + "=" * 75)
print("WRITING CORRECT NER MASK")
print("=" * 75)

profile.update(
    driver="GTiff",
    dtype="uint8",
    count=1,
    nodata=0,
    compress="deflate",
    tiled=False,
    BIGTIFF="IF_SAFER"
)

profile.pop(
    "blockxsize",
    None
)

profile.pop(
    "blockysize",
    None
)


with rasterio.open(
    OUTPUT_MASK,
    "w",
    **profile
) as dst:

    dst.write(
        mask,
        1
    )


print(
    f"\nSaved:\n{OUTPUT_MASK}"
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("FINAL VERIFICATION")
print("=" * 75)

with rasterio.open(
    OUTPUT_MASK
) as check:

    arr = check.read(1)

    print(
        f"CRS: {check.crs}"
    )

    print(
        f"Dimensions: "
        f"{check.width} x {check.height}"
    )

    print(
        f"Resolution: "
        f"{check.res}"
    )

    print(
        f"NoData: "
        f"{check.nodata}"
    )

    unique, counts = np.unique(
        arr,
        return_counts=True
    )

    print(
        "\nValue counts:"
    )

    for value, count in zip(
        unique,
        counts
    ):

        print(
            f"  {value}: "
            f"{count:,}"
        )


print("\n" + "=" * 75)
print("STEP 68H COMPLETE")
print("=" * 75)