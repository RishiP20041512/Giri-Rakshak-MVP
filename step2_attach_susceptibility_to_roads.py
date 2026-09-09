from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString
from shapely.ops import transform, linemerge


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

ROAD_GPKG = ROOT / "pilot_route_data" / "pilot_road_network.gpkg"

SUSCEPTIBILITY_RASTER = (
    ROOT
    / "processed"
    / "step70_8factor_susceptibility_probability.tif"
)

OUTPUT_DIR = ROOT / "pilot_route_data"

OUTPUT_GPKG = OUTPUT_DIR / "pilot_roads_with_susceptibility.gpkg"
OUTPUT_CSV = OUTPUT_DIR / "pilot_roads_with_susceptibility.csv"


# ============================================================
# SETTINGS
# ============================================================

ROAD_LAYER = "pilot_roads"

# Sample approximately every 125 m along each road.
# The susceptibility raster itself is 250 m.
SAMPLE_SPACING_M = 125.0


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_lsi(value):

    if value is None or not np.isfinite(value):
        return "NO_DATA"

    if value < 0.20:
        return "VERY LOW"

    elif value < 0.40:
        return "LOW"

    elif value < 0.60:
        return "MODERATE"

    elif value < 0.80:
        return "HIGH"

    else:
        return "VERY HIGH"


# ============================================================
# GET LINE PARTS
# ============================================================

def get_line_parts(geometry):

    if geometry is None or geometry.is_empty:
        return []

    if isinstance(geometry, LineString):
        return [geometry]

    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)

    # Try merging other line-like geometries
    try:
        merged = linemerge(geometry)

        if isinstance(merged, LineString):
            return [merged]

        if isinstance(merged, MultiLineString):
            return list(merged.geoms)

    except Exception:
        pass

    return []


# ============================================================
# CREATE SAMPLE POINTS
# ============================================================

def sample_points_along_geometry(geometry):

    points = []

    line_parts = get_line_parts(geometry)

    for line in line_parts:

        length = line.length

        if length <= 0:
            continue

        # Number of intervals
        n = max(1, int(np.ceil(length / SAMPLE_SPACING_M)))

        for i in range(n + 1):

            distance = min(i * SAMPLE_SPACING_M, length)

            point = line.interpolate(distance)

            points.append(point)

    return points


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 2 — ATTACH RF SUSCEPTIBILITY TO ROAD NETWORK")
    print("=" * 70)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not ROAD_GPKG.exists():
        raise FileNotFoundError(
            f"Road network not found:\n{ROAD_GPKG}"
        )

    if not SUSCEPTIBILITY_RASTER.exists():
        raise FileNotFoundError(
            f"Susceptibility raster not found:\n"
            f"{SUSCEPTIBILITY_RASTER}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # READ ROADS
    # --------------------------------------------------------

    print("\n[1/6] Reading pilot road network...")

    roads = gpd.read_file(
        ROAD_GPKG,
        layer=ROAD_LAYER
    )

    print(f"Roads loaded: {len(roads):,}")
    print(f"Road CRS: {roads.crs}")

    # --------------------------------------------------------
    # OPEN SUSCEPTIBILITY RASTER
    # --------------------------------------------------------

    print("\n[2/6] Opening RF susceptibility raster...")

    with rasterio.open(SUSCEPTIBILITY_RASTER) as src:

        print(f"Raster CRS: {src.crs}")
        print(f"Raster size: {src.width} x {src.height}")
        print(f"Raster resolution: {src.res}")
        print(f"NoData: {src.nodata}")

        raster_crs = src.crs
        raster_nodata = src.nodata

        # ----------------------------------------------------
        # TRANSFORM ROAD GEOMETRY TO RASTER CRS
        # ----------------------------------------------------

        transformer = Transformer.from_crs(
            roads.crs,
            raster_crs,
            always_xy=True
        )

        def project_geometry(geom):
            return transform(
                transformer.transform,
                geom
            )

        # ----------------------------------------------------
        # PROCESS EACH ROAD
        # ----------------------------------------------------

        print("\n[3/6] Sampling susceptibility along roads...")
        print("This may take a little time because there are many roads.")

        mean_values = []
        max_values = []
        valid_sample_counts = []

        total_roads = len(roads)

        for idx, geometry in enumerate(roads.geometry):

            if idx % 1000 == 0:
                print(
                    f"Processing road {idx:,} / "
                    f"{total_roads:,}"
                )

            projected_geometry = project_geometry(geometry)

            points = sample_points_along_geometry(
                projected_geometry
            )

            if not points:

                mean_values.append(np.nan)
                max_values.append(np.nan)
                valid_sample_counts.append(0)

                continue

            coordinates = [
                (point.x, point.y)
                for point in points
            ]

            values = []

            for value in src.sample(coordinates):

                v = float(value[0])

                # Handle NoData
                if raster_nodata is not None:
                    if np.isclose(v, raster_nodata):
                        continue

                # Ignore invalid values
                if not np.isfinite(v):
                    continue

                # RF susceptibility should be 0–1
                if v < 0 or v > 1:
                    continue

                values.append(v)

            if values:

                mean_values.append(
                    float(np.mean(values))
                )

                max_values.append(
                    float(np.max(values))
                )

                valid_sample_counts.append(
                    len(values)
                )

            else:

                mean_values.append(np.nan)
                max_values.append(np.nan)
                valid_sample_counts.append(0)

    # --------------------------------------------------------
    # ATTACH RESULTS
    # --------------------------------------------------------

    print("\n[4/6] Attaching susceptibility values...")

    roads["LSI_mean"] = mean_values
    roads["LSI_max"] = max_values
    roads["LSI_valid_samples"] = valid_sample_counts

    roads["risk_class"] = roads["LSI_mean"].apply(
        classify_lsi
    )

    # --------------------------------------------------------
    # QA
    # --------------------------------------------------------

    print("\n[5/6] Quality check...")

    valid = roads["LSI_mean"].notna()

    print(f"Total roads:       {len(roads):,}")
    print(f"With susceptibility:{valid.sum():,}")
    print(f"Without data:      {(~valid).sum():,}")

    if valid.any():

        print("\nLSI_mean statistics:")

        print(
            roads.loc[valid, "LSI_mean"].describe()
        )

        print("\nRisk class counts:")

        print(
            roads["risk_class"]
            .value_counts(dropna=False)
            .sort_index()
        )

    # --------------------------------------------------------
    # SAVE GPKG
    # --------------------------------------------------------

    print("\n[6/6] Saving outputs...")

    if OUTPUT_GPKG.exists():
        OUTPUT_GPKG.unlink()

    roads.to_file(
        OUTPUT_GPKG,
        layer="pilot_roads_risk",
        driver="GPKG"
    )

    # CSV without geometry
    roads.drop(columns="geometry").to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE!")
    print("=" * 70)

    print(f"\nGeoPackage:")
    print(OUTPUT_GPKG)

    print(f"\nCSV:")
    print(OUTPUT_CSV)

    print("\nNew road fields:")
    print("  LSI_mean")
    print("  LSI_max")
    print("  LSI_valid_samples")
    print("  risk_class")

    print("\nExample interpretation:")
    print("  LOW       → road has relatively low modeled susceptibility")
    print("  MODERATE  → caution")
    print("  HIGH      → prioritize alternative route")
    print("  VERY HIGH → strong avoidance candidate")

    print("\nIMPORTANT:")
    print("LSI is the existing RF susceptibility score.")
    print("It is NOT a literal probability of landslide occurrence.")


if __name__ == "__main__":
    main()