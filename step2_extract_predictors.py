from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform


ROOT = Path(__file__).resolve().parent

PREDICTOR_DIR = ROOT / "processed" / "predictors"

RASTERS = {
    "elevation_m": PREDICTOR_DIR / "elevation_250m.tif",
    "slope_deg": PREDICTOR_DIR / "slope_250m.tif",
    "rainfall_3day": PREDICTOR_DIR / "rainfall_3day_250m.tif",
    "soil_moisture": PREDICTOR_DIR / "soil_moisture_3day_250m.tif",
    "ndvi": PREDICTOR_DIR / "ndvi_250m.tif",
    "distance_to_road_m": PREDICTOR_DIR / "distance_to_road_250m.tif",
    "lineament_density": PREDICTOR_DIR / "lineament_density_250m.tif",
    "geomorph_origin": PREDICTOR_DIR / "geomorph_origin_250m.tif",
}


def extract_value(raster_path, lat, lon):
    """
    Extract one real raster value at the given WGS84 latitude/longitude.
    """

    with rasterio.open(raster_path) as src:

        # Convert WGS84 lon/lat -> raster CRS
        x, y = transform(
            "EPSG:4326",
            src.crs,
            [lon],
            [lat],
        )

        x = x[0]
        y = y[0]

        # Convert map coordinates -> raster row/column
        row, col = src.index(x, y)

        # Check bounds
        if (
            row < 0
            or row >= src.height
            or col < 0
            or col >= src.width
        ):
            return None

        value = src.read(1)[row, col]

        # Handle NoData
        if src.nodata is not None:
            if np.isclose(value, src.nodata):
                return None

        # Handle NaN
        if not np.isfinite(value):
            return None

        return float(value)


def extract_all_predictors(lat, lon):

    results = {}

    print("\nExtracting REAL predictor values...")
    print("-" * 60)

    for name, raster_path in RASTERS.items():

        print(f"{name:<25}", end="")

        if not raster_path.exists():
            print("❌ FILE NOT FOUND")
            results[name] = None
            continue

        value = extract_value(
            raster_path,
            lat,
            lon,
        )

        results[name] = value

        if value is None:
            print("❌ NO DATA")
        else:
            print(f"✅ {value:.6f}")

    return results


def main():

    print("=" * 60)
    print("GIRI-RAKSHAK — STEP 2")
    print("LAT/LON → REAL 8 PREDICTORS")
    print("=" * 60)

    # GUWAHATI TEST LOCATION
    lat = 26.1445
    lon = 91.7362

    print(f"\nLatitude  : {lat}")
    print(f"Longitude : {lon}")

    predictors = extract_all_predictors(
        lat,
        lon,
    )

    print("\n" + "=" * 60)
    print("FINAL PREDICTOR VALUES")
    print("=" * 60)

    for name, value in predictors.items():

        if value is None:
            print(f"{name:<25} = MISSING")
        else:
            print(f"{name:<25} = {value:.6f}")

    missing = [
        name
        for name, value in predictors.items()
        if value is None
    ]

    print("\n" + "=" * 60)

    if missing:

        print("❌ PREDICTION INPUT INCOMPLETE")
        print("\nMissing factors:")

        for name in missing:
            print(f"  - {name}")

        print(
            "\nWe will NOT replace missing values with 0."
        )

    else:

        print("✅ ALL 8 PREDICTORS AVAILABLE")
        print(
            "\nThis coordinate has all required "
            "real inputs for the RF model."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()