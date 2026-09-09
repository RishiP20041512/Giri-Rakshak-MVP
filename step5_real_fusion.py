from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import rasterio
from rasterio.warp import transform

from risk_fusion import fuse_risk
from dynamic.config import EAST_KHASI_HILLS
from dynamic.pipeline import run_dynamic_layer


SUSCEPTIBILITY_RASTER = (
    ROOT / "processed" / "step70_8factor_susceptibility_probability.tif"
)


def get_real_susceptibility(lat, lon):

    with rasterio.open(SUSCEPTIBILITY_RASTER) as src:

        # Convert GPS coordinates EPSG:4326
        # to the raster CRS (EPSG:6933)
        x, y = transform(
            "EPSG:4326",
            src.crs,
            [lon],
            [lat]
        )

        row, col = src.index(x[0], y[0])

        # Check that the location is actually inside the raster
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None

        value = src.read(1)[row, col]

        if src.nodata is not None and value == src.nodata:
            return None

        return float(value)


def main():

    print("=" * 60)
    print("GIRI-RAKSHAK — REAL STATIC + DYNAMIC FUSION")
    print("=" * 60)

    cfg = EAST_KHASI_HILLS

    lat = cfg.station_lat
    lon = cfg.station_lon

    print(f"\nPilot district : {cfg.district_name}")
    print(f"Latitude       : {lat}")
    print(f"Longitude      : {lon}")

    # --------------------------------------------------------
    # REAL STATIC SUSCEPTIBILITY
    # --------------------------------------------------------

    susceptibility = get_real_susceptibility(lat, lon)

    if susceptibility is None:
        print("\nERROR: No valid susceptibility value at pilot location.")
        return

    print(f"\nReal RF Susceptibility : {susceptibility:.4f}")

    # --------------------------------------------------------
    # REAL DYNAMIC PIPELINE
    # --------------------------------------------------------

    print("\nRunning dynamic rainfall pipeline...")

    dynamic_result = run_dynamic_layer(cfg)

    trigger = dynamic_result.rainfall.trigger_score

    print(f"Real Dynamic Trigger   : {trigger:.4f}")

    # --------------------------------------------------------
    # REAL FUSION
    # --------------------------------------------------------

    result = fuse_risk(
        susceptibility_score=susceptibility,
        trigger_score=trigger
    )

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL REAL RISK RESULT")
    print("=" * 60)

    print(f"Static Susceptibility : {result.susceptibility_score:.4f}")
    print(f"Dynamic Trigger       : {result.trigger_score:.4f}")
    print(f"Final Risk Score      : {result.risk_score:.4f}")
    print(f"Risk Level            : {result.risk_level}")

    print("\nAudit trail:")

    for line in result.reasoning:
        print(" -", line)

    print("\nDynamic pipeline state:", dynamic_result.final_state)
    print("Wetness source:", dynamic_result.wetness_source)

    print("=" * 60)


if __name__ == "__main__":
    main()