"""
sensor_fusion.py

Closes an MVP "SHOULD HAVE" gap (architecture doc Section 15): blending a
live soil-moisture reading with the rainfall-derived Antecedent Wetness
Index, from either
  (a) a real IoT ground sensor at a known village, or
  (b) NASA SMAP satellite soil moisture (via Earth Engine) where no
      physical sensor exists — this is the "sensors give local
      ground-truth, SMAP/API give spatial coverage everywhere" design
      from architecture doc Section 6.2/6.3.
"""

from typing import Optional

from .config import DistrictConfig
from .rainfall_trigger import normalize, calibrate_api_with_sensor


def blend_iot_sensor_reading(raw_api_value: float, live_soil_moisture_pct: float,
                              cfg: DistrictConfig,
                              soil_moisture_pct_range=(0.0, 60.0)) -> dict:
    """
    raw_api_value: output of antecedent_wetness_index() for this cell (mm-scale).
    live_soil_moisture_pct: a real IoT sensor's volumetric water content
        reading (%), typically 0-60% for soil.

    Both are normalized to 0-1 before blending (they're on different
    physical scales — mm of effective rainfall memory vs % soil water
    content — so blending raw units would silently misweight one of them).
    Returns a dict with both normalized inputs and the blended result,
    so the blend is auditable rather than a single opaque number.
    """
    api_norm = normalize(raw_api_value, *cfg.api_norm_range)
    sensor_norm = normalize(live_soil_moisture_pct, *soil_moisture_pct_range)
    blended = calibrate_api_with_sensor(api_norm, sensor_norm, cfg)
    return {
        "raw_api_value": raw_api_value,
        "api_normalized": round(api_norm, 3),
        "sensor_reading_pct": live_soil_moisture_pct,
        "sensor_normalized": round(sensor_norm, 3),
        "sensor_blend_weight": cfg.sensor_blend_weight,
        "blended_wetness_0_1": round(blended, 3),
    }


def fetch_smap_soil_moisture(lat: float, lon: float, date: str) -> Optional[float]:
    """
    Real satellite soil-moisture reading (surface, %) from NASA SMAP via
    Earth Engine, for a point with no physical IoT sensor — the
    "satellite fills sensor coverage gaps" role from Section 6.3 /
    Section 4's data table.

    Returns None (not a crash) if no SMAP granule is available for the
    date/location, or if Earth Engine isn't authenticated — callers
    should fall back to the pure rainfall-derived API value in that case.
    """
    try:
        import ee
    except ImportError:
        print("[sensor_fusion] earthengine-api not installed; skipping SMAP fetch.")
        return None

    try:
        point = ee.Geometry.Point([lon, lat])
        collection = (
            ee.ImageCollection("NASA/SMAP/SPL4SMGP/007")
            .filterDate(date, ee.Date(date).advance(1, "day"))
            .filterBounds(point)
        )
        if collection.size().getInfo() == 0:
            return None
        image = collection.first()
        # sm_surface is volumetric water content, m3/m3 (0-~0.5) -> convert to %
        value = image.select("sm_surface").reduceRegion(
            reducer=ee.Reducer.mean(), geometry=point, scale=10000,
        ).get("sm_surface").getInfo()
        return round(value * 100, 2) if value is not None else None
    except Exception as exc:  # noqa: BLE001 - graceful degradation, not a silent failure
        print(f"[sensor_fusion] SMAP fetch failed ({exc!r}); falling back to rainfall-only API.")
        return None


def get_wetness_estimate(raw_api_value: float, cfg: DistrictConfig,
                          lat: float = None, lon: float = None, date: str = None,
                          local_iot_reading_pct: Optional[float] = None) -> dict:
    """
    Single entry point the pipeline should call: prefers a real local IoT
    sensor reading if you have one, falls back to SMAP if lat/lon/date
    are given, falls back to pure rainfall-derived API if neither is
    available. Always returns which source was actually used.
    """
    if local_iot_reading_pct is not None:
        result = blend_iot_sensor_reading(raw_api_value, local_iot_reading_pct, cfg)
        result["source"] = "local_iot_sensor"
        return result

    if lat is not None and lon is not None and date is not None:
        smap_pct = fetch_smap_soil_moisture(lat, lon, date)
        if smap_pct is not None:
            result = blend_iot_sensor_reading(raw_api_value, smap_pct, cfg)
            result["source"] = "smap_satellite"
            return result

    api_norm = normalize(raw_api_value, *cfg.api_norm_range)
    return {
        "raw_api_value": raw_api_value,
        "api_normalized": round(api_norm, 3),
        "blended_wetness_0_1": round(api_norm, 3),
        "source": "rainfall_derived_api_only",
    }
