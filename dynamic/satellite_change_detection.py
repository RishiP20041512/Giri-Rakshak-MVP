"""
satellite_change_detection.py

Rule/formula-based satellite change detection core (architecture doc
Section 6.3). Pure numpy math, no training. This module operates on
plain numpy arrays (from a GeoTIFF, or from Earth Engine pixel data
pulled down via gee_satellite.py) so it is testable without any Earth
Engine dependency.

Two independent detectors, either of which can flag "possible surface
disturbance" (bare-soil exposure, vegetation loss, new scarp):
  1. NDVI differencing (optical) — works when cloud-free.
  2. SAR backscatter log-ratio change — works in all weather, critical
     for NER's monsoon cloud cover.
"""

import numpy as np

from .config import DistrictConfig


# ---------------------------------------------------------------------------
# 1. NDVI
# ---------------------------------------------------------------------------
def compute_ndvi(nir_band: np.ndarray, red_band: np.ndarray) -> np.ndarray:
    nir = nir_band.astype("float32")
    red = red_band.astype("float32")
    denom = nir + red
    denom[denom == 0] = 1e-6
    ndvi = (nir - red) / denom
    return np.clip(ndvi, -1.0, 1.0)


def ndvi_difference(ndvi_before: np.ndarray, ndvi_after: np.ndarray) -> np.ndarray:
    """Large negative value = vegetation lost between the two dates."""
    return ndvi_after - ndvi_before


def ndvi_disturbance_mask(ndvi_diff: np.ndarray, cfg: DistrictConfig) -> np.ndarray:
    return ndvi_diff <= cfg.ndvi_drop_threshold


# ---------------------------------------------------------------------------
# 2. SAR backscatter change
# ---------------------------------------------------------------------------
def sar_log_ratio(vv_before_db: np.ndarray, vv_after_db: np.ndarray) -> np.ndarray:
    return vv_after_db - vv_before_db


def sar_disturbance_mask(sar_diff_db: np.ndarray, cfg: DistrictConfig) -> np.ndarray:
    return np.abs(sar_diff_db) >= cfg.sar_abs_threshold_db


# ---------------------------------------------------------------------------
# 3. Combined flag + summary
# ---------------------------------------------------------------------------
def combined_disturbance_flag(ndvi_mask: np.ndarray = None, sar_mask: np.ndarray = None) -> np.ndarray:
    """
    Logical OR — either signal alone is enough to raise a screening flag.
    This is a screening tool, not a verdict; flagged cells still get
    corroborated by susceptibility + trigger score + field verification
    in the Risk Fusion Engine (Section 8).
    """
    if ndvi_mask is None and sar_mask is None:
        raise ValueError("At least one of ndvi_mask or sar_mask must be provided")
    if ndvi_mask is None:
        return sar_mask
    if sar_mask is None:
        return ndvi_mask
    return ndvi_mask | sar_mask


def disturbance_summary(mask: np.ndarray, cfg: DistrictConfig) -> dict:
    flagged_pixels = int(np.sum(mask))
    area_ha = (flagged_pixels * cfg.pixel_area_m2) / 10000.0
    return {"flagged_pixels": flagged_pixels, "estimated_area_ha": round(area_ha, 2)}


# ---------------------------------------------------------------------------
# GETTING REAL DATA FROM A LOCAL GEOTIFF (reference)
# ---------------------------------------------------------------------------
def load_band(geotiff_path: str, band_index: int = 1) -> np.ndarray:
    """Loads a single band from a real GeoTIFF using rasterio."""
    import rasterio
    with rasterio.open(geotiff_path) as src:
        return src.read(band_index)
