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


def disturbance_centroid_fraction(mask: np.ndarray):
    """
    Mean (row, col) position of all True pixels in `mask`, normalized to
    [0, 1] by the mask's own shape. Returns None if nothing is flagged.

    row_frac=0 is the top row of the array (north edge, since Earth Engine
    sampleRectangle arrays run north-to-south), col_frac=0 is the left
    column (west edge, arrays run west-to-east). Callers combine this with
    the query region's real lat/lon bounds to get an actual centroid
    coordinate — this function only knows about the pixel grid.
    """
    rows, cols = np.where(mask)
    if rows.size == 0:
        return None
    row_frac = float(np.mean(rows)) / max(mask.shape[0] - 1, 1)
    col_frac = float(np.mean(cols)) / max(mask.shape[1] - 1, 1)
    return row_frac, col_frac


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


def disturbance_summary(mask: np.ndarray, cfg: DistrictConfig, pixel_area_m2: float = None) -> dict:
    """
    pixel_area_m2: area of ONE pixel in the mask being summarized, in m^2.
    Defaults to cfg.pixel_area_m2 (30m DEM/terrain-factor pixels, per
    config.py) for backward compatibility, but callers working with a
    different native resolution (e.g. 10m Sentinel-1/Sentinel-2 pixels
    via gee_satellite.py) should pass the correct value explicitly —
    using the 30m DEM assumption on 10m satellite pixels over-reports
    the disturbed area by ~9x.
    """
    flagged_pixels = int(np.sum(mask))
    effective_pixel_area_m2 = pixel_area_m2 if pixel_area_m2 is not None else cfg.pixel_area_m2
    area_ha = (flagged_pixels * effective_pixel_area_m2) / 10000.0
    return {"flagged_pixels": flagged_pixels, "estimated_area_ha": round(area_ha, 2)}


# ---------------------------------------------------------------------------
# GETTING REAL DATA FROM A LOCAL GEOTIFF (reference)
# ---------------------------------------------------------------------------
def load_band(geotiff_path: str, band_index: int = 1) -> np.ndarray:
    """Loads a single band from a real GeoTIFF using rasterio."""
    import rasterio
    with rasterio.open(geotiff_path) as src:
        return src.read(band_index)