"""
gee_satellite.py

Real Google Earth Engine data-fetching layer for satellite_change_detection.py.

GEE is used ONLY to pull real pixel data down to numpy arrays (or to compute
region-level stats). The disturbance-detection math remains in
satellite_change_detection.py.

REQUIRES: earthengine-api, with one-time authentication in the environment.
"""

from typing import Optional, Tuple

import numpy as np


def _require_ee():
    try:
        import ee
    except ImportError as exc:
        raise ImportError(
            "earthengine-api is not installed. Run: pip install earthengine-api"
        ) from exc
    return ee


def get_cloud_masked_ndvi_image(region, start_date: str, end_date: str, max_cloud: int = 40):
    """Return least-cloudy cloud-masked Sentinel-2 NDVI and timestamp."""
    ee = _require_ee()

    def mask_clouds(img):
        scl = img.select("SCL")
        mask = scl.remap([3, 8, 9, 10], [0, 0, 0, 0], 1)
        return img.updateMask(mask)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(mask_clouds)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    count = collection.size().getInfo()
    if count == 0:
        return None, None
    image = collection.first()
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    timestamp = image.get("system:time_start").getInfo()
    return ndvi, timestamp


def get_sentinel1_vv_db(region, start_date: str, end_date: str):
    """Return Sentinel-1 VV backscatter in dB and timestamp."""
    ee = _require_ee()
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .sort("system:time_start")
    )
    count = collection.size().getInfo()
    if count == 0:
        return None, None
    image = collection.first()
    return image.select("VV"), image.get("system:time_start").getInfo()


def region_ndvi_diff_stats(region, ndvi_before, ndvi_after) -> dict:
    """Region-level NDVI difference summary."""
    ee = _require_ee()
    diff = ndvi_after.subtract(ndvi_before).rename("NDVI_diff")
    stats = diff.reduceRegion(
        reducer=ee.Reducer.mean().combine(ee.Reducer.minMax(), sharedInputs=True),
        geometry=region,
        scale=10,
        maxPixels=1e9,
    ).getInfo()
    return stats


def region_disturbed_area_ha(region, ndvi_diff_image, drop_threshold: float = -0.25) -> float:
    """Area (hectares) where NDVI dropped by more than drop_threshold."""
    ee = _require_ee()
    disturbance_mask = ndvi_diff_image.lte(drop_threshold)
    area_m2 = disturbance_mask.multiply(ee.Image.pixelArea()).reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=10,
        maxPixels=1e9,
    ).getInfo()
    key = list(area_m2.keys())[0]
    return round(area_m2[key] / 10000.0, 2) if area_m2.get(key) else 0.0


def _reproject_to_reference(image, reference_image, resampling="bilinear"):
    """Put an EE image on exactly the same grid as a reference image."""
    if reference_image is None:
        return image
    projection = reference_image.projection()
    return image.resample(resampling).reproject(projection)


def _crop_to_common_shape(a: np.ndarray, b: np.ndarray):
    """Crop two small arrays to their common shape.

    Earth Engine scenes can differ by one pixel at their tile/footprint edge
    even when the same geographic rectangle is requested. The crop is only
    used after both images have been put on a common projection.
    """
    rows = min(a.shape[0], b.shape[0])
    cols = min(a.shape[1], b.shape[1])
    return a[:rows, :cols], b[:rows, :cols]


def sample_rectangle_as_numpy(
    image,
    region,
    scale: int = 10,
    band: Optional[str] = None,
    reference_image=None,
    resampling: str = "bilinear",
) -> np.ndarray:
    """Pull actual small-area Earth Engine pixels into numpy.

    ``reference_image`` is optional, but when supplied the image is first
    reprojected to the reference image's projection/grid. This prevents
    Sentinel-1 and Sentinel-2 (or before/after scenes) from returning arrays
    with slightly different dimensions or pixel alignment.
    """
    _require_ee()
    img = image if band is None else image.select(band)
    if reference_image is not None:
        img = _reproject_to_reference(img, reference_image, resampling)
    sampled = img.sampleRectangle(region=region, defaultValue=0)
    key = band if band else sampled.bandNames().getInfo()[0]
    array = np.array(sampled.get(key).getInfo(), dtype="float32")
    return array


def fetch_disturbance_flag_for_village(
    region,
    before_window: Tuple[str, str],
    after_window: Tuple[str, str],
    cfg,
    max_cloud: int = 50,
) -> dict:
    """Fetch real Sentinel-2/Sentinel-1 data and run the project detector."""
    from . import satellite_change_detection as scd

    ndvi_before_img, _ = get_cloud_masked_ndvi_image(
        region, *before_window, max_cloud
    )
    ndvi_after_img, _ = get_cloud_masked_ndvi_image(
        region, *after_window, max_cloud
    )

    sar_before_img, _ = get_sentinel1_vv_db(region, *before_window)
    sar_after_img, _ = get_sentinel1_vv_db(region, *after_window)

    ndvi_mask = None
    sar_mask = None
    sources_used = []

    # ------------------------------------------------------------
    # Sentinel-2: force before/after onto the BEFORE scene grid.
    # This fixes the common (225,205) vs (224,204) footprint mismatch.
    # ------------------------------------------------------------
    if ndvi_before_img is not None and ndvi_after_img is not None:
        ndvi_before_arr = sample_rectangle_as_numpy(
            ndvi_before_img,
            region,
            band="NDVI",
            reference_image=ndvi_before_img,
            resampling="bilinear",
        )
        ndvi_after_arr = sample_rectangle_as_numpy(
            ndvi_after_img,
            region,
            band="NDVI",
            reference_image=ndvi_before_img,
            resampling="bilinear",
        )
        ndvi_before_arr, ndvi_after_arr = _crop_to_common_shape(
            ndvi_before_arr, ndvi_after_arr
        )
        diff = scd.ndvi_difference(ndvi_before_arr, ndvi_after_arr)
        ndvi_mask = scd.ndvi_disturbance_mask(diff, cfg)
        sources_used.append("NDVI (Sentinel-2)")

    # ------------------------------------------------------------
    # Sentinel-1: force before/after onto the BEFORE SAR grid.
    # ------------------------------------------------------------
    if sar_before_img is not None and sar_after_img is not None:
        sar_before_arr = sample_rectangle_as_numpy(
            sar_before_img,
            region,
            band="VV",
            reference_image=sar_before_img,
            resampling="bilinear",
        )
        sar_after_arr = sample_rectangle_as_numpy(
            sar_after_img,
            region,
            band="VV",
            reference_image=sar_before_img,
            resampling="bilinear",
        )
        sar_before_arr, sar_after_arr = _crop_to_common_shape(
            sar_before_arr, sar_after_arr
        )
        sar_diff = scd.sar_log_ratio(sar_before_arr, sar_after_arr)
        sar_mask = scd.sar_disturbance_mask(sar_diff, cfg)
        sources_used.append("SAR (Sentinel-1)")

    if not sources_used:
        return {
            "status": "no_data",
            "reason": "No cloud-free optical or SAR scene found in either window.",
            "sources_used": [],
        }

    # NDVI and SAR are both nominally 10 m but can still have different
    # array dimensions/projections. Align the two detector masks before
    # combining them, using the common upper-left window.
    if ndvi_mask is not None and sar_mask is not None:
        rows = min(ndvi_mask.shape[0], sar_mask.shape[0])
        cols = min(ndvi_mask.shape[1], sar_mask.shape[1])
        ndvi_mask = ndvi_mask[:rows, :cols]
        sar_mask = sar_mask[:rows, :cols]

    combined_mask = scd.combined_disturbance_flag(ndvi_mask, sar_mask)
    summary = scd.disturbance_summary(combined_mask, cfg)
    summary.update(
        {
            "status": "ok",
            "sources_used": sources_used,
            "disturbance_detected": bool(summary["flagged_pixels"] > 0),
        }
    )
    return summary
