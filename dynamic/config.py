"""
config.py

Centralized calibration constants for the Dynamic Trigger Layer.

WHY THIS FILE EXISTS
---------------------
The architecture doc's whole pitch is "rule-augmented, auditable, not a
black box" (Section 18, point 4). That claim only holds up if every
threshold used anywhere in the pipeline lives in ONE place that a
non-programmer (an official, a judge, a teammate) can open and read —
not scattered as default arguments across a dozen function signatures.

Every district/pilot-region gets its own DistrictConfig instance. Do not
share one global threshold across districts — see the "WHY PER-DISTRICT
CALIBRATION MATTERS" note at the bottom of this file; it is not optional.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DistrictConfig:
    # --- Identity -----------------------------------------------------
    district_name: str
    station_lat: float
    station_lon: float

    # --- Rainfall ID-threshold (Section 6.1) ---------------------------
    # MVP simplified two-window thresholds (flat mm cutoffs).
    threshold_24h_mm: float = 150.0
    threshold_72h_mm: float = 300.0

    # Full power-law ID curve I = a * D^(-b) (future upgrade, not yet
    # wired into the pipeline by default — kept here so it's ready).
    id_curve_a: float = 30.0
    id_curve_b: float = 0.5

    # --- Antecedent Wetness Index / API (Section 6.2) -------------------
    awi_decay_k: float = 0.9
    awi_window_days: int = 15

    # --- Sensor fusion (blending live soil-moisture sensor with API) ---
    # 0 = trust rainfall-derived API fully, 1 = trust the sensor fully.
    sensor_blend_weight: float = 0.5

    # --- Combined TriggerScore weighting (Section 6.4) -------------------
    trigger_weight_id: float = 0.6
    trigger_weight_api: float = 0.4
    id_norm_range: Tuple[float, float] = (0.0, 3.0)
    api_norm_range: Tuple[float, float] = (0.0, 150.0)

    # --- Watch / Warning state machine (Section 6.5) ---------------------
    watch_threshold: float = 0.6
    warning_threshold: float = 0.8

    # --- Satellite change detection (Section 6.3) ------------------------
    ndvi_drop_threshold: float = -0.25       # NDVI difference flag
    sar_abs_threshold_db: float = 3.0        # SAR backscatter change flag
    pixel_area_m2: float = 900.0             # 30m x 30m Sentinel/SRTM pixel

    # --- Free-text audit note: who calibrated this and against what ----
    calibration_note: str = "Uncalibrated default — replace before field use."


# ---------------------------------------------------------------------------
# WHY PER-DISTRICT CALIBRATION MATTERS (read this before changing defaults)
# ---------------------------------------------------------------------------
# Sohra (Cherrapunji) and Mawsynram in East Khasi Hills are among the
# wettest places on Earth. A published 2024 study of East Khasi Hills
# landslides recorded rainfall >100mm on 12 of the 27 days between
# 25 May and 20 June 2024 at the Sohra block alone. A flat 100mm/24h
# threshold — reasonable for most of India — would fire a "warning" in
# that district roughly every other day of the monsoon, producing alert
# fatigue that erodes trust exactly as your architecture doc warns
# against (Section 11: "duplicate prevention... cooldown window").
#
# backtest.py demonstrates this concretely: it checks a flat 100mm
# threshold against real historical events and shows it is already
# close to firing on a day with no confirmed major landslide, while a
# recalibrated, station-specific threshold (see
# `recommend_thresholds_from_climatology` in backtest.py) separates the
# two cases correctly.
#
# DEFAULT DISTRICT CONFIGS BELOW ARE STARTING POINTS ONLY. Recalibrate
# every one of them against your pilot district's own historical
# rainfall + landslide inventory using backtest.py before using this
# in front of judges or officials as anything other than a demo.
# ---------------------------------------------------------------------------

EAST_KHASI_HILLS = DistrictConfig(
    district_name="East Khasi Hills, Meghalaya (Sohra/Mawsynram belt)",
    station_lat=25.2702,
    station_lon=91.7323,
    threshold_24h_mm=180.0,     # recalibrated — see backtest.py; flat 100mm is too low for this micro-climate
    threshold_72h_mm=350.0,
    calibration_note=(
        "Recalibrated using 2 real events (backtest.py): a confirmed "
        "landslide-triggering 242.4mm/24h spell (East Khasi Hills, "
        "Jun 2024 study) and a confirmed major landslide at 470.4mm/24h "
        "(Sohra, 21 Jun 2026, IMD). Threshold set below the smaller true "
        "event with margin, above a 96.8mm/24h day with no confirmed "
        "major landslide reported. Still only 2 data points — expand "
        "before deployment."
    ),
)

GENERIC_NER_DISTRICT = DistrictConfig(
    district_name="Generic NER pilot district (uncalibrated placeholder)",
    station_lat=25.5788,
    station_lon=91.8933,
    calibration_note="Placeholder — replace station coords and thresholds with your actual pilot district before use.",
)
