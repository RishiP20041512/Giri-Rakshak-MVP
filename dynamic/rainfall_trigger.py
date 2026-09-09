"""
rainfall_trigger.py

Rule/formula-based rainfall trigger for the Dynamic Trigger Layer
(architecture doc Section 6.1, 6.2, 6.4, 6.5).

No training, no labels. Every function is deterministic and takes its
calibration constants from a DistrictConfig (config.py) rather than
hardcoded defaults, so the whole pipeline stays auditable per-district.
"""

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

from .config import DistrictConfig


# ---------------------------------------------------------------------------
# 1. ROLLING CUMULATIVE RAINFALL
# ---------------------------------------------------------------------------
def rolling_cumulative_rainfall(rainfall_series: pd.Series, windows_hours=(1, 6, 24, 72)) -> pd.DataFrame:
    """
    rainfall_series: pandas Series of rainfall (mm), indexed by hourly DatetimeIndex.
    Returns a DataFrame with one column per rolling window, e.g. 'cum_24h'.
    """
    df = pd.DataFrame({"rainfall_mm": rainfall_series})
    for w in windows_hours:
        df[f"cum_{w}h"] = df["rainfall_mm"].rolling(window=w, min_periods=1).sum()
    return df


# ---------------------------------------------------------------------------
# 2. INTENSITY-DURATION (ID) THRESHOLD — full curve (future upgrade)
# ---------------------------------------------------------------------------
def id_threshold_curve(duration_hours: float, cfg: DistrictConfig) -> float:
    """
    Classic power-law ID threshold: I = a * D^(-b). Not used by the MVP
    pipeline (run_dynamic_layer uses simplified_two_window_trigger
    instead) — kept here, calibrated per-district, for the Phase 2+
    upgrade noted in architecture doc Section 6.1.
    """
    return cfg.id_curve_a * (duration_hours ** (-cfg.id_curve_b))


def id_exceedance_ratio(cumulative_rainfall_mm: float, duration_hours: float, cfg: DistrictConfig) -> float:
    if duration_hours <= 0:
        return 0.0
    observed_intensity = cumulative_rainfall_mm / duration_hours
    critical_intensity = id_threshold_curve(duration_hours, cfg)
    if critical_intensity <= 0:
        return 0.0
    return observed_intensity / critical_intensity


# ---------------------------------------------------------------------------
# 3. SIMPLIFIED TWO-WINDOW TRIGGER (MVP — active in the pipeline)
# ---------------------------------------------------------------------------
def simplified_two_window_trigger(cum_24h_mm: float, cum_72h_mm: float, cfg: DistrictConfig) -> float:
    """
    MVP simplification: two independently-calibrated flat thresholds,
    read from cfg.threshold_24h_mm / cfg.threshold_72h_mm (per-district,
    NOT one global number — see config.py's calibration note on why).
    Returns the worse of the two exceedance ratios.
    """
    ratio_24h = cum_24h_mm / cfg.threshold_24h_mm if cfg.threshold_24h_mm > 0 else 0.0
    ratio_72h = cum_72h_mm / cfg.threshold_72h_mm if cfg.threshold_72h_mm > 0 else 0.0
    return max(ratio_24h, ratio_72h)


# ---------------------------------------------------------------------------
# 4. ANTECEDENT WETNESS INDEX (API)
# ---------------------------------------------------------------------------
def antecedent_wetness_index(daily_rainfall_mm: pd.Series, cfg: DistrictConfig) -> pd.Series:
    """
    API_t = sum_{i=0}^{window_days} ( rainfall[t-i] * k^i )
    Approximates subsurface saturation from rainfall alone — works for
    any grid cell, even ones with no physical soil-moisture sensor.
    """
    window_days = cfg.awi_window_days
    weights = cfg.awi_decay_k ** np.arange(window_days + 1)
    api = daily_rainfall_mm.rolling(window=window_days + 1, min_periods=1).apply(
        lambda x: np.dot(x[::-1][:len(weights)], weights[:len(x)]), raw=True
    )
    return api


def calibrate_api_with_sensor(api_value_norm: float, live_soil_moisture_pct_norm: float, cfg: DistrictConfig) -> float:
    """
    Blends a live IoT soil-moisture reading with the rainfall-derived
    API estimate. Both inputs must already be normalized to 0-1 (use
    normalize()) before calling this — see sensor_fusion.py for a worked
    example including a synthetic sensor reading.
    """
    w = cfg.sensor_blend_weight
    return (1 - w) * api_value_norm + w * live_soil_moisture_pct_norm


# ---------------------------------------------------------------------------
# 5. NORMALIZATION + COMBINED TRIGGER SCORE
# ---------------------------------------------------------------------------
def normalize(value: float, min_val: float, max_val: float) -> float:
    if max_val <= min_val:
        return 0.0
    scaled = (value - min_val) / (max_val - min_val)
    return float(np.clip(scaled, 0.0, 1.0))


def combined_trigger_score(id_exceedance: float, api_value: float, cfg: DistrictConfig) -> float:
    """
    TriggerScore = w1 * normalized(ID exceedance) + w2 * normalized(API)
    Simple auditable weighted sum, not a trained model.
    """
    n_id = normalize(id_exceedance, *cfg.id_norm_range)
    n_api = normalize(api_value, *cfg.api_norm_range)
    return round(cfg.trigger_weight_id * n_id + cfg.trigger_weight_api * n_api, 4)


# ---------------------------------------------------------------------------
# 6. WATCH / WARNING STATE MACHINE
# ---------------------------------------------------------------------------
def alert_state(trigger_score: float, cfg: DistrictConfig) -> str:
    """
    "normal" -> "watch" (internal-only, Section 6.5) -> "warning" (public).
    """
    if trigger_score >= cfg.warning_threshold:
        return "warning"
    elif trigger_score >= cfg.watch_threshold:
        return "watch"
    return "normal"


# ---------------------------------------------------------------------------
# 7. RESULT OBJECT — carries an audit trail, not just a number
# ---------------------------------------------------------------------------
@dataclass
class RainfallTriggerResult:
    timestamp: object
    cum_24h_mm: float
    cum_72h_mm: float
    id_exceedance_ratio: float
    api_value: float
    trigger_score: float
    state: str
    district: str
    reasoning: list  # human-readable trace, for the "auditable, not a black box" requirement

    def as_dict(self):
        d = asdict(self)
        d["reasoning"] = list(self.reasoning)
        return d


def evaluate_rainfall_trigger(cum_df: pd.DataFrame, api_series: pd.Series,
                               at_timestamp, cfg: DistrictConfig) -> RainfallTriggerResult:
    """
    Given precomputed rolling-cumulative rainfall and API series, evaluate
    the trigger at a specific timestamp and return a fully-auditable
    result: not just the number, but *why* it landed where it did — this
    is what a district official or hackathon judge should be able to
    read and trust without touching code.
    """
    cum_24h = float(cum_df.loc[at_timestamp, "cum_24h"])
    cum_72h = float(cum_df.loc[at_timestamp, "cum_72h"])
    id_ratio = simplified_two_window_trigger(cum_24h, cum_72h, cfg)

    day = pd.Timestamp(at_timestamp).normalize()
    api_val = float(api_series.loc[day]) if day in api_series.index else 0.0

    trigger = combined_trigger_score(id_ratio, api_val, cfg)
    state = alert_state(trigger, cfg)

    reasoning = [
        f"24h cumulative rainfall = {cum_24h:.1f}mm vs district threshold {cfg.threshold_24h_mm:.0f}mm "
        f"-> ratio {cum_24h / cfg.threshold_24h_mm:.2f}",
        f"72h cumulative rainfall = {cum_72h:.1f}mm vs district threshold {cfg.threshold_72h_mm:.0f}mm "
        f"-> ratio {cum_72h / cfg.threshold_72h_mm:.2f}",
        f"ID exceedance ratio (worse of the two) = {id_ratio:.2f}",
        f"Antecedent Wetness Index = {api_val:.1f}",
        f"TriggerScore = {cfg.trigger_weight_id} x normalized(ID) + {cfg.trigger_weight_api} x normalized(AWI) = {trigger:.3f}",
        f"State = '{state}' (watch >= {cfg.watch_threshold}, warning >= {cfg.warning_threshold})",
    ]

    return RainfallTriggerResult(
        timestamp=at_timestamp, cum_24h_mm=cum_24h, cum_72h_mm=cum_72h,
        id_exceedance_ratio=round(id_ratio, 3), api_value=round(api_val, 2),
        trigger_score=trigger, state=state, district=cfg.district_name,
        reasoning=reasoning,
    )
