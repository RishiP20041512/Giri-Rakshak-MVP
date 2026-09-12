"""
pipeline.py

The single entry point for the Dynamic Trigger Layer. Ties together:
  - live rainfall (data_sources.py)
  - ID-threshold + Antecedent Wetness Index (rainfall_trigger.py)
  - optional sensor/SMAP wetness fusion (sensor_fusion.py)
  - optional satellite disturbance flag (gee_satellite.py / satellite_change_detection.py)

into ONE result object per location, with a full reasoning trace. This is
what a Risk Fusion Engine (architecture doc Section 8, not built yet)
would call once per grid cell each recompute cycle.

Satellite and sensor inputs are optional and independent of each other —
missing either one degrades gracefully rather than blocking the whole
pipeline, per architecture doc Section 8's "missing sensor data" rule.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .config import DistrictConfig
from .data_sources import fetch_live_rainfall
from .rainfall_trigger import (
    rolling_cumulative_rainfall,
    antecedent_wetness_index,
    evaluate_rainfall_trigger,
    RainfallTriggerResult,
)
from .sensor_fusion import get_wetness_estimate


@dataclass
class DynamicLayerResult:
    rainfall: RainfallTriggerResult
    wetness_source: str
    blended_wetness_0_1: Optional[float]
    satellite_disturbance: Optional[dict]
    final_state: str
    escalation_reason: Optional[str]

    def summary(self) -> str:
        lines = [f"=== Dynamic Trigger Layer — {self.rainfall.district} ==="]
        lines += [f"  {r}" for r in self.rainfall.reasoning]
        lines.append(f"  Wetness source used: {self.wetness_source}")
        if self.satellite_disturbance:
            lines.append(f"  Satellite disturbance check: {self.satellite_disturbance}")
        lines.append(f"  FINAL STATE: {self.final_state}"
                      + (f"  (escalated: {self.escalation_reason})" if self.escalation_reason else ""))
        return "\n".join(lines)


def apply_satellite_escalation(current_level: str, tier_order: list, satellite_result: Optional[dict]):
    """
    Shared satellite-escalation rule (architecture doc Section 8): a
    confirmed satellite disturbance bumps the current alert tier UP BY
    ONE STEP along `tier_order`, capped at the top tier — never folded
    numerically into any underlying score (Section 6.4).

    Generalized over `tier_order` so the SAME rule serves both this
    module's 3-tier normal/watch/warning state machine and a caller's
    own tier scale (e.g. the app's 4-tier LOW/MEDIUM/HIGH/CRITICAL risk
    badge) — one authoritative implementation instead of two drifting
    copies of the same policy.

    Returns (new_level, escalation_reason_or_None). A missing/absent
    `disturbance_detected` (e.g. a "no_data" satellite result) is
    treated as "nothing to escalate", not an error.
    """
    if not satellite_result or not satellite_result.get("disturbance_detected"):
        return current_level, None
    if current_level not in tier_order:
        return current_level, None
    idx = tier_order.index(current_level)
    if idx >= len(tier_order) - 1:
        return current_level, None  # already at the top tier
    new_level = tier_order[idx + 1]
    reason = f"Satellite disturbance flag forced escalation from '{current_level}' to '{new_level}'."
    return new_level, reason


def run_dynamic_layer(cfg: DistrictConfig,
                       local_iot_reading_pct: Optional[float] = None,
                       satellite_region=None,
                       satellite_before_window: Optional[tuple] = None,
                       satellite_after_window: Optional[tuple] = None,
                       at_timestamp=None,
                       rainfall_series: Optional[pd.Series] = None) -> DynamicLayerResult:
    """
    Run the full dynamic layer for one district/village config, "now"
    (or at `at_timestamp` if you're replaying history — see backtest.py).

    rainfall_series: pass a pre-fetched series to avoid a live API call
        (used heavily by backtest.py); otherwise this fetches live data
        via data_sources.fetch_live_rainfall().

    satellite_region / before_window / after_window: pass an
    ee.Geometry + two (start, end) date tuples to also run a real
    satellite disturbance check via gee_satellite.py. Skipped if None.
    """
    if rainfall_series is None:
        rainfall_series = fetch_live_rainfall(cfg.station_lat, cfg.station_lon)

    cum_df = rolling_cumulative_rainfall(rainfall_series)
    daily_rain = rainfall_series.resample("D").sum()
    api_series = antecedent_wetness_index(daily_rain, cfg)

    ts = at_timestamp if at_timestamp is not None else cum_df.index[-1]
    rainfall_result = evaluate_rainfall_trigger(cum_df, api_series, ts, cfg)

    wetness = get_wetness_estimate(
        raw_api_value=rainfall_result.api_value, cfg=cfg,
        lat=cfg.station_lat, lon=cfg.station_lon, date=str(pd.Timestamp(ts).date()),
        local_iot_reading_pct=local_iot_reading_pct,
    )

    satellite_result = None
    if satellite_region is not None and satellite_before_window and satellite_after_window:
        from . import gee_satellite
        satellite_result = gee_satellite.fetch_disturbance_flag_for_village(
            satellite_region, satellite_before_window, satellite_after_window, cfg,
        )

    final_state = rainfall_result.state
    # Satellite override: architecture doc Section 8 — a confirmed
    # satellite anomaly can force an escalation regardless of the
    # numeric TriggerScore (kept as an explicit override, not folded
    # numerically into the score, per Section 6.4).
    final_state, escalation_reason = apply_satellite_escalation(
        final_state, ["normal", "watch", "warning"], satellite_result
    )

    return DynamicLayerResult(
        rainfall=rainfall_result,
        wetness_source=wetness["source"],
        blended_wetness_0_1=wetness.get("blended_wetness_0_1"),
        satellite_disturbance=satellite_result,
        final_state=final_state,
        escalation_reason=escalation_reason,
    )