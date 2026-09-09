"""
data_sources.py

Real-data fetchers for the Dynamic Trigger Layer. No synthetic data here —
this module only wraps live external APIs.

  - fetch_live_rainfall():        Open-Meteo forecast API (past + next few days)
  - fetch_historical_rainfall():  Open-Meteo archive API (any past date range)

Open-Meteo needs no API key and has no auth step, which is why the notebook
used it for the real-data validation cells. It is a reanalysis/model
product, not a physical rain gauge — for a real deployment you would prefer
IMD station/API data where available and use Open-Meteo as a gap-filler
(this is exactly the "missing sensor data -> fall back to nearest
station/interpolated grid" rule from architecture doc Section 8).

Requires: `requests`, `pandas` (both already in your Colab environment).
"""

import requests
import pandas as pd


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_live_rainfall(lat: float, lon: float, past_days: int = 60,
                         forecast_days: int = 3, timezone: str = "Asia/Kolkata") -> pd.Series:
    """
    Pulls hourly precipitation for the last `past_days` plus a short
    forecast window. Returns a pandas Series (mm), indexed by hourly
    DatetimeIndex — feed this straight into rainfall_trigger.py functions.

    Raises requests.HTTPError / requests.ConnectionError on failure —
    catch these at the call site and fall back to the last cached value
    per architecture doc Section 8 ("missing sensor data" handling).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": timezone,
    }
    resp = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return pd.Series(
        data["hourly"]["precipitation"],
        index=pd.to_datetime(data["hourly"]["time"]),
        name="rainfall_mm",
    )


def fetch_historical_rainfall(lat: float, lon: float, start_date: str, end_date: str,
                               timezone: str = "Asia/Kolkata") -> pd.Series:
    """
    Pulls hourly precipitation for an arbitrary past date range
    (YYYY-MM-DD strings). Used by backtest.py to replay real events.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "precipitation",
        "timezone": timezone,
    }
    resp = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    return pd.Series(
        data["hourly"]["precipitation"],
        index=pd.to_datetime(data["hourly"]["time"]),
        name="rainfall_mm",
    )


def fetch_historical_rainfall_safe(lat: float, lon: float, start_date: str, end_date: str,
                                    fallback_peak_24h_mm: float = None) -> tuple:
    """
    Same as fetch_historical_rainfall(), but never raises. Returns
    (series_or_None, used_fallback: bool). If the live call fails (no
    network, API down, rate limit) and `fallback_peak_24h_mm` was
    provided (e.g. an IMD-reported figure from a news source), returns
    None for the series but signals the caller to use the reported
    figure directly — this is how backtest.py stays runnable even
    without internet access, while being explicit that it fell back.
    """
    try:
        series = fetch_historical_rainfall(lat, lon, start_date, end_date)
        return series, False
    except Exception as exc:  # noqa: BLE001 - deliberately broad, this is a graceful-degradation path
        print(f"[data_sources] Live fetch failed ({exc!r}); "
              f"falling back to reported figure: {fallback_peak_24h_mm} mm/24h")
        return None, True
