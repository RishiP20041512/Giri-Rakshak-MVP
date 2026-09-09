import streamlit as st
from pathlib import Path
import json
import urllib.parse
import urllib.request
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import mm as REPORT_MM
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak
    )
except Exception:
    colors = None

from risk_fusion import fuse_risk
from dynamic.config import EAST_KHASI_HILLS


# ============================================================
# GIRI-RAKSHAK — SEARCH-FIRST JUDGE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Giri-Rakshak",
    page_icon="⛰️",
    layout="wide",
)

ROOT = Path(__file__).parent

# ============================================================
# PROJECT FILES
# ============================================================

SUS_RASTER = ROOT / "processed" / "step70_8factor_susceptibility_probability.tif"
STATE_CSV = ROOT / "processed" / "step72_statewise_susceptibility.csv"
STATE_MAP_DIR = ROOT / "processed" / "state_maps"

FACTOR_RASTERS = {
    "Elevation": ROOT / "processed" / "predictors" / "elevation_250m.tif",
    "Slope": ROOT / "processed" / "predictors" / "slope_250m.tif",
    "3-Day Rainfall": ROOT / "processed" / "predictors" / "rainfall_3day_250m.tif",
    "Soil Moisture": ROOT / "processed" / "predictors" / "soil_moisture_3day_250m.tif",
    "NDVI": ROOT / "processed" / "predictors" / "ndvi_250m.tif",
    "Distance to Road": ROOT / "processed" / "predictors" / "distance_to_road_250m.tif",
    "Lineament Density": ROOT / "processed" / "predictors" / "lineament_density_250m.tif",
    "Geomorphological Origin": ROOT / "processed" / "predictors" / "geomorph_origin_250m.tif",
}

ROUTE_GPKG = ROOT / "pilot_route_data" / "pilot_roads_with_cost.gpkg"

STATES = [
    "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Sikkim", "Tripura"
]

STATE_CENTERS = {
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Sikkim": (27.5330, 88.5122),
    "Tripura": (23.9408, 91.9882),
}

# Seven real monitoring locations used by the project.
MONITORING = {
    "East Khasi Hills, Meghalaya": (25.2702, 91.7323),
    "Guwahati / Kamrup Metropolitan, Assam": (26.1445, 91.7362),
    "Senapati, Manipur": (25.2030, 94.3124),
    "Mon, Nagaland": (26.7167, 95.0667),
    "Agartala / West Tripura, Tripura": (23.8315, 91.2868),
    "Mangan, Sikkim": (27.50115, 88.53553),
    "Itanagar, Arunachal Pradesh": (27.4728, 94.9120),
}

# Historical validated snapshot retained for transparent reference only.
VALIDATED = {
    "East Khasi Hills, Meghalaya": (0.5677, 0.2069, 0.4234, "MEDIUM"),
    "Guwahati / Kamrup Metropolitan, Assam": (0.6830, 0.2089, 0.4934, "MEDIUM"),
    "Senapati, Manipur": (0.5518, 0.1983, 0.4104, "MEDIUM"),
    "Mon, Nagaland": (0.3864, 0.3676, 0.3789, "MEDIUM"),
    "Agartala / West Tripura, Tripura": (0.4184, 0.2345, 0.3448, "MEDIUM"),
    "Mangan, Sikkim": (0.8299, 0.4363, 0.6724, "HIGH"),
    "Itanagar, Arunachal Pradesh": (0.7527, 0.1880, 0.5268, "HIGH"),
}

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.hero {
    padding: 1.45rem 2rem;
    border-radius: 22px;
    background: linear-gradient(135deg,#0b3d2e,#147a55);
    color: white;
    margin-bottom: 1.1rem;
}
.hero h1 {font-size: 42px; margin: 0 0 4px 0;}
.hero p {font-size: 17px; margin: 0; opacity: .95;}
.search-card {
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    border: 1px solid #dbe7e1;
    background: #f8fbf9;
    margin-bottom: 1rem;
}
.result-title {font-size: 28px; font-weight: 700; margin-bottom: 0.2rem;}
.small-note {color:#64736b; font-size:13px;}
.risk-high {color:#c2410c; font-weight:800; font-size:24px;}
.risk-medium {color:#a16207; font-weight:800; font-size:24px;}
.risk-low {color:#15803d; font-weight:800; font-size:24px;}
.risk-critical {color:#b91c1c; font-weight:900; font-size:24px;}

.forecast-card {background:#f8fafc;border:1px solid #dbe4ec;border-radius:14px;padding:10px 6px;text-align:center;min-height:190px;}
.forecast-day {font-weight:800;font-size:15px;}
.forecast-date {font-size:12px;color:#64748b;}
.forecast-icon {font-size:34px;margin:12px 0;}
.forecast-rain {font-size:12px;margin:4px 0;}
.forecast-risk {font-size:21px;font-weight:900;margin-top:10px;color:#0f5132;}
.forecast-level {font-size:11px;font-weight:800;letter-spacing:.05em;}
.monitor-card {background:#071b2e;color:white;border-radius:16px;padding:16px 18px;margin-bottom:12px;min-height:125px;box-shadow:0 6px 18px rgba(0,0,0,.12);}
.monitor-head {display:flex;justify-content:space-between;color:#dbeafe;font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:.06em;}
.monitor-dot {color:#f97316;}
.monitor-place {font-size:16px;font-weight:800;margin-top:13px;}
.monitor-trigger {font-size:11px;color:#cbd5e1;margin-top:3px;}
.monitor-score {font-size:28px;font-weight:900;text-align:right;margin-top:-28px;}
.monitor-meta {font-size:11px;color:#cbd5e1;margin-top:18px;}

.route-card {border-radius:16px;padding:18px;margin-bottom:12px;min-height:250px;border:2px solid #dbe4ec;background:#fff;}
.route-card.safe {border-color:#16a34a;}
.route-card.caution {border-color:#eab308;}
.route-card.danger {border-color:#dc2626;}
.route-title {font-size:19px;font-weight:900;}
.route-sub {font-size:12px;font-weight:800;margin:5px 0 15px;}
.route-info {font-size:13px;line-height:1.7;}
.route-action {margin-top:15px;padding:10px;border-radius:9px;text-align:center;font-weight:900;background:#166534;color:white;}
.route-card.caution .route-action {background:#a16207;}
.route-card.danger .route-action {background:#b91c1c;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>⛰️ Giri-Rakshak</h1>
    <p>AI-Based Landslide Susceptibility • Dynamic Warning • Satellite Monitoring • Route Awareness</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def classify_sus(v):
    if v is None or not np.isfinite(v):
        return "UNAVAILABLE"
    if v < .20: return "VERY LOW"
    if v < .40: return "LOW"
    if v < .60: return "MODERATE"
    if v < .80: return "HIGH"
    return "VERY HIGH"


def risk_badge(level):
    return {
        "LOW": "🟢 LOW",
        "MEDIUM": "🟡 MEDIUM",
        "HIGH": "🟠 HIGH",
        "CRITICAL": "🔴 CRITICAL",
        "UNAVAILABLE": "⚪ UNAVAILABLE",
    }.get(str(level).upper(), str(level))


def risk_class_css(level):
    return {
        "LOW": "risk-low",
        "MEDIUM": "risk-medium",
        "HIGH": "risk-high",
        "CRITICAL": "risk-critical",
    }.get(str(level).upper(), "")


def value_at_raster(path, lat, lon):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(x[0], y[0])
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return None
        v = float(src.read(1)[row, col])
        if src.nodata is not None and np.isclose(v, src.nodata):
            return None
        if not np.isfinite(v):
            return None
        return v


def get_real_sus(lat, lon):
    return value_at_raster(SUS_RASTER, lat, lon)


def read_sus_crop(lat, lon, radius_km=12):
    if not SUS_RASTER.exists():
        return None
    with rasterio.open(SUS_RASTER) as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
        cx, cy = x[0], y[0]
        r = radius_km * 1000
        win = from_bounds(cx-r, cy-r, cx+r, cy+r, src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        arr = src.read(1, window=win)
        tr = src.window_transform(win)
        if src.nodata is not None:
            arr = np.ma.masked_where(arr == src.nodata, arr)
        else:
            arr = np.ma.masked_invalid(arr)
        return arr, tr


def make_susceptibility_plot(lat, lon, name):
    result = read_sus_crop(lat, lon, 12)
    if result is None:
        return None
    arr, tr = result
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        arr,
        extent=(
            tr.c,
            tr.c + tr.a * arr.shape[1],
            tr.f + tr.e * arr.shape[0],
            tr.f,
        ),
        vmin=0,
        vmax=1,
        cmap="RdYlGn_r",
        interpolation="bilinear",  # smoother visual; underlying data stay 250 m
    )
    with rasterio.open(SUS_RASTER) as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
    ax.scatter(
        x[0], y[0], s=100, marker="*",
        edgecolor="black", linewidth=1.2, zorder=5
    )
    ax.set_title(
        f"{name} — Local RF Susceptibility",
        fontsize=15, fontweight="bold"
    )
    ax.set_xlabel("Model X")
    ax.set_ylabel("Model Y")
    fig.colorbar(im, ax=ax, label="RF susceptibility score (0–1)")
    fig.tight_layout()
    return fig


@st.cache_data(ttl=1800)
def geocode(q):
    """Real Northeast India geocoding with a few query fallbacks."""
    queries = [q.strip()]
    if "," in q:
        parts = [p.strip() for p in q.split(",") if p.strip()]
        if parts:
            queries.append(parts[0])
        if len(parts) > 1:
            queries.append(f"{parts[0]}, {parts[1]}")
    else:
        queries.append(f"{q}, India")

    results = []
    seen = set()
    state_lower = {s.lower() for s in STATES}
    for query in queries:
        params = {
            "name": query,
            "count": 50,
            "language": "en",
            "format": "json",
            "countryCode": "IN",
        }
        url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode())
        except Exception:
            continue
        for x in data.get("results", []):
            if str(x.get("country_code", "")).upper() != "IN":
                continue
            admin1 = str(x.get("admin1", "")).strip()
            name = str(x.get("name", "")).strip()
            admin2 = str(x.get("admin2", "")).strip()
            # Open-Meteo usually puts the Indian state in admin1.
            if admin1.lower() not in state_lower and name.lower() not in state_lower:
                continue
            key = (name.lower(), admin1.lower(), round(float(x.get("latitude", 0)), 4), round(float(x.get("longitude", 0)), 4))
            if key not in seen:
                seen.add(key)
                x["_admin2"] = admin2
                results.append(x)

    # Prefer an exact name match, then a result whose state matches the user's text.
    qlow = q.lower().strip()
    exact = [x for x in results if str(x.get("name", "")).lower() == qlow]
    if exact:
        return exact + [x for x in results if x not in exact]
    return results


@st.cache_data(ttl=600)
def weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "past_days": 15,
        "forecast_days": 7,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,surface_pressure",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,rain,precipitation_probability,weather_code,wind_speed_10m,surface_pressure",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode())


def rain_history(payload):
    """Use only hours up to the current hour for recent rainfall."""
    h = payload["hourly"]
    times = pd.to_datetime(h["time"])
    rain = pd.Series(h["precipitation"], dtype=float).fillna(0.0)
    now = pd.Timestamp(payload.get("current", {}).get("time"))
    if pd.isna(now):
        now = times.iloc[0]
    mask = times <= now
    past = rain[mask]
    if past.empty:
        past = rain.iloc[:1]
    return float(past.iloc[-24:].sum()), float(past.iloc[-72:].sum())


def antecedent_wetness_index(payload, decay=0.9, window_days=15):
    """Real rainfall-derived AWI from historical daily precipitation only."""
    daily = payload.get("daily", {})
    dates = pd.to_datetime(daily.get("time", []))
    rain = pd.Series(daily.get("precipitation_sum", []), dtype=float).fillna(0.0)
    if len(dates) == 0 or rain.empty:
        return None
    now = pd.Timestamp(payload.get("current", {}).get("time"))
    if pd.isna(now):
        now = dates.max()
    mask = dates.normalize() <= now.normalize()
    hist = rain[mask].tail(window_days).to_numpy(dtype=float)
    if hist.size == 0:
        return None
    weights = decay ** np.arange(hist.size - 1, -1, -1, dtype=float)
    return float(np.sum(hist * weights))


def dynamic_trigger(r24, r72, threshold24=150, threshold72=300):
    ratio24 = r24 / max(threshold24, 1e-9)
    ratio72 = r72 / max(threshold72, 1e-9)
    exceed = max(ratio24, ratio72)
    return max(0, min(1, exceed / 3)), ratio24, ratio72


def satellite_map(lat, lon, name, zoom=12):
    if folium is None:
        return None
    m = folium.Map(location=[lat, lon], zoom_start=zoom, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Satellite imagery",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="OpenStreetMap",
        name="Road map",
        overlay=False,
        control=True,
    ).add_to(m)
    folium.Marker(
        [lat, lon],
        tooltip=name,
        popup=f"{name}<br>RF susceptibility: {get_real_sus(lat, lon)}",
        icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(m)
    folium.LayerControl().add_to(m)
    return m


@st.cache_data(ttl=1800)
def load_route_roads():
    if gpd is None or not ROUTE_GPKG.exists():
        return None
    roads = gpd.read_file(ROUTE_GPKG, layer="pilot_roads_cost")
    roads["geometry"] = roads.geometry.simplify(0.00008, preserve_topology=True)
    return roads


@st.cache_data(ttl=600)
def live_monitoring_rows():
    """Show the project's validated snapshot values exactly; live rainfall remains live."""
    rows = []
    for label, (lat, lon) in MONITORING.items():
        snap = VALIDATED.get(label)
        try:
            w = weather(lat, lon)
            r24, r72 = rain_history(w)
        except Exception:
            r24, r72 = None, None
        if snap is not None:
            s, trig, r, level = snap
        else:
            s, trig, r, level = None, None, None, "UNAVAILABLE"
        rows.append({"Location": label, "S": s, "T": trig, "Final R": r,
                     "Risk": level, "24h rain (mm)": r24, "72h rain (mm)": r72})
    return pd.DataFrame(rows)


def state_mean(state):
    """Read the actual mean_susceptibility column, never the first numeric column."""
    if not STATE_CSV.exists():
        return None
    try:
        sdf = pd.read_csv(STATE_CSV)
        name_col = "state" if "state" in sdf.columns else sdf.columns[0]
        mean_col = next((c for c in sdf.columns if str(c).strip().lower() in {"mean_susceptibility", "mean susceptibility", "mean_sus"}), None)
        if mean_col is None:
            return None
        target = state.lower().replace("ā", "a").replace("ī", "i").replace("ṅ", "n").replace("ē", "e").replace("ḷ", "l")
        for _, row in sdf.iterrows():
            raw = str(row[name_col]).lower()
            cleaned = raw.replace("ā", "a").replace("ī", "i").replace("ṅ", "n").replace("ē", "e").replace("ḷ", "l")
            if target == cleaned or target in cleaned:
                v = float(row[mean_col])
                return v if np.isfinite(v) else None
    except Exception:
        pass
    return None


def weather_icon(code, rain_mm=0):
    code = int(code) if code is not None else 0
    if rain_mm > 10 or code in {61, 63, 65, 80, 81, 82, 95, 96, 99}:
        return "🌧️"
    if code in {51, 53, 55, 56, 57}:
        return "🌦️"
    if code in {1, 2, 3}:
        return "⛅"
    if code in {45, 48}:
        return "🌫️"
    return "☀️"


def forecast_risk_rows(payload, susceptibility, threshold24, threshold72):
    """Build the next 7 calendar days from the real forecast, excluding past days."""
    daily = payload["daily"]
    rain = np.asarray(daily.get("precipitation_sum", []), dtype=float)
    prob = np.asarray(daily.get("precipitation_probability_max", []), dtype=float)
    codes = np.asarray(daily.get("weather_code", np.zeros(len(rain))), dtype=float)
    dates = pd.to_datetime(daily.get("time", []))
    current_date = pd.to_datetime(payload.get("current", {}).get("time", str(dates[0] if len(dates) else "today"))).normalize()
    valid_idx = [i for i, d in enumerate(dates) if d.normalize() >= current_date]
    valid_idx = valid_idx[:7]
    rows = []
    for i in valid_idx:
        r24 = max(float(rain[i]), 0.0)
        start_i = max(0, i - 2)
        r72 = float(np.nansum(rain[start_i:i + 1]))
        ratio24 = r24 / max(threshold24, 1e-9)
        ratio72 = r72 / max(threshold72, 1e-9)
        trigger = max(0.0, min(1.0, max(ratio24, ratio72) / 3.0))
        risk = None if susceptibility is None else float(np.clip(0.6 * susceptibility + 0.4 * trigger, 0, 1))
        if risk is None:
            level = "UNAVAILABLE"
        elif risk >= .75:
            level = "CRITICAL"
        elif risk >= .50:
            level = "HIGH"
        elif risk >= .25:
            level = "MEDIUM"
        else:
            level = "LOW"
        rows.append({
            "date": dates[i].strftime("%a"),
            "date_full": dates[i].strftime("%d %b"),
            "code": int(codes[i]),
            "icon": weather_icon(codes[i], r24),
            "rain": r24,
            "prob": int(np.clip(prob[i] if i < len(prob) else 0, 0, 100)),
            "risk": risk,
            "risk_pct": None if risk is None else round(risk * 100),
            "level": level,
        })
    return rows


def render_forecast_cards(rows):
    st.subheader("🌦️ 7-Day Weather + Landslide Risk Forecast")
    st.caption("Real Open-Meteo forecast. Rainfall updates the operational landslide-risk indicator; this is not a calibrated probability of a landslide.")
    cols = st.columns(7)
    for col, row in zip(cols, rows):
        with col:
            risk_text = "—" if row["risk_pct"] is None else f"{row['risk_pct']}%"
            html = f"""<div class="forecast-card">
<div class="forecast-day">{row['date']}</div>
<div class="forecast-date">{row['date_full']}</div>
<div class="forecast-icon">{row['icon']}</div>
<div class="forecast-rain">💧 {row['rain']:.1f} mm</div>
<div class="forecast-rain">☔ {row['prob']}% rain</div>
<div class="forecast-risk">{risk_text}</div>
<div class="forecast-level">{row['level']}</div>
</div>"""
            st.markdown(html, unsafe_allow_html=True)


def render_priority_monitoring(df):
    st.subheader("📡 Priority Monitoring")
    st.caption("Seven real project monitoring locations • live weather-derived risk status")
    for start in range(0, len(df), 3):
        cols = st.columns(3)
        for col, (_, row) in zip(cols, df.iloc[start:start + 3].iterrows()):
            level = str(row.get("Risk", "UNAVAILABLE"))
            score = row.get("Final R")
            score_text = "—" if pd.isna(score) else f"{float(score):.3f}"
            loc_name = str(row["Location"])
            rain24 = row.get("24h rain (mm)", "—")
            trigger = row.get("T", "—")
            with col:
                html = f"""<div class="monitor-card">
<div class="monitor-head"><span>Real-Time Risk</span><span class="monitor-dot">●</span></div>
<div class="monitor-place">{loc_name}</div>
<div class="monitor-trigger">Trigger: {level}</div>
<div class="monitor-score">{score_text}</div>
<div class="monitor-meta">🌧️ {rain24} mm / 24h &nbsp; • &nbsp; T {trigger}</div>
</div>"""
                st.markdown(html, unsafe_allow_html=True)


def _norm_name(value):
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def road_coordinates(row):
    try:
        pt = row.geometry.representative_point()
        return float(pt.y), float(pt.x)
    except Exception:
        return None, None


def route_options_for_area(roads, district, trigger):
    if roads is None or roads.empty or "district" not in roads.columns:
        return []
    r = roads[roads["district"].astype(str).map(_norm_name) == _norm_name(district)].copy()
    if r.empty:
        return []
    for col in ["LSI_mean", "length_km", "base_travel_time_min", "routing_cost_min", "historical_event_indicator"]:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")
    if "LSI_mean" not in r.columns:
        r["LSI_mean"] = 0.5
    r["LSI_mean"] = r["LSI_mean"].fillna(0.5)
    r["display_risk"] = np.clip(0.6 * r["LSI_mean"] + 0.4 * float(trigger), 0, 1)
    r["display_status"] = r["routing_status"].astype(str).str.upper() if "routing_status" in r.columns else "AVAILABLE"
    r["road_label"] = r["road_name"].fillna("Unnamed OSM road").astype(str) if "road_name" in r.columns else "Unnamed OSM road"
    r["road_label"] = r["road_label"].replace({"": "Unnamed OSM road", "nan": "Unnamed OSM road"})
    named = r[r["road_label"].str.lower() != "unnamed osm road"]
    safe_pool = named if not named.empty else r[r["display_status"] != "BLOCKED"]
    safe_pool = safe_pool.sort_values(["display_risk", "routing_cost_min"], na_position="last")
    safe = safe_pool.iloc[0] if not safe_pool.empty else r.sort_values("display_risk").iloc[0]
    danger = r.sort_values(["display_risk", "LSI_mean"], ascending=False, na_position="last").iloc[0]
    alt_pool = safe_pool[safe_pool.index != safe.name]
    alt = alt_pool.iloc[0] if not alt_pool.empty else safe
    out=[]; seen=set()
    for row in [safe, alt, danger]:
        if row.name not in seen:
            out.append(row); seen.add(row.name)
    return out


def make_pdf_report(loc, state, sus, trig, risk_score, risk_level, cur, r24, r72,
                    th24, th72, factor_rows, state_map_path, local_fig, forecast_rows):
    if colors is None:
        return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=14*REPORT_MM, leftMargin=14*REPORT_MM,
        topMargin=14*REPORT_MM, bottomMargin=14*REPORT_MM,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
                            fontSize=20, leading=24, spaceAfter=8)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13,
                        leading=16, spaceBefore=8, spaceAfter=5)
    body = styles["BodyText"]
    story = []

    story.append(Paragraph("Giri-Rakshak", title))
    story.append(Paragraph("Landslide Risk Intelligence Report", title))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Location:</b> {loc['name']} — {loc.get('admin2','')}, {state}<br/>"
        f"<b>Latitude:</b> {loc['lat']:.5f}<br/>"
        f"<b>Longitude:</b> {loc['lon']:.5f}<br/>"
        f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}", body))

    story.append(Paragraph("Current Risk Summary", h2))
    risk_data = [["Static Susceptibility", "Dynamic Trigger", "Operational Risk", "Level"],
                 ["Unavailable" if sus is None else f"{sus:.4f}", f"{trig:.4f}",
                  "Unavailable" if risk_score is None else f"{risk_score:.4f}", risk_level]]
    t = Table(risk_data, colWidths=[40*REPORT_MM]*4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .5, colors.grey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(t)

    story.append(Paragraph("Live Weather", h2))
    dyn_data = [
        ["Temperature", "Humidity", "Pressure", "Wind"],
        [f"{cur['temperature_2m']:.1f} °C", f"{cur['relative_humidity_2m']:.0f}%",
         f"{cur['surface_pressure']:.0f} hPa", f"{cur['wind_speed_10m']:.1f} km/h"],
        ["Rain now", "Recent 24h", "Recent 72h", "Thresholds"],
        [f"{cur['rain']:.1f} mm", f"{r24:.1f} mm", f"{r72:.1f} mm",
         f"{th24:.0f}/{th72:.0f} mm"],
    ]
    td = Table(dyn_data, colWidths=[40*REPORT_MM]*4)
    td.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), .5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f2ed")),
        ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#e8f2ed")),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,2), (-1,2), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(td)

    story.append(Paragraph("7-Day Weather + Landslide Risk Forecast", h2))
    fc = [["Day", "Weather", "Rain (mm)", "Rain chance", "Risk indicator", "Status"]]
    for r in forecast_rows:
        fc.append([r["date_full"], r["icon"], f"{r['rain']:.1f}", f"{r['prob']}%",
                   "—" if r["risk_pct"] is None else f"{r['risk_pct']}%", r["level"]])
    tfc = Table(fc, colWidths=[27*REPORT_MM, 22*REPORT_MM, 27*REPORT_MM, 27*REPORT_MM, 32*REPORT_MM, 28*REPORT_MM])
    tfc.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
    ]))
    story.append(tfc)
    story.append(Paragraph("Risk indicator is an operational decision-support score, not a calibrated landslide probability.", body))

    # User requested the eight-factor table in the PDF, not the main dashboard.
    story.append(Paragraph("Eight RF Conditioning Factors", h2))
    ft = [["Factor", "Observed model input"]] + [[str(r["Factor"]), str(r["Observed model input"])] for r in factor_rows]
    tf = Table(ft, colWidths=[65*REPORT_MM, 95*REPORT_MM])
    tf.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .4, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    story.append(tf)

    if state_map_path.exists():
        story.append(PageBreak())
        story.append(Paragraph(f"{state} — State Susceptibility", h2))
        story.append(RLImage(str(state_map_path), width=170*REPORT_MM, height=125*REPORT_MM))
    if local_fig is not None:
        local_buf = BytesIO()
        local_fig.savefig(local_buf, format="png", dpi=150, bbox_inches="tight")
        local_buf.seek(0)
        story.append(Paragraph(f"{loc['name']} — Local Susceptibility", h2))
        story.append(RLImage(local_buf, width=170*REPORT_MM, height=112*REPORT_MM))

    story.append(Paragraph("Model Validation", h2))
    story.append(Paragraph(
        "Production Random Forest: 700 trees, 8 predictors, 250 m grid. "
        "Independent evaluation on 74 unseen Meghalaya locations: ROC-AUC 0.8355, "
        "PR-AUC 0.7535, Balanced Accuracy 0.7829, training samples 991.", body))
    story.append(Paragraph(
        "Data note: live weather is fetched from Open-Meteo. The displayed risk indicator is a decision-support score and should not be interpreted as a calibrated probability of a landslide.", body))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# 1. SEARCH — FIRST SCREEN
# ============================================================

st.header("🔎 Search Northeast India")

with st.container():
    q = st.text_input(
        "Search city / district / town / village",
        placeholder="Example: Cherrapunji, Meghalaya",
        label_visibility="collapsed",
    )

    search_query = q.strip()

    if st.button("🔍 SEARCH LOCATION", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a Northeast India location.")
        else:
            try:
                results = geocode(search_query)
                if not results:
                    st.error("No valid Northeast India result found. Try a city, district, or village with its state.")
                else:
                    # Prefer an exact state result when the user searched only a state.
                    exact_state = None
                    for r in results:
                        if str(r.get("name", "")).strip().lower() in {s.lower() for s in STATES}:
                            exact_state = r
                            break
                    chosen = exact_state if exact_state is not None else results[0]
                    st.session_state["location"] = {
                        "name": chosen.get("name", search_query),
                        "lat": float(chosen["latitude"]),
                        "lon": float(chosen["longitude"]),
                        "admin1": chosen.get("admin1", ""),
                        "admin2": chosen.get("admin2", ""),
                        "state_only": str(chosen.get("name", "")).strip().lower() in {s.lower() for s in STATES},
                    }
                    st.session_state["searched"] = True
                    st.rerun()
            except Exception as e:
                st.error(f"Search failed: {e}")


# ============================================================
# 2. DON'T SHOW MAPS BEFORE SEARCH
# ============================================================

if "location" not in st.session_state or not st.session_state.get("searched", False):
    st.info("🔎 Search a Northeast India location to begin the analysis.")
    st.stop()

loc = st.session_state["location"]
state = str(loc.get("admin1", "")).strip()
if state not in STATES and loc.get("name") in STATES:
    state = loc["name"]

# For a state-only search, use the state center for the local satellite/risk point.
if loc.get("state_only") and state in STATE_CENTERS:
    loc["lat"], loc["lon"] = STATE_CENTERS[state]
    loc["admin1"] = state
    loc["admin2"] = "State-level search"

st.markdown("---")
st.markdown(
    f'<div class="result-title">📍 {loc["name"]}</div>'
    f'<div class="small-note">{loc.get("admin2", "")}, {state} • '
    f'{loc["lat"]:.5f}, {loc["lon"]:.5f}</div>',
    unsafe_allow_html=True,
)


# ============================================================
# 3. COMPUTE SEARCHED LOCATION VALUES
# ============================================================

sus = get_real_sus(loc["lat"], loc["lon"])

try:
    w = weather(loc["lat"], loc["lon"])
    cur = w["current"]
    r24, r72 = rain_history(w)
except Exception as e:
    w = None
    cur = None
    r24, r72 = 0.0, 0.0
    weather_error = str(e)
else:
    weather_error = None

is_east_khasi = (
    abs(loc["lat"] - EAST_KHASI_HILLS.station_lat) < 0.15
    and abs(loc["lon"] - EAST_KHASI_HILLS.station_lon) < 0.15
)

if is_east_khasi:
    th24 = float(EAST_KHASI_HILLS.threshold_24h_mm)
    th72 = float(EAST_KHASI_HILLS.threshold_72h_mm)
    threshold_note = "East Khasi Hills pilot configuration"
else:
    th24 = 150.0
    th72 = 300.0
    threshold_note = "Generic NER prototype thresholds — not locally calibrated"

trig, ratio24, ratio72 = dynamic_trigger(r24, r72, th24, th72)

if sus is not None:
    fusion = fuse_risk(sus, trig)
    risk_score = fusion.risk_score
    risk_level = fusion.risk_level
else:
    fusion = None
    risk_score = None
    risk_level = "UNAVAILABLE"


# ============================================================
# 4. TOP RISK INTELLIGENCE — DYNAMIC FIRST
# ============================================================

st.markdown("---")
st.header("🚨 Current Risk Intelligence")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Static Susceptibility (S)", "Unavailable" if sus is None else f"{sus:.4f}")
c2.metric("Dynamic Trigger (T)", f"{trig:.4f}")
c3.metric("Final Risk (R)", "Unavailable" if risk_score is None else f"{risk_score:.4f}")
c4.metric("Risk Level", risk_badge(risk_level))

if risk_level in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
    st.markdown(
        f'<div class="{risk_class_css(risk_level)}">{risk_badge(risk_level)}</div>',
        unsafe_allow_html=True,
    )

if weather_error:
    st.warning(f"Live weather could not be loaded: {weather_error}")


# ============================================================
# 5. SIDE-BY-SIDE MAPS
# ============================================================

st.markdown("---")
st.header("🗺️ Geospatial Intelligence")

map_left, map_right = st.columns(2)

with map_left:
    st.subheader(f"🛰️ {loc['name']} — Satellite View")
    if folium is None:
        st.error("Install folium and streamlit-folium to enable the interactive satellite map.")
    else:
        sat = satellite_map(loc["lat"], loc["lon"], loc["name"], zoom=12 if not loc.get("state_only") else 7)
        st_folium(sat, use_container_width=True, height=500)
        st.caption("Satellite imagery + Road Map toggle. The red marker is the searched location.")

        sat_html = sat.get_root().render()
        st.download_button(
            "⬇️ Download Satellite Map",
            data=sat_html,
            file_name=f'{loc["name"].replace(" ", "_")}_satellite_map.html',
            mime="text/html",
            key="download_satellite",
        )

with map_right:
    st.subheader(f"🛰️ {state} — Satellite + RF View")
    state_map_path = STATE_MAP_DIR / (state.replace(" ", "_") + "_susceptibility_map.png")
    sat_tab, rf_tab = st.tabs(["🛰️ SATELLITE", "🗺️ RF SUSCEPTIBILITY"])
    with sat_tab:
        if folium is None:
            st.error("Install folium and streamlit-folium to enable the satellite view.")
        else:
            sat_state = satellite_map(loc["lat"], loc["lon"], loc["name"], zoom=9 if state == "Sikkim" else 7)
            st_folium(sat_state, use_container_width=True, height=500, key="state_satellite_view_v6")
            st.caption("🛰️ Real satellite basemap. Use the map controls to zoom and switch layers.")
    with rf_tab:
        if state == "Mizoram":
            st.warning("RF susceptibility is currently unavailable for Mizoram because production predictor coverage is incomplete.")
        if state_map_path.exists():
            st.image(state_map_path, use_container_width=True)
            with open(state_map_path, "rb") as f:
                st.download_button("⬇️ Download State Map PNG", data=f.read(), file_name=state_map_path.name, mime="image/png", key="download_state_map_v6")
        else:
            st.warning("State susceptibility map PNG not found.")
    mean_s = state_mean(state)
    if mean_s is not None:
        st.metric("State mean susceptibility", f"{mean_s:.3f}")


# ============================================================
# 6. LOCAL SUSCEPTIBILITY — ONLY WHEN A LOCATION WAS SEARCHED
# ============================================================

if not loc.get("state_only"):
    st.markdown("---")
    st.header(f"🎯 {loc['name']} — Local Susceptibility")

    if sus is None:
        st.error("No valid RF susceptibility prediction exists at this exact location.")
        local_fig = None
    else:
        local_fig = make_susceptibility_plot(
            loc["lat"], loc["lon"], loc["name"]
        )
        if local_fig is not None:
            st.pyplot(local_fig, use_container_width=True)
            png_buffer = BytesIO()
            local_fig.savefig(
                png_buffer,
                format="png",
                dpi=200,
                bbox_inches="tight",
            )
            png_buffer.seek(0)
            st.download_button(
                "⬇️ Download Local Susceptibility PNG",
                data=png_buffer.getvalue(),
                file_name=f'{loc["name"].replace(" ", "_")}_susceptibility_map.png',
                mime="image/png",
                key="download_local_sus",
            )
else:
    local_fig = None


# ============================================================
# 7. EIGHT STATIC FACTORS — PDF ONLY
# ============================================================

factor_rows = []
for name, path in FACTOR_RASTERS.items():
    v = value_at_raster(path, loc["lat"], loc["lon"])
    if v is None:
        display = "Unavailable"
    elif name == "Elevation":
        display = f"{v:.2f} m"
    elif name == "Slope":
        display = f"{v:.2f}°"
    elif name == "3-Day Rainfall":
        display = f"{v:.2f} mm"
    elif name == "Soil Moisture":
        display = f"{v:.3f}"
    elif name == "NDVI":
        display = f"{v:.3f}"
    elif name == "Distance to Road":
        display = f"{v:.1f} m"
    elif name == "Lineament Density":
        display = f"{v:.4f}"
    else:
        geom = {1: "Denudational Origin", 2: "Fluvial Origin", 3: "Glacial Origin",
                4: "Lacustrine Origin", 5: "Structural Origin", 6: "Water Bodies"}
        display = geom.get(round(v), str(round(v)))
    factor_rows.append({"Factor": name, "Observed model input": display})

# ============================================================
# 8. DYNAMIC CONDITIONS
# ============================================================

st.markdown("---")
st.header(f"🌧️ Dynamic Trigger — {loc['name']}")

if cur is not None:
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Temperature", f"{cur['temperature_2m']:.1f} °C")
    w2.metric("Humidity", f"{cur['relative_humidity_2m']:.0f}%")
    w3.metric("Pressure", f"{cur['surface_pressure']:.0f} hPa")
    w4.metric("Wind speed", f"{cur['wind_speed_10m']:.1f} km/h")

    w5, w6, w7, w8 = st.columns(4)
    w5.metric("Rain now", f"{cur['rain']:.1f} mm")
    w6.metric("Recent 24h rain", f"{r24:.1f} mm")
    w7.metric("Recent 72h rain", f"{r72:.1f} mm")
    w8.metric("Rain probability", f"{max(w['hourly']['precipitation_probability'][:24]):.0f}%")

st.subheader("🎯 Rainfall Trigger")
d1, d2, d3 = st.columns(3)
d1.metric("24h threshold", f"{th24:.0f} mm")
d2.metric("72h threshold", f"{th72:.0f} mm")
d3.metric("Trigger score (T)", f"{trig:.4f}")
st.caption(threshold_note)

awi = antecedent_wetness_index(w) if w is not None else None
awi_col, _ = st.columns([1, 3])
with awi_col:
    awi_text = "—" if awi is None else f"{awi:.1f}"
    st.markdown(f'''<div class="route-card" style="border-color:#0f766e;min-height:120px;">
<div class="route-title">💧 ANTECEDENT WETNESS</div>
<div class="route-sub">Current AWI</div>
<div style="font-size:34px;font-weight:900;color:#0f766e;">{awi_text}</div>
<div class="small-note">15-day rainfall decay (k = 0.9)</div>
</div>''', unsafe_allow_html=True)


# ============================================================
# 9. FORECAST
# ============================================================

if w is not None:
    st.markdown("---")
    forecast_rows = forecast_risk_rows(w, sus, th24, th72)
    render_forecast_cards(forecast_rows)
else:
    forecast_rows = []


# ============================================================
# 10. SEVEN-POINT LIVE MONITORING
# ============================================================

st.markdown("---")
st.header("📡 7-Point NER Dynamic Monitoring")
monitor_df = live_monitoring_rows()
render_priority_monitoring(monitor_df)

# ============================================================
# 12. REPORT DOWNLOADS
# ============================================================

st.markdown("---")
st.header("📄 Download Location Report")

report_bytes = make_pdf_report(
    loc=loc,
    state=state,
    sus=sus,
    trig=trig,
    risk_score=risk_score,
    risk_level=risk_level,
    cur=cur if cur is not None else {
        "temperature_2m": 0,
        "relative_humidity_2m": 0,
        "surface_pressure": 0,
        "wind_speed_10m": 0,
        "rain": 0,
    },
    r24=r24,
    r72=r72,
    th24=th24,
    th72=th72,
    factor_rows=factor_rows,
    state_map_path=STATE_MAP_DIR / (state.replace(" ", "_") + "_susceptibility_map.png"),
    local_fig=local_fig,
    forecast_rows=forecast_rows,
)

if report_bytes is not None:
    st.download_button(
        "📄 Download Complete Risk Report (PDF)",
        data=report_bytes,
        file_name=f'Giri_Rakshak_{loc["name"].replace(" ", "_")}_Report.pdf',
        mime="application/pdf",
        use_container_width=True,
    )
else:
    st.info("Install reportlab to enable PDF report generation.")


# ============================================================
# 13. ROUTE AWARENESS — THREE PILOT AREAS

st.markdown("---")
st.header("🛣️ Landslide-Aware Route Options")
st.caption("Pilot route intelligence: Kamrup Metropolitan • East Khasi Hills • West Tripura")

if gpd is None:
    st.info("Install geopandas to display the route network.")
elif not ROUTE_GPKG.exists():
    st.error("The real pilot road GeoPackage is missing. Copy pilot_route_data/pilot_roads_with_cost.gpkg into the project folder.")
else:
    route_districts = {
        "Kamrup Metropolitan, Assam": ("Kamrup Metropolitan", "Assam"),
        "East Khasi Hills, Meghalaya": ("East Khasi Hills", "Meghalaya"),
        "West Tripura, Tripura": ("West Tripura", "Tripura"),
    }
    current_area = next((label for label, (d, sname) in route_districts.items()
                         if _norm_name(loc.get("admin2", "")) == _norm_name(d)
                         and _norm_name(loc.get("admin1", "")) == _norm_name(sname)), None)
    route_choice = st.selectbox(
        "Route pilot area", list(route_districts.keys()),
        index=list(route_districts.keys()).index(current_area) if current_area else 0,
        key="route_pilot_area_v6")
    district, st_name = route_districts[route_choice]
    roads = load_route_roads()
    options = route_options_for_area(roads, district, trig)

    if options:
        if folium is not None:
            rm = folium.Map(location=STATE_CENTERS[st_name], zoom_start=10, control_scale=True)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery", name="🛰️ Satellite", overlay=False, control=True).add_to(rm)
            folium.TileLayer(
                tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attr="OpenStreetMap", name="Road Map", overlay=False, control=True).add_to(rm)
            route_colors = {"AVAILABLE":"green","CAUTION":"orange","AVOID_IF_ALTERNATIVE":"red","BLOCKED":"black"}
            district_roads = roads[roads["district"].astype(str).map(_norm_name) == _norm_name(district)]
            for _, rr in district_roads.head(5000).iterrows():
                status = str(rr.get("routing_status", "AVAILABLE")).upper()
                color = route_colors.get(status, "gray")
                folium.GeoJson(rr.geometry.__geo_interface__,
                    style_function=lambda feature, c=color: {"color":c,"weight":2,"opacity":.65},
                    tooltip=f"{rr.get('road_name','Unnamed OSM road')} | {status} | LSI {float(rr.get('LSI_mean',0)):.4f}").add_to(rm)
            folium.LayerControl().add_to(rm)
            st_folium(rm, use_container_width=True, height=480, key="route_map_v6")

        safe, alt, danger = options[0], options[1] if len(options)>1 else options[0], options[-1]
        def card_data(row):
            risk=float(row.get("display_risk",0)); s=float(row.get("LSI_mean",.5))
            hv=row.get("historical_event_indicator",0)
            hist=int(hv) if pd.notna(hv) else 0
            lat_r,lon_r=road_coordinates(row)
            return risk,s,hist,lat_r,lon_r
        safe_risk,safe_s,safe_hist,safe_lat,safe_lon=card_data(safe)
        alt_risk,alt_s,alt_hist,alt_lat,alt_lon=card_data(alt)
        danger_risk,danger_s,danger_hist,danger_lat,danger_lon=card_data(danger)
        danger_status=str(danger.get("routing_status","AVAILABLE")).upper()
        traffic="HEAVY" if danger_status in {"BLOCKED","AVOID_IF_ALTERNATIVE"} else "NORMAL"
        danger_coords="—" if danger_lat is None else f"{danger_lat:.5f}, {danger_lon:.5f}"
        safe_coords="—" if safe_lat is None else f"{safe_lat:.5f}, {safe_lon:.5f}"
        alt_coords="—" if alt_lat is None else f"{alt_lat:.5f}, {alt_lon:.5f}"

        st.subheader("⚠️ PRIMARY ROUTE HAZARD")
        st.markdown(f"""<div class="route-card danger"><div class="route-title">🔴 PRIMARY ROUTE HAZARD</div><div class="route-sub">HAZARD CORRIDOR</div><div class="route-info"><b>Road:</b> {danger.get('road_label','Unnamed OSM road')}<br><b>Active status:</b> {danger_status}<br><b>Traffic:</b> {traffic}<br><b>Historical evidence:</b> {danger_hist} event(s)<br><b>Latitude, Longitude:</b> {danger_coords}<br><b>RF susceptibility:</b> {danger_s:.4f}<br><b>Dynamic trigger:</b> {trig:.4f}<br><b>Risk score:</b> {danger_risk:.4f}</div><div class="route-action">DO NOT TAKE ROUTE A</div></div>""", unsafe_allow_html=True)

        st.subheader("🟢 TAKE ROUTE B")
        st.markdown(f"""<div class="route-card safe"><div class="route-title">🟢 TAKE ROUTE B</div><div class="route-sub">RECOMMENDED SAFE ALTERNATIVE</div><div class="route-info"><b>State:</b> {st_name}<br><b>District:</b> {district}<br><b>Road:</b> {safe.get('road_label','Unnamed OSM road')}<br><b>Latitude, Longitude:</b> {safe_coords}<br><b>Distance:</b> {float(safe.get('length_km',0) or 0):.2f} km<br><b>ETA:</b> {float(safe.get('base_travel_time_min',0) or 0):.1f} min<br><b>RF susceptibility:</b> {safe_s:.4f}<br><b>Dynamic trigger:</b> {trig:.4f}<br><b>Risk score:</b> {safe_risk:.4f}<br><b>Historical events:</b> {safe_hist}<br><b>Historical penalty:</b> {float(safe.get('historical_penalty',1.0) or 1.0):.2f}<br><b>Dynamic penalty:</b> {float(safe.get('dynamic_penalty',1.0) or 1.0):.2f}<br><b>Route cost:</b> {float(safe.get('routing_cost_min',0) or 0):.2f}</div><div class="route-action">✓ TAKE THIS ROAD</div></div>""", unsafe_allow_html=True)

        st.subheader("🔴 DO NOT TAKE ROUTE A")
        st.markdown(f"""<div class="route-card danger"><div class="route-title">🔴 DO NOT TAKE ROUTE A</div><div class="route-sub">HAZARD CORRIDOR</div><div class="route-info"><b>Road:</b> {danger.get('road_label','Unnamed OSM road')}<br><b>Distance:</b> {float(danger.get('length_km',0) or 0):.2f} km<br><b>ETA:</b> {float(danger.get('base_travel_time_min',0) or 0):.1f} min<br><b>RF susceptibility:</b> {danger_s:.4f}<br><b>Dynamic trigger:</b> {trig:.4f}<br><b>Risk score:</b> {danger_risk:.4f}<br><b>Historical events:</b> {danger_hist}<br><b>Historical penalty:</b> {float(danger.get('historical_penalty',1.0) or 1.0):.2f}<br><b>Traffic:</b> {traffic}<br><b>Active incident:</b> {danger_status}</div><div class="route-action">DO NOT TAKE ROUTE A</div></div>""", unsafe_allow_html=True)
    else:
        st.warning(f"No road records were found for {district}. Check the latest pilot_roads_with_cost.gpkg in pilot_route_data.")

# 14. MODEL VALIDATION
# ============================================================

st.markdown("---")
st.header("🎯 Model Validation")
v1, v2, v3, v4 = st.columns(4)
v1.metric("Independent ROC-AUC", "0.8355")
v2.metric("PR-AUC", "0.7535")
v3.metric("Balanced Accuracy", "0.7829")
v4.metric("Training Samples", "991")
st.caption("Independent evaluation on 74 unseen Meghalaya locations.")

st.header("🧠 What each layer does")
st.markdown("""
**STATIC LAYER**
- 8 real raster predictors.
- Production Random Forest produces a 0–1 susceptibility score.
- 250 m model grid.

**DYNAMIC LAYER**
- Current/recent rainfall and forecast weather at the searched coordinate.
- Computes the rainfall trigger independently of the trained RF.

**RISK DECISION**
- Static susceptibility is combined with the live rainfall trigger for an operational risk indicator.
- LOW / MEDIUM / HIGH / CRITICAL operational status.

**SATELLITE**
- Real satellite imagery basemap centered on the searched location.
- Satellite disturbance can be corroborating evidence, not proof of a landslide.

**DATA INTEGRITY**
- No synthetic susceptibility or weather values are inserted.
- Mizoram remains unavailable where production predictor coverage is incomplete.
""")

st.success(
    "Giri-Rakshak is search-first: the state map and local satellite view appear only after a location is searched."
)
