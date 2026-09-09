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
        "past_days": 3,
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
    rows = []
    for label, (lat, lon) in MONITORING.items():
        s = get_real_sus(lat, lon)
        try:
            w = weather(lat, lon)
            r24, r72 = rain_history(w)
            trig, ratio24, ratio72 = dynamic_trigger(r24, r72)
            if s is not None:
                fusion = fuse_risk(s, trig)
                r = fusion.risk_score
                level = fusion.risk_level
            else:
                r = None
                level = "UNAVAILABLE"
            rows.append({
                "Location": label,
                "S": None if s is None else round(s, 4),
                "T": round(trig, 4),
                "Final R": None if r is None else round(r, 4),
                "Risk": level,
                "24h rain (mm)": round(r24, 1),
                "72h rain (mm)": round(r72, 1),
            })
        except Exception:
            rows.append({
                "Location": label,
                "S": None if s is None else round(s, 4),
                "T": None,
                "Final R": None,
                "Risk": "WEATHER UNAVAILABLE",
                "24h rain (mm)": None,
                "72h rain (mm)": None,
            })
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


def route_options_for_area(roads, district, trigger):
    if roads is None or roads.empty:
        return []
    r = roads[roads["district"].astype(str).str.strip().str.lower() == district.lower()].copy()
    if r.empty:
        return []
    for col in ["LSI_mean", "length_km", "base_travel_time_min", "routing_cost_min"]:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")
    r["display_risk"] = np.clip(0.6 * r["LSI_mean"].fillna(0.5) + 0.4 * float(trigger), 0, 1)
    r["display_status"] = r["routing_status"].astype(str) if "routing_status" in r.columns else "AVAILABLE"
    r["road_label"] = r["road_name"].fillna("Unnamed road").astype(str) if "road_name" in r.columns else "Unnamed road"
    r["road_label"] = r["road_label"].replace("", "Unnamed road")
    r = r.sort_values(["routing_cost_min", "display_risk"], na_position="last")
    chosen, seen = [], set()
    for _, row in r.iterrows():
        name = row["road_label"]
        key = name.lower()
        if key in seen and key != "unnamed road":
            continue
        chosen.append(row)
        seen.add(key)
        if len(chosen) >= 3:
            break
    if len(chosen) >= 2 and "display_risk" in r.columns:
        high = r.sort_values("display_risk", ascending=False).iloc[0]
        if float(high["display_risk"]) > float(chosen[-1]["display_risk"]) + 0.05:
            chosen[-1] = high
    return chosen


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
    st.subheader(f"🗺️ {state} — RF Susceptibility")

    state_map_path = STATE_MAP_DIR / (
        state.replace(" ", "_") + "_susceptibility_map.png"
    )

    if state == "Mizoram":
        st.warning(
            "RF susceptibility is currently unavailable for Mizoram because the production "
            "predictor stack has incomplete geomorphology coverage. No synthetic value is inserted."
        )

    if state_map_path.exists():
        st.image(state_map_path, use_container_width=True)
        with open(state_map_path, "rb") as f:
            st.download_button(
                "⬇️ Download State Map PNG",
                data=f.read(),
                file_name=state_map_path.name,
                mime="image/png",
                key="download_state_map",
            )
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
# 13. ROUTE AWARENESS — THREE OPTIONS
# ============================================================

st.markdown("---")
st.header("🛣️ Landslide-Aware Route Options")

if gpd is None:
    st.info("Install geopandas to display the route network.")
elif not ROUTE_GPKG.exists():
    st.info("The real pilot road GeoPackage is not available in this local project copy.")
else:
    route_districts = {
        "Kamrup Metropolitan, Assam": ("Kamrup Metropolitan", "Assam"),
        "East Khasi Hills, Meghalaya": ("East Khasi Hills", "Meghalaya"),
    }
    current_area = None
    for label, (district, st_name) in route_districts.items():
        if str(loc.get("admin2", "")).strip().lower() == district.lower() and str(loc.get("admin1", "")).strip().lower() == st_name.lower():
            current_area = label
            break

    route_choice = st.selectbox(
        "Route pilot area",
        list(route_districts.keys()),
        index=list(route_districts.keys()).index(current_area) if current_area else 0,
        key="route_pilot_area",
    )
    district, st_name = route_districts[route_choice]
    roads = load_route_roads()
    options = route_options_for_area(roads, district, trig)

    if options:
        if folium is not None:
            rm = folium.Map(location=STATE_CENTERS[st_name], zoom_start=10, control_scale=True)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery", name="🛰️ Satellite"
            ).add_to(rm)
            route_colors = {"AVAILABLE": "green", "CAUTION": "orange", "AVOID_IF_ALTERNATIVE": "red", "BLOCKED": "black"}
            for _, r in roads[roads["district"].astype(str).str.strip().str.lower() == district.lower()].head(5000).iterrows():
                status = str(r.get("routing_status", "AVAILABLE"))
                color = route_colors.get(status, "gray")
                folium.GeoJson(
                    r.geometry.__geo_interface__,
                    style_function=lambda feature, c=color: {"color": c, "weight": 2, "opacity": .65},
                    tooltip=f"{r.get('road_name','Unnamed road')} | {status} | LSI {float(r.get('LSI_mean',0)):.3f}",
                ).add_to(rm)
            folium.LayerControl().add_to(rm)
            st_folium(rm, use_container_width=True, height=480)

        st.subheader("🚦 Choose the safest feasible option")
        st.caption("These options are generated from the real OSM-based pilot road network and existing road-risk fields. They are road options, not a full source-to-destination shortest-path result.")
        cards = st.columns(3)
        labels = [
            ("🟢", "TAKE THIS ROUTE", "RECOMMENDED SAFE OPTION", "safe"),
            ("🟡", "ALTERNATIVE", "AVAILABLE WITH CAUTION", "caution"),
            ("🔴", "DO NOT PRIORITIZE", "HIGHER-RISK OPTION", "danger"),
        ]
        for idx, (col, row) in enumerate(zip(cards, options[:3])):
            risk = float(row.get("display_risk", 0))
            if risk >= .75 or str(row.get("display_status", "")).upper() in {"BLOCKED", "AVOID_IF_ALTERNATIVE"}:
                label = ("🔴", "DO NOT TAKE", "HAZARD CORRIDOR", "danger")
            elif risk >= .30:
                label = ("🟡", "USE WITH CAUTION", "ALTERNATIVE OPTION", "caution")
            else:
                label = labels[idx]
            status = str(row.get("display_status", "AVAILABLE"))
            distance = float(row.get("length_km", 0) or 0)
            eta = float(row.get("base_travel_time_min", 0) or 0)
            road_name = str(row.get("road_label", "Unnamed road"))
            with col:
                html = f"""<div class=\"route-card {label[3]}\">
<div class=\"route-title\">{label[0]} {label[1]}</div>
<div class=\"route-sub\">{label[2]}</div>
<div class=\"route-info\">
<b>State:</b> {st_name}<br>
<b>District:</b> {district}<br>
<b>Road:</b> {road_name}<br>
<b>Distance:</b> {distance:.2f} km<br>
<b>ETA:</b> {eta:.1f} min<br>
<b>Risk indicator:</b> {risk:.3f}<br>
<b>Road status:</b> {status}
</div>
<div class=\"route-action\">{label[1]}</div>
</div>"""
                st.markdown(html, unsafe_allow_html=True)
    else:
        st.info("No route records were found for this pilot district.")

# ============================================================
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
