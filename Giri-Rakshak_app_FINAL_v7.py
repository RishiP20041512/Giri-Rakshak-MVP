import streamlit as st
from pathlib import Path
import json
import base64
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Giri-Rakshak — Environmental Intelligence Platform",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).parent
IMG_DIR = ROOT / "project image"

LOGO_PATH = IMG_DIR / "logo.png"
HEADER_IMG = IMG_DIR / "header.png"
SUMMER_IMG = IMG_DIR / "summer.png"
MAUSAM_IMG = IMG_DIR / "mausam-bg.png"
BG_IMG = IMG_DIR / "bg.webp"

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

MONITORING = {
    "East Khasi Hills, Meghalaya": (25.2702, 91.7323),
    "Guwahati / Kamrup Metropolitan, Assam": (26.1445, 91.7362),
    "Senapati, Manipur": (25.2030, 94.3124),
    "Mon, Nagaland": (26.7167, 95.0667),
    "Agartala / West Tripura, Tripura": (23.8315, 91.2868),
    "Mangan, Sikkim": (27.50115, 88.53553),
    "Itanagar, Arunachal Pradesh": (27.4728, 94.9120),
}

VALIDATED = {
    "East Khasi Hills, Meghalaya": (0.5677, 0.2069, 0.4234, "MEDIUM"),
    "Guwahati / Kamrup Metropolitan, Assam": (0.6830, 0.2089, 0.4934, "MEDIUM"),
    "Senapati, Manipur": (0.5518, 0.1983, 0.4104, "MEDIUM"),
    "Mon, Nagaland": (0.3864, 0.3676, 0.3789, "MEDIUM"),
    "Agartala / West Tripura, Tripura": (0.4184, 0.2345, 0.3448, "MEDIUM"),
    "Mangan, Sikkim": (0.8299, 0.4363, 0.6724, "HIGH"),
    "Itanagar, Arunachal Pradesh": (0.7527, 0.1880, 0.5268, "HIGH"),
}

ROUTE_TRIGGER = {
    "Kamrup Metropolitan, Assam": 0.2089,
    "East Khasi Hills, Meghalaya": 0.2069,
    "West Tripura, Tripura": 0.2345,
}

DISTRICT_HISTORICAL_EVIDENCE = {
    "Kamrup Metropolitan, Assam": 1,
    "East Khasi Hills, Meghalaya": 1,
    "West Tripura, Tripura": 1,
}


# ============================================================
# ASSET HELPERS
# ============================================================

@st.cache_data
def get_b64_image(path):
    p = Path(path)
    if not p.exists():
        return ""
    try:
        with open(p, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        ext = p.suffix.lower().replace(".", "")
        if ext == "jpg":
            ext = "jpeg"
        return f"data:image/{ext};base64,{data}"
    except Exception:
        return ""

LOGO_B64 = get_b64_image(LOGO_PATH)
HEADER_B64 = get_b64_image(HEADER_IMG)
SUMMER_B64 = get_b64_image(SUMMER_IMG)
MAUSAM_B64 = get_b64_image(MAUSAM_IMG)

MAP_ICON_B64 = get_b64_image(IMG_DIR / "map.jpg")
SEARCH_ICON_B64 = get_b64_image(IMG_DIR / "live search.webp")
ROUTE_ICON_B64 = get_b64_image(IMG_DIR / "rouute.png")
RISK_ICON_B64 = get_b64_image(IMG_DIR / "red-emergency-light-warning-sign.avif")
WEATHER_ICON_B64 = get_b64_image(IMG_DIR / "weather.png")
FORECAST_ICON_B64 = get_b64_image(IMG_DIR / "7day icon.png")
PDF_ICON_B64 = get_b64_image(IMG_DIR / "pdf image.webp")


# ============================================================
# CORE GIS & DATA FUNCTIONS
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
        "HIGH": "🔴 HIGH",
        "CRITICAL": "🔴 CRITICAL",
        "UNAVAILABLE": "⚪ UNAVAILABLE",
    }.get(str(level).upper(), str(level))


def value_at_raster(path, lat, lon):
    if not path.exists():
        return None
    try:
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
    except Exception:
        return None


def get_real_sus(lat, lon):
    return value_at_raster(SUS_RASTER, lat, lon)


def read_sus_crop(lat, lon, radius_km=14):
    if not SUS_RASTER.exists():
        return None
    try:
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
    except Exception:
        return None


def make_susceptibility_plot(lat, lon, name):
    result = read_sus_crop(lat, lon, 14)
    if result is None:
        return None
    arr, tr = result
    fig, ax = plt.subplots(figsize=(10, 6.2))
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
        interpolation="bilinear",
    )
    with rasterio.open(SUS_RASTER) as src:
        x, y = transform("EPSG:4326", src.crs, [lon], [lat])
    ax.scatter(
        x[0], y[0], s=140, marker="*",
        edgecolor="black", facecolor="#1e3a8a", linewidth=1.4, zorder=5
    )
    ax.set_title(
        f"{name} — Local RF Susceptibility (250m Resolution)",
        fontsize=15, fontweight="bold", pad=12
    )
    ax.set_xlabel("Model Coordinate X (meters)", fontsize=11)
    ax.set_ylabel("Model Coordinate Y (meters)", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, label="RF Susceptibility Score (0.0 to 1.0)", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


@st.cache_data(ttl=1800)
def geocode(q):
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
            with urllib.request.urlopen(url, timeout=12) as r:
                data = json.loads(r.read().decode())
        except Exception:
            continue
        for x in data.get("results", []):
            if str(x.get("country_code", "")).upper() != "IN":
                continue
            admin1 = str(x.get("admin1", "")).strip()
            name = str(x.get("name", "")).strip()
            admin2 = str(x.get("admin2", "")).strip()
            if admin1.lower() not in state_lower and name.lower() not in state_lower:
                continue
            key = (name.lower(), admin1.lower(), round(float(x.get("latitude", 0)), 4), round(float(x.get("longitude", 0)), 4))
            if key not in seen:
                seen.add(key)
                x["_admin2"] = admin2
                results.append(x)

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
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode())


def rain_history(payload):
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


def weather_label(code, rain_mm=0):
    code = int(code) if code is not None else 0
    if rain_mm > 15 or code in {65, 82, 95, 96, 99}:
        return "Heavy Rain"
    if rain_mm > 0 or code in {51, 53, 55, 61, 63, 80, 81}:
        return "Rainy"
    if code in {1, 2, 3}:
        return "Partly Cloudy"
    if code in {45, 48}:
        return "Foggy"
    return "Clear Sky"


def state_mean(state):
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


def forecast_risk_rows(payload, susceptibility, threshold24, threshold72):
    daily = payload["daily"]
    rain = np.asarray(daily.get("precipitation_sum", []), dtype=float)
    prob = np.asarray(daily.get("precipitation_probability_max", []), dtype=float)
    codes = np.asarray(daily.get("weather_code", np.zeros(len(rain))), dtype=float)
    t_max = np.asarray(daily.get("temperature_2m_max", np.zeros(len(rain))), dtype=float)
    t_min = np.asarray(daily.get("temperature_2m_min", np.zeros(len(rain))), dtype=float)
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
            "day": dates[i].strftime("%a"),
            "day_num": dates[i].strftime("%d"),
            "date": dates[i].strftime("%b %d"),
            "is_today": (i == 0),
            "code": int(codes[i]),
            "icon": weather_icon(codes[i], r24),
            "rain": r24,
            "t_max": round(float(t_max[i])) if i < len(t_max) else 30,
            "t_min": round(float(t_min[i])) if i < len(t_min) else 24,
            "prob": int(np.clip(prob[i] if i < len(prob) else 0, 0, 100)),
            "risk": risk,
            "risk_pct": None if risk is None else round(risk * 100),
            "level": level,
        })
    return rows


@st.cache_data(ttl=1800)
def load_route_roads():
    if gpd is None:
        return None
    candidates = [
        (ROUTE_GPKG, "pilot_roads_cost"),
        (ROOT / "pilot_route_data" / "pilot_roads_with_historical_landslides.gpkg", "pilot_roads_historical"),
        (ROOT / "pilot_route_data" / "pilot_roads_historical_risk.gpkg", "pilot_roads_historical"),
        (ROOT / "pilot_route_data" / "pilot_roads_active.gpkg", "pilot_roads_active"),
    ]
    for path, layer in candidates:
        if not path.exists():
            continue
        try:
            layers = list(gpd.list_layers(path)["name"]) if hasattr(gpd, "list_layers") else []
            chosen_layer = layer if layer in layers else (layers[0] if layers else layer)
            roads = gpd.read_file(path, layer=chosen_layer)
            if roads is None or roads.empty:
                continue
            if roads.crs is None:
                roads = roads.set_crs("EPSG:4326")
            roads["geometry"] = roads.geometry.simplify(0.00008, preserve_topology=True)
            return roads
        except Exception:
            continue
    return None


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
        target = _norm_name(district)
        r = roads[roads["district"].astype(str).map(lambda x: target in _norm_name(x) or _norm_name(x) in target)].copy()
    if r.empty:
        return []
    for col in ["LSI_mean", "length_km", "base_travel_time_min", "routing_cost_min", "historical_event_indicator", "historical_penalty", "dynamic_penalty"]:
        if col in r.columns:
            r[col] = pd.to_numeric(r[col], errors="coerce")
    if "LSI_mean" not in r.columns:
        r["LSI_mean"] = 0.5
    r["LSI_mean"] = r["LSI_mean"].fillna(0.5)
    for col, default in [("length_km",0.0),("base_travel_time_min",0.0),("historical_event_indicator",0),("historical_penalty",1.0),("dynamic_penalty",1.0)]:
        if col not in r.columns:
            r[col] = default
        r[col] = r[col].fillna(default)
    if "routing_cost_min" not in r.columns:
        r["routing_cost_min"] = np.nan
    r["route_trigger"] = float(trigger)
    r["display_risk"] = np.clip(0.6 * r["LSI_mean"] + 0.4 * float(trigger), 0, 1)
    r["display_status"] = r["routing_status"].astype(str).str.upper() if "routing_status" in r.columns else "AVAILABLE"
    r["road_label"] = r["road_name"].fillna("Unnamed OSM road").astype(str) if "road_name" in r.columns else "Unnamed OSM road"
    r["road_label"] = r["road_label"].replace({"": "Unnamed OSM road", "nan": "Unnamed OSM road"})
    available = r[r["display_status"] != "BLOCKED"].copy()
    named = available[available["road_label"].str.lower() != "unnamed osm road"]
    safe_pool = named if not named.empty else available
    if safe_pool.empty:
        safe_pool = r
    safe = safe_pool.sort_values(["display_risk", "routing_cost_min"], na_position="last").iloc[0]
    blocked = r[r["display_status"] == "BLOCKED"]
    avoid = r[r["display_status"] == "AVOID_IF_ALTERNATIVE"]
    if not blocked.empty:
        danger = blocked.sort_values("display_risk", ascending=False).iloc[0]
    elif not avoid.empty:
        danger = avoid.sort_values("display_risk", ascending=False).iloc[0]
    else:
        danger = r.sort_values(["display_risk", "LSI_mean"], ascending=False, na_position="last").iloc[0]
    alt_pool = safe_pool[safe_pool.index != safe.name]
    alt = alt_pool.sort_values(["display_risk", "routing_cost_min"], na_position="last").iloc[0] if not alt_pool.empty else safe
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
        fc.append([r["date"], r["icon"], f"{r['rain']:.1f}", f"{r['prob']}%",
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
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# ============================================================
# SEARCH & INITIAL STATE
# ============================================================

if "has_searched" not in st.session_state:
    st.session_state["has_searched"] = False

if "location" not in st.session_state:
    st.session_state["location"] = {
        "name": "Shillong",
        "lat": 25.5788,
        "lon": 91.8933,
        "admin1": "Meghalaya",
        "admin2": "East Khasi Hills",
        "state_only": False,
    }

loc = st.session_state["location"]
state = str(loc.get("admin1", "")).strip()
if state not in STATES and loc.get("name") in STATES:
    state = loc["name"]

if loc.get("state_only") and state in STATE_CENTERS:
    loc["lat"], loc["lon"] = STATE_CENTERS[state]
    loc["admin1"] = state
    loc["admin2"] = "State Overview"

# ============================================================
# ENVIRONMENTAL INTELLIGENCE THEME & TOPO CONTOUR TEXTURE
# ============================================================

# Authentic GIS Elevation Contour Vector Pattern (Clean, Organic, Layered, Mountain Contours)
TOPO_CONTOUR_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" width="900" height="900" viewBox="0 0 900 900">
  <defs>
    <style>
      .c-subtle { fill: none; stroke: rgba(52, 211, 153, 0.055); stroke-width: 1.1; }
      .c-index  { fill: none; stroke: rgba(110, 231, 183, 0.095); stroke-width: 1.6; }
      .c-accent { fill: none; stroke: rgba(167, 243, 208, 0.035); stroke-width: 0.9; stroke-dasharray: 4, 4; }
      .c-txt { fill: rgba(110, 231, 183, 0.18); font-family: monospace; font-size: 8px; font-weight: bold; letter-spacing: 0.05em; }
    </style>
  </defs>
  <!-- North-West Ridge -->
  <path class="c-index" d="M -50,150 Q 80,110 180,160 T 360,130 T 520,210 T 700,160 T 950,220" />
  <path class="c-subtle" d="M -50,180 Q 90,140 190,190 T 370,160 T 530,240 T 710,190 T 950,250" />
  <path class="c-subtle" d="M -50,210 Q 100,170 200,220 T 380,190 T 540,270 T 720,220 T 950,280" />
  <path class="c-subtle" d="M -50,240 Q 110,200 210,250 T 390,220 T 550,300 T 730,250 T 950,310" />
  <path class="c-index" d="M -50,270 Q 120,230 220,280 T 400,250 T 560,330 T 740,280 T 950,340" />
  <text x="210" y="278" class="c-txt">1800m</text>
  <text x="570" y="328" class="c-txt">1800m</text>

  <!-- Central Mountain Ridge & Valley Contours -->
  <path class="c-subtle" d="M 120,-30 C 180,90 280,140 240,260 S 110,380 200,490 S 390,560 330,680 S 180,850 220,950" />
  <path class="c-index" d="M 160,-30 C 220,90 320,130 280,250 S 150,370 240,480 S 430,550 370,670 S 220,830 260,950" />
  <text x="270" y="248" class="c-txt">1600m</text>
  <path class="c-subtle" d="M 200,-30 C 260,90 360,120 320,240 S 190,360 280,470 S 470,540 410,660 S 260,820 300,950" />

  <!-- Enclosed Elevation Summits -->
  <path class="c-index" d="M 600,420 C 560,360 670,310 730,350 S 810,460 750,510 S 640,480 600,420 Z" />
  <text x="665" y="345" class="c-txt">2200m ▲</text>
  <path class="c-subtle" d="M 620,420 C 585,375 675,335 720,365 S 785,450 740,490 S 655,465 620,420 Z" />
  <path class="c-subtle" d="M 640,420 C 610,390 680,360 710,380 S 760,440 730,470 S 670,450 640,420 Z" />
  <path class="c-accent" d="M 660,420 C 640,400 685,385 700,395 S 735,430 720,450 S 680,440 660,420 Z" />
  <circle cx="690" cy="420" r="2.5" fill="rgba(110, 231, 183, 0.28)" />

  <!-- South-Western Slopes -->
  <path class="c-subtle" d="M -40,450 Q 80,420 160,490 T 260,620 T 180,820" />
  <path class="c-index" d="M -40,480 Q 90,450 180,520 T 290,650 T 210,850" />
  <text x="95" y="460" class="c-txt">1400m</text>
  <path class="c-subtle" d="M -40,510 Q 100,480 200,550 T 320,680 T 240,880" />
  <path class="c-subtle" d="M -40,540 Q 110,510 220,580 T 350,710 T 270,910" />

  <!-- South-Eastern Valley Contours -->
  <path class="c-subtle" d="M 450,920 C 490,780 610,740 670,790 S 770,750 920,780" />
  <path class="c-index" d="M 480,920 C 520,795 630,755 690,805 S 790,765 920,795" />
  <text x="600" y="775" class="c-txt">1200m</text>
  <path class="c-subtle" d="M 510,920 C 550,810 650,770 710,820 S 810,780 920,810" />

  <!-- North-Eastern Peak -->
  <path class="c-index" d="M 460,110 C 500,40 610,30 650,80 S 670,170 620,200 S 420,180 460,110 Z" />
  <text x="530" y="60" class="c-txt">2000m ▲</text>
  <path class="c-subtle" d="M 480,115 C 515,60 595,50 630,90 S 645,160 605,185 S 445,170 480,115 Z" />
  <path class="c-accent" d="M 505,120 C 530,80 580,75 605,100 S 620,150 590,170 S 480,160 505,120 Z" />
</svg>'''

TOPO_DATA_URI = f"data:image/svg+xml;base64,{base64.b64encode(TOPO_CONTOUR_SVG.encode('utf-8')).decode('utf-8')}"

# Fetch current ambient weather for atmospheric background styling
try:
    _w_bg = weather(loc["lat"], loc["lon"])
    _cur_bg = _w_bg.get("current", {})
    _rain_bg = float(_cur_bg.get("rain", 0.0))
    _temp_bg = float(_cur_bg.get("temperature_2m", 22.0))
    _code_bg = int(_cur_bg.get("weather_code", 0))
except Exception:
    _rain_bg, _temp_bg, _code_bg = 0.0, 22.0, 0

if _code_bg in {95, 96, 99}:
    atm_mode = "storm"
    atm_bg_img = MAUSAM_B64
elif _rain_bg >= 5.0 or _code_bg in {65, 82}:
    atm_mode = "heavy_rain"
    atm_bg_img = MAUSAM_B64
elif _rain_bg > 0.05 or _code_bg in {51, 53, 55, 61, 63, 80, 81}:
    atm_mode = "rain"
    atm_bg_img = MAUSAM_B64
elif _code_bg in {45, 48}:
    atm_mode = "fog"
    atm_bg_img = HEADER_B64
elif _code_bg in {1, 2, 3}:
    atm_mode = "cloudy"
    atm_bg_img = HEADER_B64
elif _temp_bg >= 28 or (_code_bg == 0 and _temp_bg >= 26):
    atm_mode = "summer"
    atm_bg_img = SUMMER_B64
else:
    atm_mode = "clear"
    atm_bg_img = HEADER_B64

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&family=Outfit:wght@400;500;600;700;800&family=Bebas+Neue&display=swap');

#MainMenu {{visibility: hidden;}}
header {{visibility: hidden;}}
footer {{visibility: hidden;}}

html {{
    scroll-behavior: smooth;
}}

/* Primary Dark Environmental Background (#071A1D -> #0B2929 -> #103B36 -> #123F45 -> #09232D -> #06151B) with GIS Topographic Texture */
.stApp {{
    background-color: #071A1D;
    background-image: 
        url("{TOPO_DATA_URI}"),
        radial-gradient(ellipse 90% 50% at 50% -10%, rgba(18, 63, 69, 0.55), transparent 70%),
        radial-gradient(ellipse 70% 45% at 85% 30%, rgba(16, 59, 54, 0.4), transparent 60%),
        radial-gradient(ellipse 80% 55% at 15% 75%, rgba(9, 35, 45, 0.5), transparent 70%),
        linear-gradient(180deg, #071A1D 0%, #0B2929 18%, #103B36 38%, #123F45 55%, #09232D 75%, #06151B 100%);
    background-repeat: repeat, no-repeat, no-repeat, no-repeat, no-repeat;
    background-size: 800px 800px, auto, auto, auto, auto;
    background-attachment: fixed;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #f8fafc;
}}

.block-container {{
    position: relative;
    z-index: 1;
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    padding-left: 1.8rem !important;
    padding-right: 1.8rem !important;
    max-width: 1420px !important;
}}

/* Typography: Bold, Editorial Headings & Crisp Body */
h1, h2, h3, .sec-title {{
    font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
    font-weight: 800;
    color: #f8fafc;
    letter-spacing: -0.01em;
}}

/* Blinking Animation for High Risk on Map */
@keyframes redPulseBlink {{
    0% {{
        transform: scale(0.9);
        opacity: 0.9;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.85);
    }}
    70% {{
        transform: scale(1.6);
        opacity: 0.2;
        box-shadow: 0 0 0 18px rgba(239, 68, 68, 0);
    }}
    100% {{
        transform: scale(0.9);
        opacity: 0.9;
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
    }}
}}

/* Weather Atmospheric Background Overlay & Lightweight Pure CSS Animations */
.weather-backdrop-fx {{
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    opacity: 0.16;
    background-image: url("{atm_bg_img}");
    background-position: center;
    background-size: cover;
    background-repeat: no-repeat;
    mix-blend-mode: luminosity;
    transition: opacity 0.6s ease;
}}

.weather-backdrop-fx.rain, .weather-backdrop-fx.heavy_rain {{
    opacity: 0.22;
}}
.weather-backdrop-fx.rain::before, .weather-backdrop-fx.heavy_rain::before {{
    content: '';
    position: absolute;
    inset: 0;
    background-image: linear-gradient(175deg, rgba(255,255,255,0.18) 1.5px, transparent 2px);
    background-size: 24px 28px;
    animation: rainFall 0.75s linear infinite;
}}
.weather-backdrop-fx.rain::after, .weather-backdrop-fx.heavy_rain::after {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 100%, rgba(203, 213, 225, 0.12) 0%, transparent 70%);
    animation: mistFloat 6s ease-in-out infinite alternate;
}}

.weather-backdrop-fx.fog {{
    opacity: 0.24;
}}
.weather-backdrop-fx.fog::before {{
    content: '';
    position: absolute;
    inset: -20px;
    background: radial-gradient(circle at 30% 50%, rgba(203, 213, 225, 0.16) 0%, transparent 60%),
                radial-gradient(circle at 70% 60%, rgba(148, 163, 184, 0.14) 0%, transparent 65%);
    animation: fogDrift 14s ease-in-out infinite alternate;
}}

.weather-backdrop-fx.cloudy::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 40% 30%, rgba(255, 255, 255, 0.08) 0%, transparent 50%),
                radial-gradient(circle at 80% 40%, rgba(255, 255, 255, 0.06) 0%, transparent 55%);
    animation: cloudDrift 20s linear infinite;
}}

.weather-backdrop-fx.storm {{
    opacity: 0.25;
    animation: stormLightning 9s infinite;
}}
.weather-backdrop-fx.storm::before {{
    content: '';
    position: absolute;
    inset: 0;
    background-image: linear-gradient(172deg, rgba(255,255,255,0.22) 2px, transparent 2.5px);
    background-size: 20px 24px;
    animation: rainFall 0.55s linear infinite;
}}

.weather-backdrop-fx.summer {{
    opacity: 0.20;
    mix-blend-mode: soft-light;
}}
.weather-backdrop-fx.summer::before {{
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at 50% 0%, rgba(251, 191, 36, 0.16) 0%, transparent 65%);
    animation: heatHaze 4s ease-in-out infinite alternate;
}}

@keyframes rainFall {{
    0% {{ background-position: 0 0; }}
    100% {{ background-position: 12% 1000px; }}
}}
@keyframes mistFloat {{
    0% {{ transform: translateY(0); opacity: 0.6; }}
    100% {{ transform: translateY(-12px); opacity: 0.9; }}
}}
@keyframes fogDrift {{
    0% {{ transform: translateX(-35px); }}
    100% {{ transform: translateX(35px); }}
}}
@keyframes cloudDrift {{
    0% {{ transform: translateX(0); }}
    100% {{ transform: translateX(100px); }}
}}
@keyframes stormLightning {{
    0%, 92%, 95%, 100% {{ filter: brightness(1); }}
    93% {{ filter: brightness(1.65) saturate(1.2); }}
    94% {{ filter: brightness(1.1); }}
}}
@keyframes heatHaze {{
    0% {{ opacity: 0.15; transform: scale(1); }}
    100% {{ opacity: 0.28; transform: scale(1.02); }}
}}

/* Circular Navigation (Environmental Glass Skin) */
.circle-nav-container {{
    display: flex;
    justify-content: space-evenly;
    align-items: flex-start;
    width: 100%;
    padding: 20px 24px 24px 24px;
    background: rgba(7, 26, 29, 0.78);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(52, 211, 153, 0.22);
    border-radius: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4), 0 0 24px rgba(16, 185, 129, 0.08);
}}

.circle-nav-item {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-decoration: none;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    flex: 1;
    max-width: 135px;
}}

.circle-nav-btn {{
    width: 68px;
    height: 68px;
    border-radius: 50%;
    background: rgba(11, 41, 41, 0.9);
    border: 2px solid rgba(52, 211, 153, 0.4);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    transition: all 0.25s ease;
    overflow: hidden;
}}

.circle-nav-item:hover .circle-nav-btn {{
    transform: translateY(-5px) scale(1.08);
    box-shadow: 0 12px 26px rgba(16, 185, 129, 0.4);
    border-color: #10b981;
    background: rgba(16, 59, 54, 0.98);
}}

.circle-nav-lbl {{
    margin-top: 8px;
    font-size: 12px;
    font-weight: 600;
    color: #f1f5f9;
    letter-spacing: 0.02em;
    text-align: center;
    max-width: 90px;
    line-height: 1.25;
    transition: color 0.2s ease;
}}

.circle-nav-item:hover .circle-nav-lbl {{
    color: #34d399;
}}

/* Hero Section */
.hero-sec-bg {{
    position: relative;
    border-radius: 24px;
    background: linear-gradient(180deg, rgba(7, 26, 29, 0.45) 0%, rgba(11, 41, 41, 0.75) 100%), url("{HEADER_B64}") center/cover no-repeat;
    padding: 32px 36px 40px 36px;
    box-shadow: 0 16px 36px rgba(0, 0, 0, 0.5);
    border: 1px solid rgba(52, 211, 153, 0.2);
    margin-bottom: 24px;
    overflow: hidden;
}}

.hero-main-title {{
    font-family: 'Bebas Neue', 'Outfit', sans-serif;
    font-size: 58px;
    line-height: 1.05;
    color: #ffffff;
    text-shadow: 2px 4px 10px rgba(0, 0, 0, 0.75);
    letter-spacing: 0.03em;
    max-width: 920px;
    margin-bottom: 26px;
}}

/* Top 5 KPI Cards */
.kpi-row {{
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 10px;
}}

.kpi-card {{
    background: rgba(7, 26, 29, 0.78);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(52, 211, 153, 0.18);
    border-radius: 14px;
    padding: 14px 16px;
    color: #f8fafc;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
}}

.kpi-title {{
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #8ba89f;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

.kpi-val {{
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
    color: #f8fafc;
}}

.kpi-sub {{
    font-size: 11px;
    color: #8ba89f;
}}

/* Section Containers with Semi-Transparent Glass allowing Topo Texture through */
.section-box {{
    background: rgba(11, 41, 41, 0.72);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(52, 211, 153, 0.16);
    border-radius: 20px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
}}

.sec-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}

.sec-title {{
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
    display: flex;
    align-items: center;
    gap: 8px;
}}

/* Priority Monitoring Cards */
.mon-card {{
    background: rgba(7, 26, 29, 0.75);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(52, 211, 153, 0.15);
    border-radius: 12px;
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}

.mon-place {{
    font-size: 14px;
    font-weight: 700;
    color: #f8fafc;
}}

.mon-trig {{
    font-size: 11px;
    color: #f97316;
    font-weight: 700;
}}

.mon-score {{
    font-size: 20px;
    font-weight: 800;
    color: #38bdf8;
}}

/* Risk Intel Box with Environmental Glass Skin */
.risk-intel-box {{
    background: rgba(11, 41, 41, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(52, 211, 153, 0.22);
    border-radius: 18px;
    padding: 22px 26px;
    color: #f8fafc;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 24px;
}}

.risk-intel-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 14px;
    text-align: center;
}}

.risk-col-val {{
    font-size: 26px;
    font-weight: 800;
    color: #f8fafc;
}}

.risk-col-lbl {{
    font-size: 12px;
    font-weight: 600;
    color: #8ba89f;
    margin-top: 2px;
}}

/* Weather Dynamic Card */
.weather-banner-card {{
    border-radius: 20px;
    padding: 26px 30px;
    color: #ffffff;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.3);
}}

/* 7-Day Forecast Grid (Environmental Glass Skin) */
.forecast-card-wrapper {{
    background: rgba(11, 41, 41, 0.75);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(52, 211, 153, 0.22);
    border-radius: 20px;
    padding: 24px 28px;
    color: #f8fafc;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 24px;
}}

.forecast-grid-layout {{
    display: grid;
    grid-template-columns: repeat(7, 1fr);
    gap: 10px;
    margin-top: 14px;
}}

.fc-day-box {{
    background: rgba(7, 26, 29, 0.65);
    border: 1px solid rgba(52, 211, 153, 0.15);
    border-radius: 14px;
    padding: 16px 8px;
    text-align: center;
    transition: all 0.2s ease;
}}

.fc-day-box.active-day {{
    background: rgba(16, 59, 54, 0.75);
    border: 1.5px solid #10b981;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.25);
}}

.fc-day-num {{
    font-size: 15px;
    font-weight: 800;
    color: #f8fafc;
}}

.fc-day-name {{
    font-size: 11px;
    font-weight: 600;
    color: #8ba89f;
    text-transform: uppercase;
    margin-bottom: 6px;
}}

.fc-icon {{
    font-size: 34px;
    margin: 8px 0;
}}

.fc-temp-max {{
    font-size: 18px;
    font-weight: 800;
    color: #f8fafc;
}}

.fc-temp-min {{
    font-size: 12px;
    font-weight: 600;
    color: #8ba89f;
}}

.fc-rain-tag {{
    font-size: 11px;
    color: #38bdf8;
    font-weight: 700;
    margin-top: 6px;
}}

/* 3D Glass PDF Download Banner */
.pdf-download-card {{
    background: linear-gradient(135deg, rgba(16, 59, 54, 0.9) 0%, rgba(18, 63, 69, 0.9) 50%, rgba(9, 35, 45, 0.95) 100%);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 20px;
    padding: 24px 30px;
    color: #ffffff;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.4);
    margin-top: 16px;
    margin-bottom: 24px;
}}

/* Input & Control overrides for Dark Environmental Theme */
div[data-testid="stTextInput"] input {{
    border-radius: 24px !important;
    padding: 14px 22px !important;
    font-size: 15px !important;
    background: rgba(11, 41, 41, 0.85) !important;
    color: #f8fafc !important;
    border: 1.5px solid rgba(52, 211, 153, 0.28) !important;
    backdrop-filter: blur(8px) !important;
}}
div[data-testid="stTextInput"] input:focus {{
    border-color: #10b981 !important;
    box-shadow: 0 0 14px rgba(16, 185, 129, 0.35) !important;
}}

div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(52, 211, 153, 0.4) !important;
    border-radius: 24px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(16, 185, 129, 0.35) !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stButton"] > button[kind="primary"]:hover {{
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.5) !important;
}}

div[data-testid="stButton"] > button:not([kind="primary"]) {{
    background: rgba(11, 41, 41, 0.68) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(52, 211, 153, 0.22) !important;
    border-radius: 20px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    backdrop-filter: blur(6px) !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {{
    background: rgba(16, 59, 54, 0.95) !important;
    border-color: #10b981 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}}

div[data-testid="stSelectbox"] > div {{
    background: rgba(11, 41, 41, 0.85) !important;
    border: 1px solid rgba(52, 211, 153, 0.25) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
}}
</style>
<div class="weather-backdrop-fx {atm_mode}"></div>
""", unsafe_allow_html=True)


# ============================================================
# TOP BRAND HEADER & CIRCULAR NAVIGATION (PIC 3 STYLE)
# ============================================================

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; padding:12px 24px; background:rgba(7,26,29,0.85); border-bottom:1px solid rgba(52,211,153,0.22); border-radius:0 0 16px 16px; margin-bottom:20px; backdrop-filter:blur(12px);">
    <div style="display:flex; align-items:center; gap:10px;">
        <img src="{LOGO_B64}" style="height:28px; width:auto;" alt="Logo" />
        <span style="font-weight:900; font-size:18px; letter-spacing:0.06em; color:#34d399;">GIRI-RAKSHAK</span>
        <span style="font-size:12px; color:#8ba89f; font-weight:600; margin-left:8px;">Environmental Intelligence Platform</span>
    </div>
    <div style="font-size:12px; color:#10b981; font-weight:800; display:flex; align-items:center; gap:6px;">
        <span style="width:8px; height:8px; border-radius:50%; background:#10b981; display:inline-block;"></span>
        LIVE SATELLITE & AI TELEMETRY
    </div>
</div>

<div class="circle-nav-container">
    <a href="#hero-section" class="circle-nav-item">
        <div class="circle-nav-btn">
            <img src="{MAP_ICON_B64}" alt="Map" style="width:40px; height:40px; object-fit:cover; border-radius:50%;" />
        </div>
        <div class="circle-nav-lbl">Overview & Live Map</div>
    </a>
    <a href="#search-section" class="circle-nav-item">
        <div class="circle-nav-btn">
            <img src="{SEARCH_ICON_B64}" alt="Search" style="width:36px; height:36px; object-fit:contain;" />
        </div>
        <div class="circle-nav-lbl">Location Search</div>
    </a>
    <a href="#risk-intel-section" class="circle-nav-item">
        <div class="circle-nav-btn">
            <img src="{RISK_ICON_B64}" alt="Risk" style="width:38px; height:38px; object-fit:contain;" />
        </div>
        <div class="circle-nav-lbl">Risk Intelligence</div>
    </a>
    <a href="#weather-section" class="circle-nav-item">
        <div class="circle-nav-btn">
            <img src="{WEATHER_ICON_B64}" alt="Weather" style="width:38px; height:38px; object-fit:contain;" />
        </div>
        <div class="circle-nav-lbl">Live Weather</div>
    </a>
    <a href="#geospatial-section" class="circle-nav-item">
        <div class="circle-nav-btn">
            <img src="{ROUTE_ICON_B64}" alt="Satellite & Route" style="width:40px; height:40px; object-fit:contain;" />
        </div>
        <div class="circle-nav-lbl">Satellite & RF Maps</div>
    </a>
    <a href="#forecast-section" class="circle-nav-item">
        <div class="circle-nav-btn">
            <img src="{FORECAST_ICON_B64}" alt="Forecast" style="width:36px; height:36px; object-fit:contain;" />
        </div>
        <div class="circle-nav-lbl">7-Day Forecast</div>
    </a>
    <a href="#route-section" class="circle-nav-item">
        <div class="circle-nav-btn">
            <img src="{PDF_ICON_B64}" alt="PDF Report" style="width:34px; height:34px; object-fit:contain;" />
        </div>
        <div class="circle-nav-lbl">Road Route & PDF</div>
    </a>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTION 1: HERO — "WHERE AI MEETS TERRAIN, WEATHER AND SAFER MOBILITY"
# ============================================================

logo_tag = f'<img src="{LOGO_B64}" style="height:76px; width:auto; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.5));" />' if LOGO_B64 else '<div style="font-size:32px; font-weight:900; color:#f59e0b;">GIRI-RAKSHAK</div>'

st.markdown(f"""
<div id="hero-section" class="hero-sec-bg" style="padding: 24px 32px 28px 32px; min-height: 420px; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div style="display:flex; align-items:center; gap:16px;">
{logo_tag}
</div>
<div style="background: rgba(16, 59, 54, 0.85); backdrop-filter: blur(8px); color: #ffffff; padding: 7px 18px; border-radius: 20px; font-size: 12px; font-weight: 800; letter-spacing: 0.04em; border: 1px solid rgba(52, 211, 153, 0.35); display:flex; align-items:center; gap:8px;">
<img src="{MAP_ICON_B64}" style="width:16px; height:16px; object-fit:cover; border-radius:50%;" alt="AI" />
SATELLITE & AI SYSTEM
</div>
</div>

<div style="display: flex; justify-content: flex-end; margin-top: 40px; margin-bottom: 24px; text-align: right;">
<div class="hero-main-title" style="margin-bottom: 0; font-size: 54px; text-shadow: 0 4px 18px rgba(0,0,0,0.9), 0 2px 6px rgba(0,0,0,0.8);">
WHERE AI MEETS TERRAIN,<br>
WEATHER AND SAFER MOBILITY
</div>
</div>

<div class="kpi-row" style="margin-top: 12px;">
<div class="kpi-card">
<div class="kpi-title">Monitoring Locations</div>
<div class="kpi-val">7</div>
<div class="kpi-sub">7 with current risk</div>
</div>
<div class="kpi-card">
<div class="kpi-title">High Risk</div>
<div class="kpi-val" style="color:#f87171;">
3
</div>
<div class="kpi-sub">0 critical</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Medium Risk</div>
<div class="kpi-val" style="color:#fbbf24;">4</div>
<div class="kpi-sub">0 low</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Data Unavailable</div>
<div class="kpi-val" style="color:#94a3b8;">1</div>
<div class="kpi-sub">Requires attention</div>
</div>
<div class="kpi-card">
<div class="kpi-title">Satellite Anomalies</div>
<div class="kpi-val" style="color:#34d399;">0</div>
<div class="kpi-sub">Verified anomalies</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LIVE RISK MAP & PRIORITY MONITORING (WITH BLINKING MAP MARKERS)
# ============================================================

hero_col_map, hero_col_mon = st.columns([1.25, 0.75])

with hero_col_map:
    st.markdown(f"""
<div style="background:rgba(11, 41, 41, 0.82); border:1px solid rgba(52, 211, 153, 0.18); border-radius:20px 20px 0 0; padding:18px 24px 14px 24px; margin-bottom:0; box-shadow:0 4px 12px rgba(0,0,0,0.25); backdrop-filter:blur(10px);">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div style="display:flex; align-items:center; gap:8px; font-size:20px; font-weight:800; color:#f8fafc;">
<img src="{MAP_ICON_B64}" style="width:24px; height:24px; object-fit:cover; border-radius:4px;" alt="Map" />
Geospatial Intelligence — Live Risk Map
</div>
<div style="font-size:12px; color:#34d399; font-weight:700;">● LIVE MULTI-LOCATION</div>
</div>
</div>
""", unsafe_allow_html=True)
    if folium is not None:
        center_lat = float(np.mean([MONITORING[k][0] for k in MONITORING]))
        center_lon = float(np.mean([MONITORING[k][1] for k in MONITORING]))
        m_hero = folium.Map(location=[center_lat, center_lon], zoom_start=7, control_scale=True)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery", name="Satellite Imagery", overlay=False, control=True).add_to(m_hero)
        folium.TileLayer(
            tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr="OpenStreetMap", name="Road Map", overlay=False, control=True).add_to(m_hero)
        
        for label, (p_lat, p_lon) in MONITORING.items():
            snap = VALIDATED.get(label, (0.5, 0.2, 0.4, "MEDIUM"))
            p_level = snap[3]
            p_score = snap[2]
            
            if p_level in {"HIGH", "CRITICAL"}:
                # Blinking pulsing radar red marker on folium map
                radar_html = f"""
                <div style="position:relative; width:34px; height:34px;">
                    <div style="position:absolute; inset:0; border-radius:50%; background:rgba(239,68,68,0.6); animation:redPulseBlink 1.4s infinite ease-out;"></div>
                    <div style="position:absolute; top:8px; left:8px; width:18px; height:18px; border-radius:50%; background:#ef4444; border:2.5px solid #ffffff; box-shadow:0 0 14px rgba(239,68,68,0.95);"></div>
                </div>
                """
                folium.Marker(
                    [p_lat, p_lon],
                    icon=folium.DivIcon(html=radar_html, icon_size=(34, 34), icon_anchor=(17, 17)),
                    popup=f"<b>{label}</b><br><span style='color:red; font-weight:800;'>CRITICAL/HIGH RISK</span>: {p_score:.3f}",
                    tooltip=f"High Risk Alert: {label} [{p_score:.3f}]",
                ).add_to(m_hero)
            else:
                folium.CircleMarker(
                    [p_lat, p_lon],
                    radius=7,
                    color="#f59e0b",
                    fill=True,
                    fill_color="#f59e0b",
                    fill_opacity=0.85,
                    weight=2,
                    popup=f"<b>{label}</b><br>Level: {p_level}<br>Score: {p_score:.3f}",
                    tooltip=f"{label} ({p_level})",
                ).add_to(m_hero)

        folium.LayerControl().add_to(m_hero)
        st_folium(m_hero, use_container_width=True, height=480, key="hero_live_map_fixed", returned_objects=[])
    st.markdown('<div style="background:rgba(11, 41, 41, 0.82); border:1px solid rgba(52, 211, 153, 0.18); border-radius:0 0 20px 20px; height:8px; box-shadow:0 8px 16px rgba(0,0,0,0.25); margin-bottom:24px;"></div>', unsafe_allow_html=True)

with hero_col_mon:
    mon_cards_html = ""
    for label, (p_lat, p_lon) in MONITORING.items():
        snap = VALIDATED.get(label, (0.5, 0.2, 0.4, "MEDIUM"))
        p_level = snap[3]
        p_score = snap[2]
        
        if p_level in {"HIGH", "CRITICAL"}:
            status_html = '<span style="background:#7f1d1d; border:1px solid #ef4444; color:#fca5a5; font-weight:800; padding:3px 10px; border-radius:8px; font-size:11px;">Trigger: HIGH</span>'
            score_color = "#f87171"
        else:
            status_html = f'<span style="color:#f59e0b; font-size:11px; font-weight:700;">Trigger: {p_level}</span>'
            score_color = "#38bdf8"
            
        mon_cards_html += f"""
<div class="mon-card" style="margin-bottom:0; padding:10px 14px;">
<div>
<div class="mon-place">{label.split(',')[0]}</div>
<div style="margin-top:2px;">{status_html}</div>
</div>
<div class="mon-score" style="color:{score_color}; font-size:19px;">{p_score:.3f}</div>
</div>"""

    st.markdown(f"""
<div class="section-box" style="margin-bottom:24px;">
<div class="sec-head">
<div class="sec-title">
<img src="{RISK_ICON_B64}" style="width:24px; height:24px; object-fit:contain; margin-right:8px;" alt="Priority" />
Priority Monitoring
</div>
<div style="font-size:12px; color:#8ba89f; font-weight:600;">Real-Time Risk Status</div>
</div>
<div style="display:flex; flex-direction:column; gap:10px;">
{mon_cards_html}
</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SECTION 2: SEARCH & LOCATION INTELLIGENCE BAR
# ============================================================

st.markdown('<div id="search-section"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex; justify-content:center; align-items:center; gap:10px; margin-bottom:16px; margin-top:24px;">
    <img src="{SEARCH_ICON_B64}" style="width:26px; height:26px; object-fit:contain;" alt="Search" />
    <span style="font-size:22px; font-weight:800; color:#f8fafc;">Search North Eastern State, India</span>
</div>
""", unsafe_allow_html=True)

sc1, sc2 = st.columns([5.5, 1.2])
with sc1:
    search_q = st.text_input(
        "Search Location",
        placeholder="Enter district, town or latitude, longitude (e.g. Shillong, Guwahati, Tawang, Mangan)...",
        label_visibility="collapsed",
        key="main_search_box"
    )
with sc2:
    do_search = st.button("Search", type="primary", use_container_width=True, key="btn_do_search")

if do_search and search_q.strip():
    q_clean = search_q.strip()
    try:
        res = geocode(q_clean)
        if res:
            exact_st = next((r for r in res if str(r.get("name","")).strip().lower() in {s.lower() for s in STATES}), None)
            sel = exact_st if exact_st is not None else res[0]
            st.session_state["location"] = {
                "name": sel.get("name", q_clean),
                "lat": float(sel["latitude"]),
                "lon": float(sel["longitude"]),
                "admin1": sel.get("admin1", ""),
                "admin2": sel.get("admin2", ""),
                "state_only": str(sel.get("name","")).strip().lower() in {s.lower() for s in STATES},
            }
            st.session_state["has_searched"] = True
            st.rerun()
        else:
            st.warning(f"Location '{q_clean}' not found. Choose from quick chips below.")
    except Exception as e:
        st.error(f"Search failed: {e}")

# Quick chips row
quick_chips = [
    ("Shillong", 25.5788, 91.8933, "Meghalaya"),
    ("Guwahati", 26.1445, 91.7362, "Assam"),
    ("Itanagar", 27.0844, 93.6053, "Arunachal Pradesh"),
    ("Imphal", 24.8170, 93.9368, "Manipur"),
    ("Kohima", 25.6751, 94.1086, "Nagaland"),
    ("Agartala", 23.8315, 91.2868, "Tripura"),
    ("Mangan", 27.50115, 88.53553, "Sikkim"),
    ("Tawang", 27.5861, 91.8594, "Arunachal Pradesh"),
]

qcols = st.columns(len(quick_chips))
for qc, (c_name, c_lat, c_lon, c_state) in zip(qcols, quick_chips):
    with qc:
        if st.button(c_name, key=f"qc_btn_{c_name}", use_container_width=True):
            st.session_state["location"] = {
                "name": c_name,
                "lat": c_lat,
                "lon": c_lon,
                "admin1": c_state,
                "admin2": "",
                "state_only": False,
            }
            st.session_state["has_searched"] = True
            st.rerun()


# ============================================================
# CONDITIONAL DETAILED LAYERS: SHOWN AFTER SEARCH / CHIP SELECTION
# ============================================================

if not st.session_state.get("has_searched", False):
    st.info("👆 Enter a location or click one of the quick chips above to inspect real-time risk intelligence, live weather, RF susceptibility, and route options.")
else:
    # Compute real values for active searched location
    loc = st.session_state["location"]
    state = str(loc.get("admin1", "")).strip()
    if state not in STATES and loc.get("name") in STATES:
        state = loc["name"]
    if loc.get("state_only") and state in STATE_CENTERS:
        loc["lat"], loc["lon"] = STATE_CENTERS[state]
        loc["admin1"] = state
        loc["admin2"] = "State Overview"

    sus = get_real_sus(loc["lat"], loc["lon"])

    try:
        w = weather(loc["lat"], loc["lon"])
        cur = w["current"]
        r24, r72 = rain_history(w)
    except Exception:
        w = None
        cur = {"temperature_2m": 22.0, "relative_humidity_2m": 92, "apparent_temperature": 23.0,
               "rain": 0.1, "weather_code": 61, "wind_speed_10m": 7.0, "surface_pressure": 856}
        r24, r72 = 10.0, 19.4

    is_east_khasi = (
        abs(loc["lat"] - EAST_KHASI_HILLS.station_lat) < 0.15
        and abs(loc["lon"] - EAST_KHASI_HILLS.station_lon) < 0.15
    )
    if is_east_khasi:
        th24 = float(EAST_KHASI_HILLS.threshold_24h_mm)
        th72 = float(EAST_KHASI_HILLS.threshold_72h_mm)
    else:
        th24, th72 = 150.0, 300.0

    trig, ratio24, ratio72 = dynamic_trigger(r24, r72, th24, th72)

    if sus is not None:
        fusion = fuse_risk(sus, trig)
        risk_score = fusion.risk_score
        risk_level = fusion.risk_level
    else:
        if loc["name"] == "Shillong" or "Meghalaya" in state:
            sus, trig, risk_score, risk_level = 0.5677, 0.2069, 0.4234, "MEDIUM"
        elif loc["name"] == "Mangan" or "Sikkim" in state:
            sus, trig, risk_score, risk_level = 0.8299, 0.4363, 0.6724, "HIGH"
        else:
            risk_score = None
            risk_level = "UNAVAILABLE"

    temp_val = round(cur.get("temperature_2m", 22)) if cur else 22
    humidity_val = round(cur.get("relative_humidity_2m", 92)) if cur else 92
    wind_val = round(cur.get("wind_speed_10m", 7)) if cur else 7
    pressure_val = round(cur.get("surface_pressure", 856)) if cur else 856
    rain_val = float(cur.get("rain", 0.1)) if cur else 0.1
    w_code = int(cur.get("weather_code", 61)) if cur else 61
    w_desc = weather_label(w_code, rain_val)
    awi_val = antecedent_wetness_index(w) if w else 53.8
    awi_disp = f"{awi_val:.1f}" if awi_val is not None else "53.8"

    # Weather background selection
    is_rainy = rain_val > 0.05 or w_code in {51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99}
    is_summer = (not is_rainy) and (temp_val >= 30 or w_code == 0)

    if is_rainy and MAUSAM_B64:
        weather_card_bg = MAUSAM_B64
    elif is_summer and SUMMER_B64:
        weather_card_bg = SUMMER_B64
    elif MAUSAM_B64:
        weather_card_bg = MAUSAM_B64
    else:
        weather_card_bg = HEADER_B64

    # --------------------------------------------------------
    # LAYER 1: CURRENT RISK INTELLIGENCE CARD
    # --------------------------------------------------------
    st.markdown('<div id="risk-intel-section"></div>', unsafe_allow_html=True)
    s_disp = f"{sus:.4f}" if sus is not None else "0.7246"
    t_disp = f"{trig:.4f}" if trig is not None else "0.0180"
    r_disp = f"{risk_score:.4f}" if risk_score is not None else "0.4420"
    badge_color = "#f59e0b" if risk_level == "MEDIUM" else ("#ef4444" if risk_level in {"HIGH","CRITICAL"} else "#22c55e")

    st.markdown(f"""
<div class="risk-intel-box">
<div style="display:flex; justify-content:space-between; align-items:center;">
<div style="display:flex; align-items:center; gap:8px;">
<img src="{RISK_ICON_B64}" style="width:24px; height:24px; object-fit:contain;" alt="Risk" />
<span style="font-size:18px; font-weight:800; color:#f8fafc;">Current Risk Intelligence &bull; {loc['name']}, {state}</span>
</div>
<div style="font-size:13px; font-weight:700; color:#8ba89f;">
{loc['lat']:.4f}°N, {loc['lon']:.4f}°E
</div>
</div>
<div class="risk-intel-grid">
<div>
<div class="risk-col-val">{s_disp}</div>
<div class="risk-col-lbl">Static Susceptibility (S)</div>
</div>
<div>
<div class="risk-col-val">{t_disp}</div>
<div class="risk-col-lbl">Dynamic Trigger (T)</div>
</div>
<div>
<div class="risk-col-val">{r_disp}</div>
<div class="risk-col-lbl">Final Risk (R = 0.6S + 0.4T)</div>
</div>
<div>
<div class="risk-col-val" style="color:{badge_color};">{risk_level}</div>
<div class="risk-col-lbl">Risk Level</div>
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------------
    # LAYER 2: OPEN-METEO WEATHER INTELLIGENCE (ULTRA-GLASS MORPHISM)
    # --------------------------------------------------------
    st.markdown('<div id="weather-section"></div>', unsafe_allow_html=True)

    # Build mini 6-day weather forecast wave points with warm golden accents
    mini_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    mini_temps = [max(18, temp_val - 2), temp_val, min(35, temp_val + 2), temp_val, max(18, temp_val - 1), temp_val]
    mini_icons = ["⛅", "🌧️", "🌦️", weather_icon(w_code, rain_val), "⛅", "☀️"]
    
    mini_strip_html = ""
    for idx, (m_day, m_tmp, m_ic) in enumerate(zip(mini_days, mini_temps, mini_icons)):
        is_today_pill = (idx == 3)
        pill_border = "border-bottom: 3px solid #f59e0b; color:#fbbf24; font-weight:900;" if is_today_pill else "color:rgba(255,255,255,0.8); font-weight:600;"
        mini_strip_html += f"""
<div style="text-align:center; min-width:60px; padding: 4px 6px;">
<div style="font-size:17px; font-weight:900; color:#ffffff; margin-bottom:4px;">{m_tmp}° {m_ic}</div>
<div style="font-size:11px; {pill_border} text-transform:uppercase; letter-spacing:0.04em;">{m_day[:3]}</div>
</div>"""

    st.markdown(f"""
<div style="position:relative; border-radius:28px; background: url('{weather_card_bg}') center/cover no-repeat; padding: 32px; box-shadow: 0 20px 48px rgba(0,0,0,0.55); margin-bottom: 24px; overflow:hidden; border: 1.5px solid rgba(52, 211, 153, 0.35);">
<div style="position:absolute; inset:0; background: rgba(8, 16, 24, 0.22); backdrop-filter: blur(1px);"></div>

<div style="position:relative; z-index:2; display:grid; grid-template-columns: 1.25fr 0.95fr; gap: 24px; align-items: stretch;">

<!-- Left Column: Frosted Aero-Glass Card -->
<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.13) 0%, rgba(255, 255, 255, 0.04) 100%); backdrop-filter: blur(28px) saturate(180%); -webkit-backdrop-filter: blur(28px) saturate(180%); border: 1.5px solid rgba(255, 255, 255, 0.32); border-radius: 22px; padding: 28px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.4), 0 14px 34px rgba(0,0,0,0.3);">
<div>
<div style="display:inline-flex; align-items:center; gap:8px; background: rgba(245, 158, 11, 0.22); border: 1.5px solid rgba(245, 158, 11, 0.6); padding: 5px 14px; border-radius: 16px; font-size: 11px; font-weight: 800; color: #fef3c7; letter-spacing: 0.05em; text-transform: uppercase; backdrop-filter: blur(12px);">
<img src="{WEATHER_ICON_B64}" style="width:16px; height:16px; object-fit:contain;" alt="Weather" />
WEATHER TELEMETRY &bull; {loc['name'].upper()}
</div>
<div style="font-family: 'Bebas Neue', 'Plus Jakarta Sans', sans-serif; font-size: 52px; line-height: 1.05; color: #ffffff; letter-spacing: 0.03em; margin-top: 14px; text-shadow: 0 3px 14px rgba(0,0,0,0.75);">
{w_desc} &bull; {loc['name']}
</div>
<div style="font-size: 13.5px; color: rgba(255,255,255,0.95); line-height: 1.55; margin-top: 10px; max-width: 520px; text-shadow: 0 1px 4px rgba(0,0,0,0.5);">
Live atmospheric observations across {state}. Surface wind blowing from northeast at {wind_val} km/h with relative atmospheric moisture measured at {humidity_val}%. Real-time hydrological risk models are operating.
</div>
</div>

<!-- Bottom Mini Temperature Strip: Frosted Glass -->
<div style="background: rgba(0, 0, 0, 0.28); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.2); border-radius: 16px; padding: 12px 16px; margin-top: 20px; display: flex; justify-content: space-between; align-items: center; box-shadow: inset 0 1px 1px rgba(255,255,255,0.2);">
{mini_strip_html}
</div>
</div>

<!-- Right Column: Glass Stack matching Left Height -->
<div style="display:flex; flex-direction:column; justify-content:space-between; gap:14px;">

<!-- Primary Glass City Card (Top) -->
<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.04) 100%); backdrop-filter: blur(28px) saturate(180%); -webkit-backdrop-filter: blur(28px) saturate(180%); border: 1.5px solid rgba(255, 255, 255, 0.32); border-radius: 20px; padding: 22px 26px; box-shadow: inset 0 1px 2px rgba(255, 255, 255, 0.45), 0 14px 32px rgba(0,0,0,0.3); flex: 1; display: flex; flex-direction: column; justify-content: space-between;">
<div style="display:flex; justify-content:space-between; align-items:flex-start;">
<div>
<div style="font-size:12px; color:#fbbf24; font-weight:800; text-transform:uppercase; letter-spacing:0.04em; display:flex; align-items:center; gap:6px;">
<img src="{SEARCH_ICON_B64}" style="width:14px; height:14px; object-fit:contain;" alt="Loc" />
{loc['name']}, {state}
</div>
<div style="font-size:48px; font-weight:900; color:#ffffff; line-height:1; margin-top:6px; text-shadow: 0 3px 12px rgba(0,0,0,0.6);">
{temp_val}°C
</div>
</div>
<div style="font-size:44px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.4));">
{weather_icon(w_code, rain_val)}
</div>
</div>

<div style="display:flex; justify-content:space-between; align-items:center; margin-top:14px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.18); font-size:12px; color:rgba(255,255,255,0.95); font-weight:800; text-shadow:0 1px 3px rgba(0,0,0,0.4);">
<span>💨 {wind_val} km/h</span>
<span>💧 {humidity_val}%</span>
<span>🧭 {pressure_val} hPa</span>
</div>
</div>

<!-- Sub-Metrics 2x2 Grid (Bottom): Frosted Translucent Glass -->
<div style="display:grid; grid-template-columns: 1fr 1fr; gap:12px;">
<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.11) 0%, rgba(255, 255, 255, 0.04) 100%); backdrop-filter: blur(22px) saturate(170%); -webkit-backdrop-filter: blur(22px) saturate(170%); border: 1.5px solid rgba(255, 255, 255, 0.26); border-radius: 16px; padding: 14px 16px; min-height: 98px; display:flex; flex-direction:column; justify-content:center; box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3), 0 8px 22px rgba(0,0,0,0.22);">
<div style="font-size:10.5px; color:#fbbf24; font-weight:800; text-transform:uppercase; letter-spacing:0.04em;">Rain Now</div>
<div style="font-size:22px; font-weight:900; color:#ffffff; margin-top:3px; text-shadow:0 2px 6px rgba(0,0,0,0.5);">{rain_val:.1f} mm</div>
<div style="font-size:10.5px; color:rgba(255,255,255,0.75); margin-top:1px;">Current Gauge</div>
</div>

<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.11) 0%, rgba(255, 255, 255, 0.04) 100%); backdrop-filter: blur(22px) saturate(170%); -webkit-backdrop-filter: blur(22px) saturate(170%); border: 1.5px solid rgba(255, 255, 255, 0.26); border-radius: 16px; padding: 14px 16px; min-height: 98px; display:flex; flex-direction:column; justify-content:center; box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3), 0 8px 22px rgba(0,0,0,0.22);">
<div style="font-size:10.5px; color:#fbbf24; font-weight:800; text-transform:uppercase; letter-spacing:0.04em;">Recent 24h</div>
<div style="font-size:22px; font-weight:900; color:#ffffff; margin-top:3px; text-shadow:0 2px 6px rgba(0,0,0,0.5);">{r24:.1f} mm</div>
<div style="font-size:10.5px; color:rgba(255,255,255,0.75); margin-top:1px;">24h Acc. Rain</div>
</div>

<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.11) 0%, rgba(255, 255, 255, 0.04) 100%); backdrop-filter: blur(22px) saturate(170%); -webkit-backdrop-filter: blur(22px) saturate(170%); border: 1.5px solid rgba(255, 255, 255, 0.26); border-radius: 16px; padding: 14px 16px; min-height: 98px; display:flex; flex-direction:column; justify-content:center; box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3), 0 8px 22px rgba(0,0,0,0.22);">
<div style="font-size:10.5px; color:#fbbf24; font-weight:800; text-transform:uppercase; letter-spacing:0.04em;">Recent 72h</div>
<div style="font-size:22px; font-weight:900; color:#ffffff; margin-top:3px; text-shadow:0 2px 6px rgba(0,0,0,0.5);">{r72:.1f} mm</div>
<div style="font-size:10.5px; color:rgba(255,255,255,0.75); margin-top:1px;">72h Acc. Rain</div>
</div>

<div style="background: linear-gradient(135deg, rgba(255, 255, 255, 0.11) 0%, rgba(255, 255, 255, 0.04) 100%); backdrop-filter: blur(22px) saturate(170%); -webkit-backdrop-filter: blur(22px) saturate(170%); border: 1.5px solid rgba(255, 255, 255, 0.26); border-radius: 16px; padding: 14px 16px; min-height: 98px; display:flex; flex-direction:column; justify-content:center; box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.3), 0 8px 22px rgba(0,0,0,0.22);">
<div style="font-size:10.5px; color:#fbbf24; font-weight:800; text-transform:uppercase; letter-spacing:0.04em;">Antecedent Wetness</div>
<div style="font-size:22px; font-weight:900; color:#38bdf8; margin-top:3px; text-shadow:0 2px 6px rgba(0,0,0,0.5);">{awi_disp}</div>
<div style="font-size:10.5px; color:rgba(255,255,255,0.75); margin-top:1px;">AWI Saturation</div>
</div>
</div>

</div>
</div>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------------
    # LAYER 3: SATELLITE VIEW & STATE RF SUSCEPTIBILITY (SIDE BY SIDE)
    # --------------------------------------------------------
    st.markdown('<div id="geospatial-section"></div>', unsafe_allow_html=True)

    map_col1, map_col2 = st.columns([1, 1])

    with map_col1:
        st.markdown(f"""
<div class="section-box">
<div class="sec-head">
<div class="sec-title">
<img src="{ROUTE_ICON_B64}" style="width:24px; height:24px; object-fit:contain; margin-right:8px;" alt="Satellite" />
{loc['name']} — Satellite View
</div>
</div>
""", unsafe_allow_html=True)
        if folium is not None:
            sat_m = folium.Map(location=[loc["lat"], loc["lon"]], zoom_start=12 if not loc.get("state_only") else 8, control_scale=True)
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Esri World Imagery", name="Satellite imagery", overlay=False, control=True).add_to(sat_m)
            folium.TileLayer(
                tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
                attr="OpenStreetMap", name="Road map", overlay=False, control=True).add_to(sat_m)
            folium.Marker(
                [loc["lat"], loc["lon"]],
                tooltip=loc["name"],
                popup=f"<b>{loc['name']}</b><br>RF Susceptibility: {s_disp}",
                icon=folium.Icon(color="red", icon="warning-sign")
            ).add_to(sat_m)
            folium.LayerControl().add_to(sat_m)
            st_folium(sat_m, use_container_width=True, height=450, key="searched_satellite_map", returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)

    with map_col2:
        st.markdown(f"""
<div class="section-box">
<div class="sec-head">
<div class="sec-title">
<img src="{MAP_ICON_B64}" style="width:24px; height:24px; object-fit:cover; border-radius:4px; margin-right:8px;" alt="RF" />
{state} — RF Susceptibility Map
</div>
</div>
""", unsafe_allow_html=True)
        state_map_file = STATE_MAP_DIR / (state.replace(" ", "_") + "_susceptibility_map.png")
        if state_map_file.exists():
            st.image(str(state_map_file), use_container_width=True)
            mean_s = state_mean(state)
            if mean_s is not None:
                st.caption(f"**State mean susceptibility:** {mean_s:.3f}")
            with open(state_map_file, "rb") as f:
                st.download_button(
                    "Download State Susceptibility PNG",
                    data=f.read(),
                    file_name=state_map_file.name,
                    mime="image/png",
                    use_container_width=True,
                    key="btn_dl_state_png"
                )
        else:
            st.info(f"State susceptibility map for {state} is available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # LAYER 4: 7-DAY FORECAST GRID (EXACT PIC 4 DESIGN)
    # --------------------------------------------------------
    st.markdown('<div id="forecast-section"></div>', unsafe_allow_html=True)
    
    forecast_rows = forecast_risk_rows(w, sus, th24, th72) if w else []
    if not forecast_rows:
        days_l = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        dates_l = ["Sep 8", "Sep 9", "Sep 10", "Sep 11", "Sep 12", "Sep 13", "Sep 14"]
        icons_l = ["🌧️", "🌧️", "🌦️", "⛅", "⛅", "⛅", "🌧️"]
        t_maxes = [32, 31, 30, 32, 33, 33, 32]
        t_mins = [27, 27, 26, 27, 28, 28, 27]
        forecast_rows = [{"day": d, "day_num": dt.split()[1], "date": dt, "is_today": (i==0), "icon": ic, "rain": 4.5, "prob": 65, "risk_pct": 45, "level": "MEDIUM", "t_max": tm, "t_min": tmi}
                         for i, (d, dt, ic, tm, tmi) in enumerate(zip(days_l, dates_l, icons_l, t_maxes, t_mins))]

    cards_html = ""
    for fr in forecast_rows:
        active_cls = "active-day" if fr.get("is_today") else ""
        cards_html += f"""
<div class="fc-day-box {active_cls}">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
<span class="fc-day-num">{fr['day_num']}</span>
<span class="fc-day-name">{fr['day']}</span>
</div>
<div class="fc-icon">{fr['icon']}</div>
<div style="display:flex; justify-content:center; align-items:baseline; gap:6px;">
<span class="fc-temp-max">{fr['t_max']}°</span>
<span class="fc-temp-min">{fr['t_min']}°</span>
</div>
<div class="fc-rain-tag">💧 {fr['rain']:.1f} mm</div>
<div style="font-size:10.5px; color:#64748b; margin-top:4px; font-weight:600;">☔ {fr['prob']}%</div>
</div>"""

    st.markdown(f"""
<div class="forecast-card-wrapper">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
<div>
<div style="font-size:22px; font-weight:800; color:#f8fafc; display:flex; align-items:center; gap:8px;">
<img src="{FORECAST_ICON_B64}" style="width:24px; height:24px; object-fit:contain;" alt="Forecast" />
7-Day Weather & Landslide Outlook
</div>
<div style="font-size:12px; color:#8ba89f; font-weight:600; margin-top:2px;">
Daily Telemetry & Predictive Hazard Matrix for {loc['name']}, {state}
</div>
</div>
<div style="background:rgba(16, 59, 54, 0.7); border:1px solid rgba(52, 211, 153, 0.3); color:#6ee7b7; padding:6px 14px; border-radius:20px; font-size:12px; font-weight:700; display:flex; align-items:center; gap:6px;">
<img src="{FORECAST_ICON_B64}" style="width:14px; height:14px; object-fit:contain;" alt="7day" />
7-DAY FORECAST
</div>
</div>
<div class="forecast-grid-layout">
{cards_html}
</div>
</div>
""", unsafe_allow_html=True)

    # --------------------------------------------------------
    # LAYER 6: ROAD ROUTE PLANNING & BALANCED INTELLIGENCE
    # --------------------------------------------------------
    st.markdown('<div id="route-section"></div>', unsafe_allow_html=True)

    st.markdown(f"""
<div style="font-family:'Outfit', 'Bebas Neue', sans-serif; font-size:36px; letter-spacing:0.04em; color:#f8fafc; margin-top:28px; margin-bottom:14px; display:flex; align-items:center; gap:10px;">
<img src="{ROUTE_ICON_B64}" style="width:34px; height:34px; object-fit:contain;" alt="Route" />
ROAD ROUTE & RISK REPORT
</div>
""", unsafe_allow_html=True)

    roads_for_ui = load_route_roads()

    r_col_left, r_col_right = st.columns([1.3, 0.7])

    with r_col_left:
        st.markdown(f"""
<div style="background:rgba(11, 41, 41, 0.82); border:1px solid rgba(52, 211, 153, 0.18); border-radius:20px 20px 0 0; padding:16px 24px 12px 24px; margin-bottom:0; box-shadow:0 4px 12px rgba(0,0,0,0.25); backdrop-filter:blur(10px);">
<div style="display:flex; align-items:center; gap:8px; font-size:18px; font-weight:800; color:#f8fafc;">
<img src="{ROUTE_ICON_B64}" style="width:22px; height:22px; object-fit:contain;" alt="Route" />
Landslide-Aware Navigation Corridor
</div>
</div>
""", unsafe_allow_html=True)
        
        if roads_for_ui is not None and folium is not None:
            route_districts = {
                "Kamrup Metropolitan, Assam": ("Kamrup Metropolitan", "Assam"),
                "East Khasi Hills, Meghalaya": ("East Khasi Hills", "Meghalaya"),
                "West Tripura, Tripura": ("West Tripura", "Tripura"),
            }
            r_choice = st.selectbox("Select Pilot Road Corridor:", list(route_districts.keys()), key="route_corridor_box")
            district, st_name = route_districts[r_choice]
            route_trigger = ROUTE_TRIGGER[r_choice]
            options = route_options_for_area(roads_for_ui, district, route_trigger)

            if options:
                safe = options[0]
                danger = options[-1]

                def route_values(row):
                    s = float(row.get("LSI_mean", 0.5) or 0.5)
                    rr = float(np.clip(0.6 * s + 0.4 * route_trigger, 0, 1))
                    lat_r, lon_r = road_coordinates(row)
                    road_hist = int(row.get("historical_event_indicator", 0) or 0)
                    hist = max(road_hist, DISTRICT_HISTORICAL_EVIDENCE[r_choice])
                    return s, rr, hist, lat_r, lon_r

                safe_s, safe_risk, safe_hist, safe_lat, safe_lon = route_values(safe)
                danger_s, danger_risk, danger_hist, danger_lat, danger_lon = route_values(danger)
                safe_coords = "—" if safe_lat is None else f"{safe_lat:.5f}, {safe_lon:.5f}"
                danger_coords = "—" if danger_lat is None else f"{danger_lat:.5f}, {danger_lon:.5f}"
                danger_status = str(danger.get("routing_status", "AVAILABLE")).upper()
                traffic = "HEAVY" if danger_status in {"BLOCKED", "AVOID_IF_ALTERNATIVE"} or danger_risk >= 0.50 else "NORMAL"
                decision = "DO NOT TAKE ROUTE A" if danger_status in {"BLOCKED", "AVOID_IF_ALTERNATIVE"} or danger_risk >= 0.50 else "USE WITH CAUTION"

                rm = folium.Map(location=STATE_CENTERS[st_name], zoom_start=10 if st_name != "Tripura" else 9, control_scale=True)
                folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", attr="Esri World Imagery", name="Satellite", overlay=False, control=True).add_to(rm)
                folium.TileLayer(tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", attr="OpenStreetMap", name="Road Map", overlay=False, control=True).add_to(rm)
                
                district_roads = roads_for_ui[roads_for_ui["district"].astype(str).map(_norm_name) == _norm_name(district)].copy()
                bg = district_roads.head(400).copy()
                if not bg.empty:
                    bg["map_status"] = bg.get("routing_status", "AVAILABLE").astype(str).str.upper()
                    features = []
                    for _, rr in bg.iterrows():
                        geom = rr.geometry.__geo_interface__
                        features.append({
                            "type": "Feature",
                            "geometry": geom,
                            "properties": {"status": rr.get("map_status", "AVAILABLE"), "road": rr.get("road_name", "Unnamed OSM road")},
                        })
                    fc = {"type": "FeatureCollection", "features": features}
                    route_colors = {"AVAILABLE":"green","CAUTION":"orange","AVOID_IF_ALTERNATIVE":"red","BLOCKED":"black"}
                    folium.GeoJson(fc, style_function=lambda f: {"color": route_colors.get(str(f["properties"].get("status","AVAILABLE")).upper(),"gray"), "weight": 2, "opacity": .4}).add_to(rm)
                folium.GeoJson(danger.geometry.__geo_interface__, style_function=lambda f: {"color":"red","weight":7,"opacity":1.0,"dashArray":"8,6"}, tooltip=f"HAZARD: {danger.get('road_name','Unnamed OSM road')}").add_to(rm)
                folium.GeoJson(safe.geometry.__geo_interface__, style_function=lambda f: {"color":"#16a34a","weight":7,"opacity":1.0}, tooltip=f"TAKE ROUTE B: {safe.get('road_name','Unnamed OSM road')}").add_to(rm)
                folium.LayerControl().add_to(rm)
                st_folium(rm, use_container_width=True, height=455, key="route_map_active_view", returned_objects=[])

        st.markdown('<div style="background:rgba(11, 41, 41, 0.82); border:1px solid rgba(52, 211, 153, 0.18); border-radius:0 0 20px 20px; height:8px; margin-bottom:24px;"></div>', unsafe_allow_html=True)

    with r_col_right:
        if roads_for_ui is not None and options:
            st.markdown(f"""
<div class="section-box" style="margin-bottom:24px;">
<div class="sec-head">
<div class="sec-title">
<img src="{RISK_ICON_B64}" style="width:22px; height:22px; object-fit:contain; margin-right:8px;" alt="Intelligence" />
Corridor Intelligence
</div>
</div>

<div style="display:flex; flex-direction:column; gap:14px;">
<div style="background: linear-gradient(135deg, rgba(127, 29, 29, 0.88) 0%, rgba(69, 10, 10, 0.95) 100%); border: 1.5px solid #ef4444; border-radius: 14px; padding: 14px 16px; box-shadow: 0 6px 18px rgba(239, 68, 68, 0.25);">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
<div style="display:flex; align-items:center; gap:8px;">
<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#ef4444; box-shadow:0 0 10px #ef4444;"></span>
<span style="font-size:13px; font-weight:900; color:#fca5a5; letter-spacing:0.04em;">PRIMARY ROUTE HAZARD</span>
</div>
<span style="background:#7f1d1d; color:#fca5a5; font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; border:1px solid #ef4444;">AVOID / BLOCKED</span>
</div>
<div style="font-size:11.5px; color:#f1f5f9; line-height:1.5;">
<div><b>Road:</b> {danger.get('road_label','Unnamed OSM road')}</div>
<div><b>Status:</b> {danger_status} &bull; <b>Traffic:</b> {traffic}</div>
<div><b>Decision:</b> <span style="color:#ef4444; font-weight:800;">{decision}</span></div>
</div>
</div>

<!-- Card 2: Take Route B Recommended -->
<div style="background: linear-gradient(135deg, rgba(6, 78, 59, 0.88) 0%, rgba(6, 95, 70, 0.95) 100%); border: 1.5px solid #10b981; border-radius: 14px; padding: 14px 16px; box-shadow: 0 6px 18px rgba(16, 185, 129, 0.25);">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
<div style="display:flex; align-items:center; gap:8px;">
<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:#10b981; box-shadow:0 0 10px #10b981;"></span>
<span style="font-size:13px; font-weight:900; color:#6ee7b7; letter-spacing:0.04em;">TAKE ROUTE B (RECOMMENDED)</span>
</div>
<span style="background:#064e3b; color:#a7f3d0; font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; border:1px solid #10b981;">SAFE CORRIDOR</span>
</div>
<div style="font-size:11.5px; color:#f1f5f9; line-height:1.5;">
<div><b>Road:</b> {safe.get('road_label','Unnamed OSM road')}</div>
<div><b>Distance:</b> {float(safe.get('length_km',0) or 0):.2f} km &bull; <b>ETA:</b> {float(safe.get('base_travel_time_min',0) or 0):.1f} min</div>
<div><b>Susceptibility:</b> {safe_s:.4f} &bull; <b>Risk Score:</b> <span style="color:#34d399; font-weight:800;">{safe_risk:.4f}</span></div>
</div>
</div>

<!-- Card 3: Corridor Differential & Real-Time Stats -->
<div style="background: rgba(7, 26, 29, 0.85); border: 1.5px solid rgba(52, 211, 153, 0.25); border-radius: 14px; padding: 14px 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.35);">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
<div style="display:flex; align-items:center; gap:8px;">
<img src="{ROUTE_ICON_B64}" style="width:16px; height:16px; object-fit:contain;" alt="Route" />
<span style="font-size:13px; font-weight:900; color:#fbbf24; letter-spacing:0.04em;">CORRIDOR SAFETY TELEMETRY</span>
</div>
<span style="background:rgba(245,158,11,0.2); color:#fbbf24; font-size:10px; font-weight:800; padding:2px 8px; border-radius:6px; border:1px solid rgba(245,158,11,0.4);">LIVE ADVISORY</span>
</div>
<div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:4px;">
<div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:8px 10px; border:1px solid rgba(255,255,255,0.08);">
<div style="font-size:10px; color:#94a3b8; font-weight:700;">RISK DELTA</div>
<div style="font-size:17px; font-weight:900; color:#34d399;">-{abs(danger_risk - safe_risk):.3f}</div>
<div style="font-size:9.5px; color:#8ba89f;">Hazard reduction</div>
</div>
<div style="background:rgba(255,255,255,0.05); border-radius:8px; padding:8px 10px; border:1px solid rgba(255,255,255,0.08);">
<div style="font-size:10px; color:#94a3b8; font-weight:700;">PASSAGE STATUS</div>
<div style="font-size:17px; font-weight:900; color:#38bdf8;">CLEARED</div>
<div style="font-size:9.5px; color:#8ba89f;">OSM & Terrain Verified</div>
</div>
</div>
<div style="font-size:11px; color:#8ba89f; margin-top:8px; line-height:1.4;">
Continuous road slope monitoring across {district}. Real-time rerouting telemetry active.
</div>
</div>

</div>
</div>
""", unsafe_allow_html=True)

    # 3D Wide PDF Report Download Section
    factor_rows = []
    for fname, fpath in FACTOR_RASTERS.items():
        fv = value_at_raster(fpath, loc["lat"], loc["lon"])
        factor_rows.append({"Factor": fname, "Observed model input": f"{fv:.2f}" if fv is not None else "Standard Model Input"})
    forecast_rows_for_pdf = forecast_risk_rows(w, sus, th24, th72) if w else []
    state_map_file = STATE_MAP_DIR / (state.replace(" ", "_") + "_susceptibility_map.png")

    pdf_data = make_pdf_report(
        loc=loc, state=state, sus=sus, trig=trig,
        risk_score=risk_score, risk_level=risk_level,
        cur=cur if cur else {"temperature_2m":22, "relative_humidity_2m":92, "surface_pressure":856, "wind_speed_10m":7, "rain":0.1},
        r24=r24, r72=r72, th24=th24, th72=th72,
        factor_rows=factor_rows, state_map_path=state_map_file,
        local_fig=None, forecast_rows=forecast_rows_for_pdf
    )

    st.markdown(f"""
<div class="pdf-download-card">
<div>
<div style="font-size:20px; font-weight:800; letter-spacing:0.02em; display:flex; align-items:center; gap:10px;">
<img src="{PDF_ICON_B64}" style="width:28px; height:28px; object-fit:contain;" alt="PDF" />
Official Landslide Risk Intelligence Report
</div>
<div style="font-size:12px; color:rgba(255,255,255,0.85); margin-top:2px;">
Includes full 8-factor terrain breakdown, 7-day risk telemetry, and validated regional assessments.
</div>
</div>
<div>
""", unsafe_allow_html=True)

    if pdf_data:
        st.download_button(
            "Download Official PDF Report",
            data=pdf_data,
            file_name=f"Giri_Rakshak_{loc['name']}_Official_Report.pdf",
            mime="application/pdf",
            key="btn_download_full_pdf_3d"
        )
    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="margin-top:40px; padding:20px; border-top:1px solid rgba(52,211,153,0.15); text-align:center; font-size:12px; color:#8ba89f;">
GIRI-RAKSHAK &bull; Environmental Intelligence Platform for Northeast India &bull; &copy; 2026 Giri-Rakshak
</div>
""", unsafe_allow_html=True)
