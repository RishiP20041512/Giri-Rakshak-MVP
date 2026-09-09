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
    from streamlit_mic_recorder import speech_to_text
except Exception:
    speech_to_text = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.units import mm
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
# ============================================================
# HISTORICAL LANDSLIDE EVENTS
# ============================================================

HISTORICAL_EVENTS = {
    "Agartala / West Tripura, Tripura": [
        {
            "event_id": "TRIPURA-HIST-001",
            "date": "2024-06-15",
            "location": "Mungiakami / Teliamura",
            "district": "Khowai",
            "latitude": 23.88223,
            "longitude": 91.70670,
            "type": "Landslide",
            "severity": "Moderate",
            "status": "HISTORICAL",
        }
    ],

    "Guwahati / Kamrup Metropolitan, Assam": [],

    "East Khasi Hills, Meghalaya": [],

    "Senapati, Manipur": [],

    "Mon, Nagaland": [],

    "Mangan, Sikkim": [],

    "Itanagar, Arunachal Pradesh": [],
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
    params = {
        "name": q if "," in q else f"{q}, India",
        "count": 20,
        "language": "en",
        "format": "json",
        "countryCode": "IN",
    }
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode())

    out = []
    for x in data.get("results", []):
        if str(x.get("country_code", "")).upper() != "IN":
            continue
        admin = str(x.get("admin1", "")).strip()
        name = str(x.get("name", "")).strip()
        if admin in STATES or name in STATES:
            out.append(x)
    return out


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
    h = payload["hourly"]
    rain = pd.Series(h["precipitation"], dtype=float).fillna(0)
    return float(rain.iloc[-24:].sum()), float(rain.iloc[-72:].sum())


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
    if not STATE_CSV.exists():
        return None
    try:
        sdf = pd.read_csv(STATE_CSV)
        for _, row in sdf.iterrows():
            raw = str(row.iloc[0])
            cleaned = raw.replace("ā", "a").replace("ī", "i").replace("ṅ", "n").replace("ē", "e").replace("ḷ", "l")
            if state.lower() in cleaned.lower():
                for col in sdf.columns:
                    try:
                        v = float(row[col])
                        if np.isfinite(v):
                            return v
                    except Exception:
                        pass
    except Exception:
        pass
    return None


def make_pdf_report(loc, state, sus, trig, risk_score, risk_level, cur, r24, r72,
                    th24, th72, factor_rows, state_map_path, local_fig):
    if colors is None:
        return None

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14*mm,
        leftMargin=14*mm,
        topMargin=14*mm,
        bottomMargin=14*mm,
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], alignment=TA_CENTER,
        fontSize=20, leading=24, spaceAfter=8
    )
    h2 = ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=13,
        leading=16, spaceBefore=8, spaceAfter=5
    )
    body = styles["BodyText"]

    story = []
    story.append(Paragraph("Giri-Rakshak", title))
    story.append(Paragraph("Landslide Risk Intelligence Report", title))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"<b>Location:</b> {loc['name']} — {loc.get('admin2','')}, {state}<br/>"
        f"<b>Coordinates:</b> {loc['lat']:.5f}, {loc['lon']:.5f}<br/>"
        f"<b>Generated:</b> {datetime.now().strftime('%d %b %Y, %H:%M')}",
        body
    ))

    story.append(Paragraph("Risk Summary", h2))
    risk_data = [
        ["Static S", "Dynamic T", "Final R", "Risk"],
        [
            "Unavailable" if sus is None else f"{sus:.4f}",
            f"{trig:.4f}",
            "Unavailable" if risk_score is None else f"{risk_score:.4f}",
            risk_level,
        ],
    ]
    t = Table(risk_data, colWidths=[35*mm]*4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#0b3d2e")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), .5, colors.grey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0,0), (-1,0), 7),
        ("TOPPADDING", (0,0), (-1,0), 7),
    ]))
    story.append(t)

    story.append(Paragraph("Dynamic Conditions", h2))
    dyn_data = [
        ["Temperature", "Humidity", "Pressure", "Wind"],
        [
            f"{cur['temperature_2m']:.1f} °C",
            f"{cur['relative_humidity_2m']:.0f}%",
            f"{cur['surface_pressure']:.0f} hPa",
            f"{cur['wind_speed_10m']:.1f} km/h",
        ],
        ["Rain now", "Recent 24h", "Recent 72h", "Rain probability"],
        [
            f"{cur['rain']:.1f} mm",
            f"{r24:.1f} mm",
            f"{r72:.1f} mm",
            f"{max(cur.get('precipitation',0),0):.1f} mm current",
        ],
    ]
    td = Table(dyn_data, colWidths=[35*mm]*4)
    td.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), .5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#e8f2ed")),
        ("BACKGROUND", (0,2), (-1,2), colors.HexColor("#e8f2ed")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,2), (-1,2), "Helvetica-Bold"),
    ]))
    story.append(td)
    story.append(Paragraph(
        f"<b>24h threshold:</b> {th24:.0f} mm &nbsp;&nbsp; "
        f"<b>72h threshold:</b> {th72:.0f} mm",
        body
    ))

    story.append(Paragraph("Eight RF Conditioning Factors", h2))
    ft = [["Factor", "Observed model input"]] + [
        [str(r["Factor"]), str(r["Observed model input"])]
        for r in factor_rows
    ]
    tf = Table(ft, colWidths=[65*mm, 95*mm])
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
        story.append(RLImage(str(state_map_path), width=170*mm, height=125*mm))

    if local_fig is not None:
        local_buf = BytesIO()
        local_fig.savefig(local_buf, format="png", dpi=150, bbox_inches="tight")
        local_buf.seek(0)
        story.append(Paragraph(f"{loc['name']} — Local Susceptibility", h2))
        story.append(RLImage(local_buf, width=170*mm, height=112*mm))

    story.append(Paragraph("Interpretation", h2))
    story.append(Paragraph(
        "Static susceptibility describes where terrain is more susceptible according to the "
        "production Random Forest. The dynamic trigger describes recent rainfall pressure. "
        "Operational risk is fused as R = 0.6 × S + 0.4 × T. This is a decision-support "
        "indicator and not a guaranteed landslide probability.",
        body
    ))

    story.append(Paragraph("Model Validation", h2))
    story.append(Paragraph(
        "Production model: Random Forest, 700 trees, 8 predictors, 250 m grid. "
        "Independent evaluation on 74 unseen Meghalaya locations: ROC-AUC 0.8355, "
        "PR-AUC 0.7535, Balanced Accuracy 0.7829, training samples 991.",
        body
    ))

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

    voice_text = None
    if speech_to_text is not None:
        vc1, vc2 = st.columns([1, 3])
        with vc1:
            voice_text = speech_to_text(
                language="en",
                start_prompt="🎤 Voice Search",
                stop_prompt="⏹️ Stop",
                just_once=True,
                use_container_width=True,
                key="voice_search",
            )
        with vc2:
            if voice_text:
                st.info(f"🎤 Voice recognized: {voice_text}")
            else:
                st.caption("You can type a location or use voice search.")
    else:
        st.caption("Optional voice search: install streamlit-mic-recorder.")

    search_query = voice_text.strip() if voice_text else q.strip()

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
            plt.close(local_fig)
else:
    local_fig = None


# ============================================================
# 7. EIGHT STATIC FACTORS
# ============================================================

st.markdown("---")
st.header("🌍 Eight Static Conditioning Factors")

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
        geom = {
            1: "Denudational Origin",
            2: "Fluvial Origin",
            3: "Glacial Origin",
            4: "Lacustrine Origin",
            5: "Structural Origin",
            6: "Water Bodies",
        }
        display = geom.get(round(v), str(round(v)))
    factor_rows.append({"Factor": name, "Observed model input": display})

st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)


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
# 9. FINAL FUSION EXPLANATION
# ============================================================

if fusion is not None:
    st.markdown("---")
    st.header("🧮 Risk Fusion")
    st.markdown("### R = 0.6 × S + 0.4 × T")
    st.write(
        f"S = {sus:.4f}  •  T = {trig:.4f}  •  "
        f"R = {risk_score:.4f}  •  {risk_badge(risk_level)}"
    )
    with st.expander("🧾 Show full risk calculation"):
        for item in fusion.reasoning:
            st.write("•", item)


# ============================================================
# 10. FORECAST
# ============================================================

if w is not None:
    st.markdown("---")
    st.header("📈 Next 7 Days")
    daily = w["daily"]
    forecast_df = pd.DataFrame({
        "Date": pd.to_datetime(daily["time"]).strftime("%d %b"),
        "Forecast rain (mm)": np.round(daily["precipitation_sum"], 1),
        "Rain probability (%)": np.round(daily["precipitation_probability_max"]).astype(int),
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)


# ============================================================
# 11. SEVEN-POINT LIVE MONITORING
# ============================================================

st.markdown("---")
st.header("📡 7-Point NER Dynamic Monitoring")
st.caption("Live weather-trigger snapshot for the seven project monitoring locations. Aizawl is not included because production RF susceptibility is unavailable there.")

monitor_df = live_monitoring_rows()

st.dataframe(
    monitor_df,
    use_container_width=True,
    hide_index=True,
)

if folium is not None:
    st.subheader("🗺️ Monitoring Network")
    center = (25.5, 92.0)
    mm = folium.Map(location=center, zoom_start=6, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="🛰️ Satellite",
    ).add_to(mm)

    for _, row in monitor_df.iterrows():
        label = row["Location"]
        lat, lon = MONITORING[label]
        level = str(row["Risk"])
        icon_color = {
            "LOW": "green",
            "MEDIUM": "orange",
            "HIGH": "red",
            "CRITICAL": "darkred",
            "UNAVAILABLE": "gray",
            "WEATHER UNAVAILABLE": "gray",
        }.get(level, "blue")
        popup = (
            f"<b>{label}</b><br>"
            f"S: {row['S']}<br>"
            f"T: {row['T']}<br>"
            f"R: {row['Final R']}<br>"
            f"Risk: {level}"
        )
        folium.Marker(
            [lat, lon],
            tooltip=f"{label} — {level}",
            popup=popup,
            icon=folium.Icon(color=icon_color, icon="info-sign"),
        ).add_to(mm)

    folium.LayerControl().add_to(mm)
    st_folium(mm, use_container_width=True, height=520)


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
# 13. ROUTE AWARENESS
# ============================================================

st.markdown("---")
st.header("🛣️ Landslide-Aware Route Awareness")

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
    )
    district, st_name = route_districts[route_choice]
    roads = load_route_roads()
    roads = roads[roads["district"].astype(str).str.strip().str.lower() == district.lower()].copy()

    if not roads.empty and folium is not None:
        rm = folium.Map(location=STATE_CENTERS[st_name], zoom_start=10, control_scale=True)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="🛰️ Satellite",
        ).add_to(rm)
        route_colors = {
            "AVAILABLE": "green",
            "CAUTION": "orange",
            "AVOID_IF_ALTERNATIVE": "red",
            "BLOCKED": "black",
        }
        for _, r in roads.head(5000).iterrows():
            status = str(r.get("routing_status", "AVAILABLE"))
            color = route_colors.get(status, "gray")
            folium.GeoJson(
                r.geometry.__geo_interface__,
                style_function=lambda feature, c=color: {"color": c, "weight": 2, "opacity": .7},
                tooltip=(
                    f"{r.get('road_name','Unnamed')} | {status} | "
                    f"LSI {float(r.get('LSI_mean',0)):.3f}"
                ),
            ).add_to(rm)
        folium.LayerControl().add_to(rm)
        st_folium(rm, use_container_width=True, height=560)
        st.caption("Green = available • Orange = caution • Red = avoid if alternative exists • Black = blocked.")


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

**RISK FUSION**
- `R = 0.6 × S + 0.4 × T`
- LOW / MEDIUM / HIGH / CRITICAL operational risk.

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
