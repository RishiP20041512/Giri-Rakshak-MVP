import streamlit as st
from pathlib import Path
import json
from datetime import date, timedelta
import urllib.parse
import urllib.request

import pandas as pd
import rasterio
from rasterio.warp import transform

from risk_fusion import fuse_risk
from dynamic.config import EAST_KHASI_HILLS, GENERIC_NER_DISTRICT
from dynamic.pipeline import run_dynamic_layer


# ============================================================
# GIRI-RAKSHAK — LANDSLIDE EARLY WARNING DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Giri-Rakshak",
    page_icon="⛰️",
    layout="wide",
)

ROOT = Path(__file__).parent


def initialize_gee():
    """Initialize Google Earth Engine using the user's existing credentials."""
    try:
        import ee
    except ImportError:
        return False, "earthengine-api is not installed. Run: pip install earthengine-api"

    try:
        ee.Initialize(project="land-slide-research")
        return True, "Google Earth Engine connected."
    except Exception as exc:
        return False, (
            "Google Earth Engine is not authenticated/initialized. "
            "Run `earthengine authenticate` once in the same environment, "
            "then restart Streamlit. Details: " + str(exc)
        )


def make_satellite_region(lat, lon, half_size_deg=0.01):
    import ee
    return ee.Geometry.Rectangle([
        lon - half_size_deg, lat - half_size_deg,
        lon + half_size_deg, lat + half_size_deg,
    ])

STATE_MAP_DIR = ROOT / "processed" / "state_maps"
STATE_CSV = ROOT / "processed" / "step72_statewise_susceptibility.csv"

SUSCEPTIBILITY_RASTER = (
    ROOT
    / "processed"
    / "step70_8factor_susceptibility_probability.tif"
)

STATES = [
    "Arunachal Pradesh",
    "Assam",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura",
]

MAP_FILES = {
    "Arunachal Pradesh": "Arunachal_Pradesh_susceptibility_map.png",
    "Assam": "Assam_susceptibility_map.png",
    "Manipur": "Manipur_susceptibility_map.png",
    "Meghalaya": "Meghalaya_susceptibility_map.png",
    "Mizoram": "Mizoram_susceptibility_map.png",
    "Nagaland": "Nagaland_susceptibility_map.png",
    "Sikkim": "Sikkim_susceptibility_map.png",
    "Tripura": "Tripura_susceptibility_map.png",
}

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }

    .hero {
        padding: 1.7rem 2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f3d2e, #176b4d);
        color: white;
        margin-bottom: 1.2rem;
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
        margin-top: 0;
    }

    .weather-card {
        padding: 1.15rem;
        border-radius: 16px;
        background: #f7faf8;
        border: 1px solid #dfe7e2;
        min-height: 125px;
        margin-bottom: 10px;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 14px;
        background: #fff4d6;
        border: 1px solid #e5c76b;
    }

    .factor-card {
        padding: 0.8rem;
        border-radius: 12px;
        background: #f5f7f6;
        border: 1px solid #dfe5e1;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>⛰️ Giri-Rakshak</h1>
        <p>
            Landslide Early Warning & Rainfall-Driven Risk Assessment
            for Northeast India
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STATE MAP
# ============================================================

st.sidebar.title("🗺️ Explore Northeast India")

selected_state = st.sidebar.selectbox("Select a state", STATES)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Static Model:** Random Forest

    **Conditioning factors:** 8

    **Spatial grid:** 250 m

    **Risk Fusion:** 60% Static + 40% Dynamic
    """
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Forecast weather is a separate operational trigger layer. "
    "It is not added to the already-trained RF model."
)

df = None
if STATE_CSV.exists():
    try:
        df = pd.read_csv(STATE_CSV)
    except Exception:
        df = None


# ============================================================
# HELPERS
# ============================================================

def get_real_susceptibility(lat, lon):
    if not SUSCEPTIBILITY_RASTER.exists():
        raise FileNotFoundError(
            f"Susceptibility raster not found: {SUSCEPTIBILITY_RASTER}"
        )

    with rasterio.open(SUSCEPTIBILITY_RASTER) as src:
        x, y = transform(
            "EPSG:4326",
            src.crs,
            [lon],
            [lat],
        )

        row, col = src.index(x[0], y[0])

        if (
            row < 0
            or row >= src.height
            or col < 0
            or col >= src.width
        ):
            raise ValueError("Location is outside the model raster.")

        value = float(src.read(1)[row, col])

        if src.nodata is not None and value == src.nodata:
            raise ValueError(
                "No RF susceptibility prediction exists at this location."
            )

        return value


@st.cache_data(ttl=600)
def geocode_location(query):
    """
    Search only inside India and keep only Northeast India results.

    Open-Meteo returns latitude/longitude plus admin1/admin2, so the same
    coordinates can then be used by the RF raster and weather API.
    """
    query = query.strip()
    if not query:
        return []

    # Adding India makes state-name searches such as "Nagaland" much more
    # reliable than sending the bare word globally.
    search_name = query if "," in query else f"{query}, India"

    params = {
        "name": search_name,
        "count": 20,
        "language": "en",
        "format": "json",
        "countryCode": "IN",
    }

    url = (
        "https://geocoding-api.open-meteo.com/v1/search?"
        + urllib.parse.urlencode(params)
    )

    with urllib.request.urlopen(url, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = data.get("results", [])

    # Never allow an unrelated Indian state/country into the dashboard.
    ner_results = []
    for r in results:
        if str(r.get("country_code", "")).upper() != "IN":
            continue

        admin1 = str(r.get("admin1", "")).strip()
        name = str(r.get("name", "")).strip()

        if admin1 in STATES or name in STATES:
            ner_results.append(r)

    # Exact state-name search: prefer the result whose admin1/name is the
    # requested state, then otherwise use the geocoder's relevance order.
    q_norm = query.casefold().strip()
    ner_results.sort(
        key=lambda r: (
            0 if str(r.get("name", "")).casefold() == q_norm else 1,
            0 if str(r.get("admin1", "")).casefold() == q_norm else 1,
        )
    )

    return ner_results


@st.cache_data(ttl=600)
def fetch_weather(lat, lon):
    """
    Real Open-Meteo weather.

    past_days=3 gives enough hourly history to calculate observed
    24h/72h rainfall at the selected location.
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": "auto",
        "past_days": 3,
        "forecast_days": 7,
        "current": (
            "temperature_2m,relative_humidity_2m,apparent_temperature,"
            "precipitation,rain,weather_code,wind_speed_10m,surface_pressure"
        ),
        "hourly": (
            "temperature_2m,relative_humidity_2m,precipitation,"
            "rain,precipitation_probability,weather_code,"
            "wind_speed_10m,surface_pressure"
        ),
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_sum,precipitation_probability_max"
        ),
    }

    url = (
        "https://api.open-meteo.com/v1/forecast?"
        + urllib.parse.urlencode(params)
    )

    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def weather_description(code):
    code = int(code)

    if code == 0:
        return "Clear sky"
    if code in (1, 2, 3):
        return "Partly cloudy"
    if code in (45, 48):
        return "Fog"
    if code in (51, 53, 55, 56, 57):
        return "Drizzle"
    if code in (61, 63, 65, 66, 67):
        return "Rain"
    if code in (71, 73, 75, 77):
        return "Snow"
    if code in (80, 81, 82):
        return "Rain showers"
    if code in (85, 86):
        return "Snow showers"
    if code in (95, 96, 99):
        return "Thunderstorm"

    return "Unknown"


def risk_badge(level):
    level = str(level).upper()

    if level == "CRITICAL":
        return "🔴 CRITICAL"
    if level == "HIGH":
        return "🟠 HIGH"
    if level == "MEDIUM":
        return "🟡 MEDIUM"
    return "🟢 LOW"


def rainfall_history(hourly):
    times = pd.Series(pd.to_datetime(hourly["time"]))
    rain = pd.Series(hourly["precipitation"], dtype="float64").fillna(0.0)

    now_time = times.iloc[-1]

    last_24 = rain.iloc[max(0, len(rain) - 24):].sum()
    last_72 = rain.iloc[max(0, len(rain) - 72):].sum()

    return float(last_24), float(last_72), times, rain


def forecast_trigger(cum_24h, cum_72h, cfg):
    ratio24 = cum_24h / max(float(cfg.threshold_24h_mm), 1e-9)
    ratio72 = cum_72h / max(float(cfg.threshold_72h_mm), 1e-9)

    exceedance = max(ratio24, ratio72)

    lo, hi = cfg.id_norm_range
    score = (exceedance - lo) / max(hi - lo, 1e-9)
    score = max(0.0, min(1.0, score))

    return score, ratio24, ratio72


def hourly_forecast_table(payload, susceptibility, cfg):
    hourly = payload["hourly"]

    times = pd.Series(pd.to_datetime(hourly["time"]))
    rain = pd.Series(hourly["precipitation"], dtype="float64").fillna(0.0)
    probability = pd.Series(
        hourly["precipitation_probability"],
        dtype="float64",
    ).fillna(0.0)

    # Only future hours.
    now = pd.Timestamp.now(tz=None)
    future_mask = times >= now

    future_indices = list(times[future_mask].index)

    rows = []

    for i in future_indices[:7 * 24:3]:
        t = times.iloc[i]

        start24 = max(0, i - 23)
        start72 = max(0, i - 71)

        cum24 = float(rain.iloc[start24:i + 1].sum())
        cum72 = float(rain.iloc[start72:i + 1].sum())

        trigger, ratio24, ratio72 = forecast_trigger(
            cum24,
            cum72,
            cfg,
        )

        projected = fuse_risk(
            susceptibility_score=susceptibility,
            trigger_score=trigger,
        )

        rows.append(
            {
                "Time": t.strftime("%a %d %b, %I %p"),
                "Rain (mm)": round(float(rain.iloc[i]), 1),
                "Rain probability": f"{int(round(probability.iloc[i]))}%",
                "24h accumulation (mm)": round(cum24, 1),
                "72h accumulation (mm)": round(cum72, 1),
                "Trigger": round(trigger, 3),
                "Projected risk": projected.risk_level,
            }
        )

    return pd.DataFrame(rows)


def daily_forecast_table(payload, susceptibility, cfg):
    daily = payload["daily"]

    dates = pd.Series(pd.to_datetime(daily["time"]))
    rain = pd.Series(
        daily["precipitation_sum"],
        dtype="float64",
    ).fillna(0.0)
    probability = pd.Series(
        daily["precipitation_probability_max"],
        dtype="float64",
    ).fillna(0.0)

    rows = []

    for i in range(len(dates)):
        daily_rain = float(rain.iloc[i])

        # For the daily outlook, use rolling 3-day forecast accumulation.
        rolling72 = float(
            rain.iloc[max(0, i - 2):i + 1].sum()
        )

        trigger, ratio24, ratio72 = forecast_trigger(
            daily_rain,
            rolling72,
            cfg,
        )

        projected = fuse_risk(
            susceptibility_score=susceptibility,
            trigger_score=trigger,
        )

        rows.append(
            {
                "Date": dates.iloc[i].strftime("%a %d %b"),
                "Forecast rain (mm)": round(daily_rain, 1),
                "Rain probability": f"{int(round(probability.iloc[i]))}%",
                "24h threshold used": f"{ratio24 * 100:.0f}%",
                "72h threshold used": f"{ratio72 * 100:.0f}%",
                "Trigger": round(trigger, 3),
                "Projected risk": projected.risk_level,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STATE SUSCEPTIBILITY MAP
# ============================================================

st.subheader(f"📍 {selected_state}")

map_path = STATE_MAP_DIR / MAP_FILES[selected_state]

if selected_state == "Mizoram":
    st.markdown(
        """
        <div class="warning-box">
            <b>⚠️ RF prediction unavailable for Mizoram</b><br><br>
            The current production susceptibility model has no valid
            prediction coverage here because the required geomorphological
            predictor coverage is incomplete.
            <br><br>
            No synthetic susceptibility values have been inserted.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if map_path.exists():
        st.image(
            str(map_path),
            caption="Mizoram — predictor coverage limitation",
            use_container_width=True,
        )

else:
    left, right = st.columns([2.2, 1])

    with left:
        if map_path.exists():
            st.image(
                str(map_path),
                caption=f"{selected_state} — Landslide Susceptibility",
                use_container_width=True,
            )
        else:
            st.error("State map not found.")

    with right:
        st.markdown("### 📊 State Susceptibility")

        state_row = None

        if df is not None:
            for _, row in df.iterrows():
                state_value = str(row.iloc[0])

                clean = (
                    state_value
                    .replace("ā", "a")
                    .replace("ī", "i")
                    .replace("ṅ", "n")
                    .replace("ē", "e")
                    .replace("ḷ", "l")
                )

                if selected_state.lower() in clean.lower():
                    state_row = row
                    break

        if state_row is not None:
            preferred = {
                "Very Low": None,
                "Low": None,
                "Moderate": None,
                "High": None,
                "Very High": None,
            }

            for col in df.columns:
                try:
                    value = float(state_row[col])
                except Exception:
                    continue

                if not 0 <= value <= 100:
                    continue

                c = col.lower().replace("_", " ")

                if "very low" in c:
                    preferred["Very Low"] = value
                elif c.strip() == "low":
                    preferred["Low"] = value
                elif "moderate" in c:
                    preferred["Moderate"] = value
                elif c.strip() == "high":
                    preferred["High"] = value
                elif "very high" in c:
                    preferred["Very High"] = value

            for label, value in preferred.items():
                if value is not None:
                    st.metric(label, f"{value:.2f}%")


# ============================================================
# LOCATION SEARCH
# ============================================================

st.markdown("---")
st.header("🔎 Search Any Northeast India Location")

st.write(
    "Search a city, district, town, village, or landmark. "
    "Weather and forecast information are fetched for the returned "
    "coordinates, while susceptibility comes from the real RF raster "
    "where prediction coverage exists."
)

search_col, button_col = st.columns([4, 1])

with search_col:
    location_query = st.text_input(
        "Location",
        value="Sohra, Meghalaya",
        placeholder="Example: Aizawl, Mizoram",
        label_visibility="collapsed",
    )

with button_col:
    search_clicked = st.button(
        "🔍 Search",
        type="primary",
        use_container_width=True,
    )

if "selected_location" not in st.session_state:
    st.session_state.selected_location = {
        "name": "Sohra, Meghalaya",
        "lat": EAST_KHASI_HILLS.station_lat,
        "lon": EAST_KHASI_HILLS.station_lon,
        "country": "India",
        "admin1": "Meghalaya",
    }

if search_clicked and location_query.strip():

    try:
        results = geocode_location(location_query.strip())

        if not results:
            st.session_state.pop("geocode_results", None)
            st.error(
                "No Northeast India location found. "
                "Try a city, district, town, village, or state name."
            )
        else:
            st.session_state.geocode_results = results

            # IMPORTANT FIX: Search immediately changes the selected
            # location to the first valid Northeast India result.
            # The previous version only changed location after a second
            # button click, which made the UI appear stuck on Guwahati/Sohra.
            first = results[0]
            st.session_state.selected_location = {
                "name": ", ".join(
                    [
                        str(x)
                        for x in [
                            first.get("name"),
                            first.get("admin2"),
                            first.get("admin1"),
                            first.get("country"),
                        ]
                        if x
                    ]
                ),
                "lat": float(first["latitude"]),
                "lon": float(first["longitude"]),
                "country": first.get("country", "India"),
                "admin1": first.get("admin1", ""),
                "admin2": first.get("admin2", ""),
            }

    except Exception as e:
        st.error(f"Location search failed: {e}")


if "geocode_results" in st.session_state:

    results = st.session_state.geocode_results

    labels = []
    for r in results:
        label = ", ".join(
            [
                str(x)
                for x in [
                    r.get("name"),
                    r.get("admin2"),
                    r.get("admin1"),
                    r.get("country"),
                ]
                if x
            ]
        )
        labels.append(label)

    st.success(
        "Search result selected automatically. "
        "Choose another match below only if needed."
    )

    selected_label = st.selectbox(
        "Choose another matching location",
        labels,
    )

    selected_result = results[labels.index(selected_label)]

    if st.button("📍 Switch to Selected Result"):
        st.session_state.selected_location = {
            "name": selected_label,
            "lat": float(selected_result["latitude"]),
            "lon": float(selected_result["longitude"]),
            "country": selected_result.get("country", ""),
            "admin1": selected_result.get("admin1", ""),
            "admin2": selected_result.get("admin2", ""),
        }
        st.rerun()


location = st.session_state.selected_location

st.success(
    f"📍 Monitoring: **{location['name']}**  |  "
    f"{location['lat']:.4f}, {location['lon']:.4f}"
)


# ============================================================
# WEATHER FOR SELECTED LOCATION
# ============================================================

st.markdown("---")
st.header("🌦️ Landslide Weather Center")

try:
    with st.spinner("Fetching real current weather + 7-day forecast..."):
        weather = fetch_weather(
            location["lat"],
            location["lon"],
        )
except Exception as e:
    weather = None
    st.error(f"Weather data could not be fetched: {e}")


if weather is not None:

    current = weather["current"]
    hourly = weather["hourly"]

    # --------------------------------------------------------
    # CURRENT WEATHER
    # --------------------------------------------------------

    st.subheader("☀️ Current Conditions")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Temperature",
            f"{current['temperature_2m']:.1f} °C",
        )

    with c2:
        st.metric(
            "Rain now",
            f"{current['rain']:.1f} mm",
        )

    with c3:
        st.metric(
            "Humidity",
            f"{current['relative_humidity_2m']:.0f}%",
        )

    with c4:
        st.metric(
            "Pressure",
            f"{current['surface_pressure']:.0f} hPa",
        )

    c5, c6, c7 = st.columns(3)

    with c5:
        st.metric(
            "Wind",
            f"{current['wind_speed_10m']:.1f} km/h",
        )

    with c6:
        st.metric(
            "Feels like",
            f"{current['apparent_temperature']:.1f} °C",
        )

    with c7:
        st.metric(
            "Condition",
            weather_description(current["weather_code"]),
        )

    # --------------------------------------------------------
    # OBSERVED 24/72 HOUR RAINFALL
    # --------------------------------------------------------

    observed24, observed72, times, rain_series = rainfall_history(hourly)

    st.subheader("🌧️ Rainfall Loading")

    r1, r2, r3 = st.columns(3)

    with r1:
        st.metric("Observed / recent 24h", f"{observed24:.1f} mm")

    with r2:
        st.metric("Observed / recent 72h", f"{observed72:.1f} mm")

    with r3:
        st.metric(
            "Current hourly rain",
            f"{current['rain']:.1f} mm",
        )

    # --------------------------------------------------------
    # STATIC SUSCEPTIBILITY
    # --------------------------------------------------------

    try:
        susceptibility = get_real_susceptibility(
            location["lat"],
            location["lon"],
        )

        st.metric(
            "Real RF susceptibility",
            f"{susceptibility:.3f}",
        )

    except Exception as e:
        susceptibility = None

        st.warning(
            f"RF susceptibility is unavailable at this location. {e}"
        )

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    is_pilot = (
        abs(location["lat"] - EAST_KHASI_HILLS.station_lat) < 0.15
        and abs(location["lon"] - EAST_KHASI_HILLS.station_lon) < 0.15
    )

    cfg = EAST_KHASI_HILLS if is_pilot else GENERIC_NER_DISTRICT

    if is_pilot:
        st.success(
            "This location is inside the East Khasi Hills pilot area. "
            "Pilot rainfall thresholds are available."
        )
    else:
        st.info(
            "This location is outside the East Khasi Hills pilot. "
            "The dashboard can still evaluate real weather and forecast "
            "rainfall, but the generic NER rainfall thresholds are "
            "prototype/un-calibrated and must not be presented as "
            "state-wide validated thresholds."
        )

    # --------------------------------------------------------
    # CURRENT OPERATIONAL TRIGGER
    # --------------------------------------------------------

    current_trigger, ratio24, ratio72 = forecast_trigger(
        observed24,
        observed72,
        cfg,
    )

    st.subheader("🎯 Current Rainfall Trigger")

    t1, t2, t3 = st.columns(3)

    with t1:
        st.metric(
            "24h threshold",
            f"{cfg.threshold_24h_mm:.0f} mm",
        )

    with t2:
        st.metric(
            "72h threshold",
            f"{cfg.threshold_72h_mm:.0f} mm",
        )

    with t3:
        st.metric(
            "Rainfall trigger",
            f"{current_trigger:.3f}",
        )

    # --------------------------------------------------------
    # CURRENT RISK
    # --------------------------------------------------------

    if susceptibility is not None:

        current_fusion = fuse_risk(
            susceptibility_score=susceptibility,
            trigger_score=current_trigger,
        )

        st.subheader("🚨 Current Landslide Risk")

        f1, f2, f3 = st.columns(3)

        with f1:
            st.metric(
                "Static susceptibility",
                f"{susceptibility:.3f}",
            )

        with f2:
            st.metric(
                "Dynamic trigger",
                f"{current_trigger:.3f}",
            )

        with f3:
            st.metric(
                "Risk score",
                f"{current_fusion.risk_score:.3f}",
            )

        st.markdown(
            f"### {risk_badge(current_fusion.risk_level)}"
        )

        with st.expander("🧾 Risk calculation"):
            for item in current_fusion.reasoning:
                st.write(f"• {item}")

        if not is_pilot:
            st.caption(
                "For non-pilot locations, this current risk is a prototype "
                "operational assessment because the generic rainfall "
                "thresholds are not locally calibrated."
            )

    # ========================================================
    # HOURLY FORECAST
    # ========================================================

    st.markdown("---")
    st.header("🌧️ Hourly Landslide Trigger Forecast")

    hourly_df = hourly_forecast_table(
        weather,
        susceptibility if susceptibility is not None else 0.0,
        cfg,
    )

    if not hourly_df.empty:

        chart_df = hourly_df.copy()
        chart_df["Time"] = pd.to_datetime(
            chart_df["Time"],
            format="%a %d %b, %I %p",
        )
        chart_df = chart_df.set_index("Time")

        st.subheader("📈 Forecast Rainfall Accumulation")

        st.line_chart(
            chart_df[
                [
                    "Rain (mm)",
                    "24h accumulation (mm)",
                    "72h accumulation (mm)",
                ]
            ],
            use_container_width=True,
        )

        st.subheader("🎯 Forecast Landslide Trigger")

        st.line_chart(
            chart_df[["Trigger"]],
            use_container_width=True,
        )

        st.dataframe(
            hourly_df,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # 7-DAY OUTLOOK
    # ========================================================

    st.markdown("---")
    st.header("📅 7-Day Landslide Outlook")

    daily_df = daily_forecast_table(
        weather,
        susceptibility if susceptibility is not None else 0.0,
        cfg,
    )

    st.dataframe(
        daily_df,
        use_container_width=True,
        hide_index=True,
    )

    if not daily_df.empty:
        max_idx = daily_df["Forecast rain (mm)"].idxmax()
        max_day = daily_df.loc[max_idx]

        st.info(
            f"🌧️ Highest forecast rainfall: "
            f"**{max_day['Forecast rain (mm)']:.1f} mm** on "
            f"**{max_day['Date']}**, with rain probability "
            f"**{max_day['Rain probability']}**."
        )

    # ========================================================
    # LANDSLIDE INTERPRETATION
    # ========================================================

    st.markdown("---")
    st.header("🤖 Landslide Rainfall Assessment")

    if susceptibility is not None:

        future_max_trigger = (
            float(daily_df["Trigger"].max())
            if not daily_df.empty
            else current_trigger
        )

        future_risk = fuse_risk(
            susceptibility_score=susceptibility,
            trigger_score=future_max_trigger,
        )

        if future_max_trigger > current_trigger:
            direction = "increasing"
        elif future_max_trigger < current_trigger:
            direction = "decreasing"
        else:
            direction = "remaining approximately stable"

        st.write(
            f"""
            **Current terrain susceptibility:** {susceptibility:.3f}

            **Current rainfall trigger:** {current_trigger:.3f}

            **Highest projected trigger in the forecast:** {future_max_trigger:.3f}

            Based on forecast rainfall accumulation, the trigger is
            expected to be **{direction}** over the forecast period.

            The corresponding highest projected fusion category is
            **{future_risk.risk_level}**.
            """
        )

        if future_max_trigger >= 0.75:
            st.error(
                "🔴 Forecast rainfall loading reaches a very high trigger "
                "range. This warrants close monitoring, especially where "
                "terrain susceptibility is high."
            )
        elif future_max_trigger >= 0.50:
            st.warning(
                "🟠 Forecast rainfall loading reaches a high trigger range. "
                "Monitor accumulation and local conditions closely."
            )
        elif future_max_trigger >= 0.25:
            st.warning(
                "🟡 Forecast rainfall produces a moderate trigger. "
                "Continue monitoring if rainfall intensifies."
            )
        else:
            st.success(
                "🟢 Forecast rainfall remains below the higher trigger "
                "ranges at this location."
            )

    st.caption(
        "This is an early-warning decision-support assessment, not a "
        "calibrated probability that a landslide will occur."
    )

    # ========================================================
    # MONTHLY CONTEXT
    # ========================================================

    st.markdown("---")
    st.header("📆 Monthly / Seasonal Landslide Context")

    st.write(
        """
        Monthly information should describe rainfall seasonality and
        historical context rather than pretending that an exact 30-day
        landslide forecast is available.
        """
    )

    monthly_note = pd.DataFrame(
        {
            "Time horizon": [
                "Now",
                "Next hours",
                "Next 7 days",
                "Monthly / seasonal",
            ],
            "Giri-Rakshak output": [
                "Observed weather + rainfall loading",
                "Hourly rainfall + trigger forecast",
                "Daily rainfall + projected trigger/risk",
                "Historical / climatological context",
            ],
            "Purpose": [
                "Current warning state",
                "Near-term escalation",
                "Early warning planning",
                "Seasonal preparedness",
            ],
        }
    )

    st.dataframe(
        monthly_note,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# SATELLITE MONITORING — REAL GEE CHANGE DETECTION
# ============================================================

st.markdown("---")
st.header("🛰️ Satellite Monitoring")
st.write(
    "Real Sentinel-2 NDVI and Sentinel-1 SAR change detection for the "
    "currently selected Northeast India location. The disturbance math is "
    "kept inside gee_satellite.py / satellite_change_detection.py."
)

with st.expander("🛰️ Run real satellite disturbance check", expanded=False):
    st.caption(
        "This checks a small village-scale area around the searched location. "
        "Optical Sentinel-2 can be unavailable during heavy cloud; Sentinel-1 "
        "SAR is used when available."
    )

    today = date.today()
    default_before_start = today - timedelta(days=60)
    default_before_end = today - timedelta(days=31)
    default_after_start = today - timedelta(days=30)
    default_after_end = today

    d1, d2 = st.columns(2)
    with d1:
        before_start = st.date_input(
            "Before — start", default_before_start, key="sat_before_start"
        )
        before_end = st.date_input(
            "Before — end", default_before_end, key="sat_before_end"
        )
    with d2:
        after_start = st.date_input(
            "After — start", default_after_start, key="sat_after_start"
        )
        after_end = st.date_input(
            "After — end", default_after_end, key="sat_after_end"
        )

    run_satellite = st.button(
        "🔬 Run Real Satellite Check",
        type="primary",
        use_container_width=True,
    )

    if run_satellite:
        if not (before_start < before_end and after_start < after_end):
            st.error("Each satellite date window must have start < end.")
        else:
            gee_ok, gee_message = initialize_gee()
            if not gee_ok:
                st.warning(gee_message)
            else:
                try:
                    try:
                        from dynamic.gee_satellite import fetch_disturbance_flag_for_village
                    except ImportError:
                        from gee_satellite import fetch_disturbance_flag_for_village

                    region = make_satellite_region(
                        location["lat"], location["lon"], half_size_deg=0.01
                    )

                    sat_is_pilot = (
                        abs(location["lat"] - EAST_KHASI_HILLS.station_lat) < 0.15
                        and abs(location["lon"] - EAST_KHASI_HILLS.station_lon) < 0.15
                    )
                    sat_cfg = EAST_KHASI_HILLS if sat_is_pilot else GENERIC_NER_DISTRICT

                    with st.spinner(
                        "Fetching real Sentinel-2 + Sentinel-1 data from Google Earth Engine..."
                    ):
                        satellite_result = fetch_disturbance_flag_for_village(
                            region,
                            (before_start.isoformat(), before_end.isoformat()),
                            (after_start.isoformat(), after_end.isoformat()),
                            sat_cfg,
                            max_cloud=50,
                        )

                    if satellite_result.get("status") == "ok":
                        sources = satellite_result.get("sources_used", [])
                        flagged = int(satellite_result.get("flagged_pixels", 0))
                        detected = bool(satellite_result.get("disturbance_detected", False))

                        st.success(
                            "Satellite check completed using: "
                            + (", ".join(sources) if sources else "no source")
                        )

                        s1, s2, s3 = st.columns(3)
                        with s1:
                            st.metric("Disturbance detected", "YES" if detected else "NO")
                        with s2:
                            st.metric("Flagged pixels", f"{flagged:,}")
                        with s3:
                            st.metric("Sources used", str(len(sources)))

                        if detected:
                            st.error(
                                "🔴 Satellite disturbance signal detected in the "
                                "selected monitoring window. Treat this as corroborating "
                                "evidence, not proof of an active landslide."
                            )
                        else:
                            st.success(
                                "🟢 No pixels crossed the configured disturbance threshold "
                                "in this monitoring window."
                            )

                        with st.expander("Satellite audit details"):
                            st.json(satellite_result)

                    else:
                        st.info(
                            "No usable satellite scene was available for the selected "
                            "before/after windows. This is a real data-availability result, "
                            "not a synthetic fallback."
                        )
                        st.json(satellite_result)

                except Exception as exc:
                    st.error(f"Satellite monitoring failed: {exc}")


# ============================================================
# MODEL VALIDATION
# ============================================================

st.markdown("---")
st.subheader("🎯 Model Validation")

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Independent ROC-AUC", "0.8355")

with m2:
    st.metric("PR-AUC", "0.7535")

with m3:
    st.metric("Balanced Accuracy", "0.7829")

with m4:
    st.metric("Training Samples", "991")

st.caption("Independent evaluation on 74 unseen Meghalaya locations.")


# ============================================================
# MODEL DETAILS
# ============================================================

st.markdown("---")
st.subheader("🤖 Model Configuration")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Algorithm**\n\nRandom Forest\n\n**Trees**\n\n700")

with c2:
    st.markdown("**Spatial resolution**\n\n250 m\n\n**Features**\n\n8")

with c3:
    st.markdown(
        "**Minimum leaf size**\n\n5\n\n"
        "**Evaluation**\n\nIndependent spatial test"
    )


# ============================================================
# FACTORS
# ============================================================

st.markdown("---")
st.subheader("🌍 Eight Conditioning Factors")

factors = [
    ("⛰️", "Elevation"),
    ("📐", "Slope"),
    ("🌧️", "3-Day Rainfall"),
    ("💧", "Soil Moisture"),
    ("🌿", "NDVI"),
    ("🛣️", "Distance to Road"),
    ("〰️", "Lineament Density"),
    ("🪨", "Geomorphological Origin"),
]

factor_cols = st.columns(4)

for i, (icon, factor) in enumerate(factors):
    with factor_cols[i % 4]:
        st.markdown(
            f"""
            <div class="factor-card">
                {icon} <b>{factor}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

st.markdown("---")
st.subheader("📖 How to Read Giri-Rakshak")

st.write(
    """
    **Susceptibility** describes how vulnerable the terrain is according
    to the trained RF model.

    **Rainfall trigger** describes how strongly observed or forecast
    rainfall accumulation is approaching the configured trigger.

    **Risk fusion** combines those two signals.

    Temperature, humidity, wind and pressure are displayed as weather
    context. They are not silently added as new RF predictors.

    A rainfall forecast does not guarantee a landslide. A high
    susceptibility value also does not mean that a landslide is currently
    happening.
    """
)


# ============================================================
# DATA INTEGRITY
# ============================================================

st.markdown("---")
st.subheader("🔬 Data Integrity")

st.success(
    """
    The susceptibility layer uses the project's real RF model outputs.
    Weather and forecast values are fetched from the live weather service.
    No synthetic susceptibility or synthetic weather values are inserted.

    East Khasi Hills has the project's pilot rainfall configuration.
    Other locations use the explicitly labelled generic prototype
    threshold configuration until local calibration data are available.
    """
)

st.caption(
    "Giri-Rakshak | Northeast India | 8-factor Random Forest | "
    "250 m model grid | Real Weather + Rainfall Trigger + Risk Fusion"
)
