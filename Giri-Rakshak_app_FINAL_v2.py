import geopandas as gpd
import streamlit as st
from pathlib import Path
import json
import urllib.parse
import urllib.request
from datetime import date, timedelta

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

from shapely.geometry import Point

try:
    import folium
    from streamlit_folium import st_folium
except Exception:
    folium = None
    st_folium = None

from risk_fusion import fuse_risk
from dynamic.config import EAST_KHASI_HILLS, GENERIC_NER_DISTRICT


# ============================================================
# GIRI-RAKSHAK — FINAL JUDGE DASHBOARD
# ============================================================

st.set_page_config(
    page_title="Giri-Rakshak",
    page_icon="⛰️",
    layout="wide",
)

ROOT = Path(__file__).parent

# ------------------------------------------------------------
# PROJECT FILES
# ------------------------------------------------------------
SUS_RASTER = ROOT / "processed" / "step70_8factor_susceptibility_probability.tif"
STATE_CSV = ROOT / "processed" / "step72_statewise_susceptibility.csv"
STATE_MAP_DIR = ROOT / "processed" / "state_maps"
NER_BOUNDARY = ROOT / "raw_data" / "boundaries" / "ner_8states.geojson"

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

# Existing road-risk package. It currently covers the Assam + Meghalaya
# 10-district pilot. West Tripura needs a separate OSM extraction before
# it can honestly be shown as a route-network pilot.
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

# Validated pilot snapshot values from the project's fusion validation run.
# These are shown only as a historical/validation reference, not as live data.
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
    padding: 1.7rem 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg,#0b3d2e,#147a55);
    color:white;
    margin-bottom:1.2rem;
}
.hero h1 {font-size:42px;margin-bottom:4px;}
.hero p {font-size:18px;margin:0;}
.card {
    padding:1rem;
    border-radius:15px;
    background:#f7faf8;
    border:1px solid #dfe7e2;
    margin-bottom:10px;
}
.small {color:#64736b;font-size:13px;}
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

def clean_state_name(x):
    s = str(x)
    for a,b in [("ā","a"),("ī","i"),("ṅ","n"),("ē","e"),("ḷ","l")]:
        s = s.replace(a,b)
    return s


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
        "LOW":"🟢 LOW",
        "MEDIUM":"🟡 MEDIUM",
        "HIGH":"🟠 HIGH",
        "CRITICAL":"🔴 CRITICAL",
    }.get(str(level).upper(), str(level))


def value_at_raster(path, lat, lon):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        x,y = transform("EPSG:4326", src.crs, [lon], [lat])
        row,col = src.index(x[0],y[0])
        if row < 0 or row >= src.height or col < 0 or col >= src.width:
            return None
        v = float(src.read(1)[row,col])
        if src.nodata is not None and np.isclose(v, src.nodata):
            return None
        return v


def read_sus_crop(lat, lon, radius_km=12):
    if not SUS_RASTER.exists():
        return None
    with rasterio.open(SUS_RASTER) as src:
        x,y = transform("EPSG:4326", src.crs, [lon],[lat])
        cx,cy = x[0],y[0]
        # approximate radius in metres in the equal-area model CRS
        r = radius_km * 1000
        win = from_bounds(cx-r, cy-r, cx+r, cy+r, src.transform)
        win = win.round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0,0,src.width,src.height))
        arr = src.read(1, window=win)
        tr = src.window_transform(win)
        arr = np.ma.masked_where((arr == src.nodata) if src.nodata is not None else ~np.isfinite(arr), arr)
        return arr, tr


def make_susceptibility_plot(lat, lon, name):
    result = read_sus_crop(lat, lon, 12)

    if result is None:
        return None

    arr, tr = result

    # ========================================================
    # FRIEND-STYLE LANDSLIDE SUSCEPTIBILITY COLOURS
    # ========================================================
    #
    # IMPORTANT:
    # These colours ONLY change the appearance.
    # The RF values and 250 m grid are NOT changed.
    #
    # 0.00 - 0.25  = Low        -> Green
    # 0.25 - 0.50  = Moderate   -> Yellow
    # 0.50 - 0.75  = High       -> Orange
    # 0.75 - 1.00  = Very High  -> Red
    #
    # ========================================================

    from matplotlib.colors import ListedColormap, BoundaryNorm

    colors = [
        "#2ca25f",   # Green  - Low
        "#fdd049",   # Yellow - Moderate
        "#f28e2b",   # Orange - High
        "#d73027",   # Red    - Very High
    ]

    cmap = ListedColormap(colors)

    bounds = [0.00, 0.25, 0.50, 0.75, 1.00]

    norm = BoundaryNorm(
        bounds,
        cmap.N
    )

    # ========================================================
    # CREATE MAP
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    im = ax.imshow(
        arr,
        extent=(
            tr.c,
            tr.c + tr.a * arr.shape[1],
            tr.f + tr.e * arr.shape[0],
            tr.f
        ),
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    # ========================================================
    # SEARCHED LOCATION
    # ========================================================

    with rasterio.open(SUS_RASTER) as src:

        x, y = transform(
            "EPSG:4326",
            src.crs,
            [lon],
            [lat]
        )

    ax.scatter(
        x[0],
        y[0],
        s=100,
        marker="*",
        facecolor="white",
        edgecolor="black",
        linewidth=1.5,
        zorder=10
    )

    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        f"{name} — Landslide Susceptibility",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    # ========================================================
    # FRIEND-STYLE LEGEND
    # ========================================================

    from matplotlib.patches import Patch

    legend_elements = [
        Patch(
            facecolor="#2ca25f",
            label="Low (0.00–0.25)"
        ),
        Patch(
            facecolor="#fdd049",
            label="Moderate (0.25–0.50)"
        ),
        Patch(
            facecolor="#f28e2b",
            label="High (0.50–0.75)"
        ),
        Patch(
            facecolor="#d73027",
            label="Very High (0.75–1.00)"
        ),
    ]

    ax.legend(
        handles=legend_elements,
        title="Susceptibility Class",
        loc="lower left",
        frameon=True
    )

    ax.grid(
        alpha=0.15
    )

    fig.tight_layout()

    return fig

# ============================================================
# NER-ONLY LOCATION SEARCH
# ============================================================

NER_STATE_ALIASES = {
    "arunachal pradesh": "Arunachal Pradesh",
    "arunachal": "Arunachal Pradesh",

    "assam": "Assam",

    "manipur": "Manipur",

    "meghalaya": "Meghalaya",

    "mizoram": "Mizoram",

    "nagaland": "Nagaland",

    "sikkim": "Sikkim",

    "tripura": "Tripura",
}


@st.cache_resource
def load_ner_boundary():
    """
    Load the real 8-state Northeast India boundary.
    """
    if gpd is None:
        raise RuntimeError(
            "GeoPandas is required for NER boundary validation."
        )

    if not NER_BOUNDARY.exists():
        raise FileNotFoundError(
            f"NER boundary not found: {NER_BOUNDARY}"
        )

    gdf = gpd.read_file(NER_BOUNDARY)

    if gdf.empty:
        raise RuntimeError("NER boundary file is empty.")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")

    return gdf


def is_inside_ner(lat, lon):
    """
    Strict geographic validation.

    A result is accepted ONLY when its coordinates
    fall inside the real 8-state NER boundary.
    """

    try:
        gdf = load_ner_boundary()

        point = Point(float(lon), float(lat))

        return bool(
            gdf.geometry.covers(point).any()
        )

    except Exception:
        return False


def normalize_state_name(name):
    """
    Normalize state names returned by the geocoder.
    """

    if not name:
        return ""

    s = str(name).strip().casefold()

    replacements = {
        "ā": "a",
        "ī": "i",
        "ṅ": "n",
        "ē": "e",
        "ḷ": "l",
    }

    for a, b in replacements.items():
        s = s.replace(a, b)

    return s


@st.cache_data(ttl=600)
def geocode(q):
    """
    NER-ONLY geocoder.

    Searches India using Nominatim and then applies
    TWO safety checks:

    1. Country must be India.
    2. Coordinates must fall inside the real NER boundary.

    Therefore places such as London, Gaza, Mumbai,
    Delhi, etc. cannot become selected locations.
    """

    query = str(q).strip()

    if not query:
        return []

    # --------------------------------------------------------
    # NER geographic bounding box
    # --------------------------------------------------------

    # west, north, east, south
    viewbox = "88.0,29.6,97.6,21.8"

    search_query = query

    if "india" not in search_query.casefold():
        search_query = f"{search_query}, India"

    params = {
        "q": search_query,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 20,
        "countrycodes": "in",
        "viewbox": viewbox,
        "bounded": 1,
    }

    url = (
        "https://nominatim.openstreetmap.org/search?"
        + urllib.parse.urlencode(params)
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Giri-Rakshak/1.0"
        },
    )

    with urllib.request.urlopen(request, timeout=20) as response:
        data = json.loads(
            response.read().decode("utf-8")
        )

    valid_results = []

    for item in data:

        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
        except Exception:
            continue

        # ----------------------------------------------------
        # CHECK 1 — INDIA
        # ----------------------------------------------------

        address = item.get("address", {})

        country_code = str(
            address.get("country_code", "")
        ).casefold()

        if country_code != "in":
            continue

        # ----------------------------------------------------
        # CHECK 2 — ACTUAL NER GEOMETRY
        # ----------------------------------------------------

        if not is_inside_ner(lat, lon):
            continue

        # ----------------------------------------------------
        # STATE
        # ----------------------------------------------------

        state = (
            address.get("state")
            or address.get("state_district")
            or ""
        )

        state_clean = normalize_state_name(state)

        matched_state = None

        for alias, canonical in NER_STATE_ALIASES.items():

            if alias in state_clean:
                matched_state = canonical
                break

        # ----------------------------------------------------
        # If geocoder does not provide state name,
        # still accept the result because the coordinates
        # already passed the real NER polygon test.
        # ----------------------------------------------------

        if matched_state is None:
            matched_state = state

        district = (
            address.get("state_district")
            or address.get("district")
            or address.get("county")
            or ""
        )

        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("suburb")
            or item.get("name")
            or query
        )

        display_name = ", ".join(
            [
                str(x)
                for x in [
                    city,
                    district,
                    matched_state,
                    "India",
                ]
                if x
            ]
        )

        valid_results.append(
            {
                "name": city,
                "display_name": display_name,
                "latitude": lat,
                "longitude": lon,
                "country": "India",
                "country_code": "IN",
                "admin1": matched_state,
                "admin2": district,
                "type": item.get("type", ""),
                "osm_type": item.get("osm_type", ""),
            }
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    unique = {}

    for r in valid_results:

        key = (
            round(r["latitude"], 5),
            round(r["longitude"], 5),
        )

        if key not in unique:
            unique[key] = r

    results = list(unique.values())

    # --------------------------------------------------------
    # BETTER ORDERING
    # --------------------------------------------------------

    q_norm = query.casefold()

    results.sort(
        key=lambda r: (
            0
            if str(r["name"]).casefold() == q_norm
            else 1,

            0
            if str(r["admin1"]).casefold() == q_norm
            else 1,
        )
    )

    return results


@st.cache_data(ttl=600)
def weather(lat,lon):
    params={
        "latitude":lat,"longitude":lon,"timezone":"auto",
        "past_days":3,"forecast_days":7,
        "current":"temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,surface_pressure",
        "hourly":"temperature_2m,relative_humidity_2m,precipitation,rain,precipitation_probability,weather_code,wind_speed_10m,surface_pressure",
        "daily":"weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max"
    }
    url="https://api.open-meteo.com/v1/forecast?"+urllib.parse.urlencode(params)
    with urllib.request.urlopen(url,timeout=20) as r:
        return json.loads(r.read().decode())


def rain_history(payload):
    """
    Calculate rainfall from the hours BEFORE the current time.
    This avoids accidentally using forecast rainfall as 'recent rainfall'.
    """

    h = payload["hourly"]

    times = pd.to_datetime(h["time"])
    rain = pd.Series(h["precipitation"], dtype=float).fillna(0)

    current_time = pd.to_datetime(
        payload["current"]["time"]
    )

    past_mask = times <= current_time

    past_rain = rain[past_mask]

    if len(past_rain) == 0:
        return 0.0, 0.0

    r24 = float(past_rain.iloc[-24:].sum())
    r72 = float(past_rain.iloc[-72:].sum())

    return r24, r72


def dynamic_trigger(r24,r72,threshold24=150,threshold72=300):
    ratio24=r24/max(threshold24,1e-9)
    ratio72=r72/max(threshold72,1e-9)
    exceed=max(ratio24,ratio72)
    # Same MVP normalization concept as the existing trigger layer:
    # 0 -> 0 and 3x threshold -> 1.
    return max(0,min(1,exceed/3)),ratio24,ratio72


# ============================================================
# REAL SATELLITE MAP — SEARCHED NER LOCATION ONLY
# ============================================================

def satellite_map(lat, lon, name, zoom=13):

    if folium is None:
        return None

    # --------------------------------------------------------
    # FORCE coordinates to numeric values
    # --------------------------------------------------------

    lat = float(lat)
    lon = float(lon)

    # --------------------------------------------------------
    # SAFETY CHECK — ONLY NORTHEAST INDIA
    # --------------------------------------------------------

    if not is_inside_ner(lat, lon):
        return None

    # --------------------------------------------------------
    # CREATE MAP AT EXACT SEARCHED COORDINATES
    # --------------------------------------------------------

    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        control_scale=True,
        tiles=None,
    )

    # --------------------------------------------------------
    # REAL SATELLITE / IMAGERY
    # --------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="🛰️ Real Satellite Imagery",
        overlay=False,
        control=True,
    ).add_to(m)

    # --------------------------------------------------------
    # NORMAL ROAD MAP
    # --------------------------------------------------------

    folium.TileLayer(
        tiles=(
            "https://{s}.tile.openstreetmap.org/"
            "{z}/{x}/{y}.png"
        ),
        attr="OpenStreetMap",
        name="🗺️ Road Map",
        overlay=False,
        control=True,
    ).add_to(m)

    # --------------------------------------------------------
    # SEARCHED LOCATION MARKER
    # --------------------------------------------------------

    folium.Marker(
        location=[lat, lon],
        tooltip=f"📍 {name}",
        popup=folium.Popup(
            f"""
            <b>📍 Searched Location</b><br><br>
            <b>{name}</b><br>
            Latitude: {lat:.6f}<br>
            Longitude: {lon:.6f}<br>
            <br>
            <b>Region:</b> Northeast India<br>
            <b>Satellite:</b> Real imagery
            """,
            max_width=300,
        ),
        icon=folium.Icon(
            color="red",
            icon="info-sign",
        ),
    ).add_to(m)

    # --------------------------------------------------------
    # SEARCHED AREA CIRCLE
    # --------------------------------------------------------

    folium.Circle(
        location=[lat, lon],
        radius=3000,
        color="red",
        fill=True,
        fill_opacity=0.08,
        weight=2,
        tooltip="Searched monitoring area — 3 km radius",
    ).add_to(m)

    # --------------------------------------------------------
    # EXACT COORDINATE LABEL
    # --------------------------------------------------------

    folium.Marker(
        location=[lat, lon],
        icon=folium.DivIcon(
            html=f"""
            <div style="
                font-size: 12px;
                font-weight: bold;
                color: white;
                background: rgba(0,0,0,0.70);
                padding: 4px 7px;
                border-radius: 5px;
                white-space: nowrap;
                margin-left: 12px;
                margin-top: -10px;
            ">
                {name}<br>
                {lat:.5f}, {lon:.5f}
            </div>
            """
        ),
    ).add_to(m)

    # --------------------------------------------------------
    # LAYER CONTROL
    # --------------------------------------------------------

    folium.LayerControl(
        collapsed=False
    ).add_to(m)

    return m


def get_real_sus(lat,lon):
    return value_at_raster(SUS_RASTER,lat,lon)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🗺️ NER Explorer")
selected_state=st.sidebar.selectbox("State",STATES)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Static:** 8-factor Random Forest  \n"
    "**Grid:** 250 m  \n"
    "**Fusion:** 60% Static + 40% Dynamic"
)

# ============================================================
# 1. STATE SUSCEPTIBILITY
# ============================================================

st.header(f"🗺️ {selected_state} — Static Susceptibility")

map_path=STATE_MAP_DIR / (
    selected_state.replace(" ","_")+"_susceptibility_map.png"
)

if selected_state=="Mizoram":
    st.warning(
        "RF susceptibility is currently unavailable for Mizoram because "
        "the production predictor stack has incomplete geomorphology coverage. "
        "No synthetic value is inserted."
    )
    if map_path.exists():
        st.image(map_path,use_container_width=True)
else:
    a,b=st.columns([2.4,1])
    with a:
        if map_path.exists():
            st.image(map_path,use_container_width=True)
        else:
            st.warning("State susceptibility PNG not found; the raster remains the source of truth.")
    with b:
        st.markdown("### What the static layer does")
        st.write(
            "The trained Random Forest applies the eight conditioning factors "
            "pixel-by-pixel and produces a 0–1 susceptibility score. "
            "The map shows the spatial pattern of that model output."
        )
        if STATE_CSV.exists():
            sdf=pd.read_csv(STATE_CSV)
            match=None
            for _,row in sdf.iterrows():
                if selected_state.lower() in clean_state_name(row.iloc[0]).lower():
                    match=row;break
            if match is not None:
                # State CSV may have mean/median columns; display numeric fields safely.
                nums=[]
                for c in sdf.columns:
                    try:
                        v=float(match[c])
                        if np.isfinite(v): nums.append((c,v))
                    except Exception: pass
               # Find the actual mean susceptibility column
mean_value = None

for c in sdf.columns:
    c_clean = c.lower().strip()

    if "mean" in c_clean and (
        "suscept" in c_clean or c_clean in ["mean_s", "mean"]
    ):
        try:
            value = float(match[c])
            if np.isfinite(value):
                mean_value = value
                break
        except Exception:
            pass

if mean_value is not None:
    st.metric(
        "State mean susceptibility",
        f"{mean_value:.3f}"
    )
else:
    st.warning("Mean susceptibility value not found in state CSV.")



# ============================================================
# 2. SEARCH ANY NORTHEAST INDIA LOCATION
# ============================================================

st.markdown("---")

st.header("🔎 Search Any Northeast India Location")

st.write(
    "Search a city, district, town, village, or landmark "
    "within Northeast India. Only locations inside the official "
    "8-state NER boundary are accepted."
)

search_col, button_col = st.columns([4, 1])

with search_col:

    q = st.text_input(
        "Search city / district / town / village",
        value="Sohra, Meghalaya",
        placeholder="Example: Tawang, Arunachal Pradesh",
        label_visibility="collapsed",
    )

with button_col:

    search_clicked = st.button(
        "🔍 SEARCH",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# DEFAULT LOCATION
# ============================================================

if "location" not in st.session_state:

    st.session_state["location"] = {
        "name": "Sohra",
        "lat": 25.2702,
        "lon": 91.7323,
        "admin1": "Meghalaya",
        "admin2": "East Khasi Hills",
        "country": "India",
    }


# ============================================================
# SEARCH
# ============================================================

if search_clicked:

    query = q.strip()

    if not query:

        st.warning(
            "Please enter a Northeast India location."
        )

    else:

        try:

            results = geocode(query)

            if not results:

                st.error(
                    "❌ Location not found inside Northeast India."
                )

                st.info(
                    "Search an Arunachal Pradesh, Assam, Meghalaya, "
                    "Manipur, Mizoram, Nagaland, Sikkim, or Tripura "
                    "location."
                )

            else:

                # First valid NER result
                first = results[0]

                st.session_state["location"] = {
                    "name": first["name"],
                    "lat": float(first["latitude"]),
                    "lon": float(first["longitude"]),
                    "admin1": first.get("admin1", ""),
                    "admin2": first.get("admin2", ""),
                    "country": "India",
                }

                st.session_state["search_results"] = results

                st.success(
                    f"📍 Selected: {first['display_name']} "
                    f"| {first['latitude']:.5f}, "
                    f"{first['longitude']:.5f}"
                )

        except Exception as e:

            st.error(
                f"❌ Search failed: {e}"
            )


# ============================================================
# MULTIPLE NER MATCHES
# ============================================================

if "search_results" in st.session_state:

    results = st.session_state["search_results"]

    if len(results) > 1:

        labels = [
            r["display_name"]
            for r in results
        ]

        selected_label = st.selectbox(
            "Other matching Northeast India locations",
            labels,
        )

        selected_result = results[
            labels.index(selected_label)
        ]

        if st.button(
            "📍 USE THIS LOCATION",
            type="secondary",
        ):

            st.session_state["location"] = {
                "name": selected_result["name"],
                "lat": float(
                    selected_result["latitude"]
                ),
                "lon": float(
                    selected_result["longitude"]
                ),
                "admin1": selected_result.get(
                    "admin1", ""
                ),
                "admin2": selected_result.get(
                    "admin2", ""
                ),
                "country": "India",
            }

            st.rerun()


# ============================================================
# FINAL SELECTED LOCATION
# ============================================================

loc = st.session_state["location"]

st.success(
    f"📍 Monitoring: "
    f"**{loc['name']}**, "
    f"{loc['admin2']}, "
    f"{loc['admin1']} | "
    f"{loc['lat']:.5f}, "
    f"{loc['lon']:.5f}"
)
st.success(f"📍 {loc['name']} — {loc['admin2']}, {loc['admin1']} | {loc['lat']:.5f}, {loc['lon']:.5f}")

# ============================================================
# 3. SEARCHED LOCATION: STATIC + ALL 8 FACTORS
# ============================================================

st.markdown("---")
st.header("🎯 Selected Location — Full Model Result")

sus=get_real_sus(loc["lat"],loc["lon"])

if sus is None:
    st.error("No valid RF susceptibility prediction exists at this exact location.")
else:
    s1,s2,s3=st.columns(3)
    s1.metric("RF Susceptibility",f"{sus:.3f}")
    s2.metric("Static Class",classify_sus(sus))
    s3.metric("Model","RF • 700 trees")

    st.subheader("🌍 Eight Static Conditioning Factors")

    factor_rows=[]
    for name,path in FACTOR_RASTERS.items():
        v=value_at_raster(path,loc["lat"],loc["lon"])
        if v is None:
            display="Unavailable"
        elif name=="Elevation":
            display=f"{v:.2f} m"
        elif name=="Slope":
            display=f"{v:.2f}°"
        elif name=="3-Day Rainfall":
            display=f"{v:.2f} mm"
        elif name=="Soil Moisture":
            display=f"{v:.3f}"
        elif name=="NDVI":
            display=f"{v:.3f}"
        elif name=="Distance to Road":
            display=f"{v:.1f} m"
        elif name=="Lineament Density":
            display=f"{v:.4f}"
        else:
            geom={
                1:"Denudational Origin",2:"Fluvial Origin",3:"Glacial Origin",
                4:"Lacustrine Origin",5:"Structural Origin",6:"Water Bodies"
            }
            display=geom.get(round(v),str(round(v)))
        factor_rows.append({"Factor":name,"Observed model input":display})

    st.dataframe(pd.DataFrame(factor_rows),use_container_width=True,hide_index=True)

    st.caption(
        "These are the eight raster inputs actually sampled at the searched pixel; "
        "the RF combines them into the susceptibility score."
    )

# ============================================================
# 4. LOCAL SUSCEPTIBILITY MAP
# ============================================================

if sus is not None:
    st.subheader("🗺️ Searched Area — Local Susceptibility Map")
    fig=make_susceptibility_plot(loc["lat"],loc["lon"],loc["name"])
    if fig is not None:
        st.pyplot(fig,use_container_width=True)
        plt.close(fig)

# ============================================================
# 5. REAL SATELLITE IMAGE — ONLY SEARCHED AREA
# ============================================================

# ============================================================
# 5. REAL SATELLITE IMAGE — ONLY SEARCHED NER AREA
# ============================================================

st.markdown("---")

st.header("🛰️ Searched Area — Real Satellite View")

st.write(
    f"""
    Real satellite imagery centered on the searched location:
    **{loc["name"]}**
    """
)

sat_lat = float(loc["lat"])
sat_lon = float(loc["lon"])

# ------------------------------------------------------------
# SAFETY CHECK
# ------------------------------------------------------------

if not is_inside_ner(sat_lat, sat_lon):

    st.error(
        "❌ Satellite view blocked: the selected coordinates "
        "are outside Northeast India."
    )

else:

    st.success(
        f"📍 Satellite center: "
        f"{sat_lat:.5f}, {sat_lon:.5f}"
    )

    if folium is None:

        st.info(
            "Install folium and streamlit-folium to enable "
            "the interactive satellite map."
        )

    else:

        sat = satellite_map(
            sat_lat,
            sat_lon,
            loc["name"],
            zoom=13,
        )

        if sat is not None:

            st_folium(
                sat,
                use_container_width=True,
                height=600,
            )

        else:

            st.error(
                "❌ Could not create the satellite map "
                "for this NER location."
            )

st.caption(
    "🛰️ Real satellite/imagery basemap from Esri World Imagery. "
    "The map is centered on the searched Northeast India location. "
    "It is not a synthetic image."
)

# ============================================================
# 6. LIVE WEATHER + DYNAMIC LAYER
# ============================================================

st.markdown("---")
st.header("🌦️ Dynamic Layer — Current Conditions at Searched Location")

try:
    w=weather(loc["lat"],loc["lon"])
    cur=w["current"]
    r24,r72=rain_history(w)

    # Use pilot thresholds only for East Khasi Hills; otherwise clearly labelled
    # generic prototype thresholds.
    is_east_khasi = (
        abs(loc["lat"]-EAST_KHASI_HILLS.station_lat)<0.15 and
        abs(loc["lon"]-EAST_KHASI_HILLS.station_lon)<0.15
    )
    if is_east_khasi:
        th24=float(EAST_KHASI_HILLS.threshold_24h_mm)
        th72=float(EAST_KHASI_HILLS.threshold_72h_mm)
        threshold_note="East Khasi Hills pilot configuration"
    else:
        th24=150.0
        th72=300.0
        threshold_note="Generic NER prototype thresholds — not locally calibrated"

    trig,ratio24,ratio72=dynamic_trigger(r24,r72,th24,th72)

    w1,w2,w3,w4=st.columns(4)
    w1.metric("Temperature",f"{cur['temperature_2m']:.1f} °C")
    w2.metric("Wind speed",f"{cur['wind_speed_10m']:.1f} km/h")
    w3.metric("Humidity",f"{cur['relative_humidity_2m']:.0f}%")
    w4.metric("Pressure",f"{cur['surface_pressure']:.0f} hPa")

    w5,w6,w7,w8=st.columns(4)
    w5.metric("Rain now",f"{cur['rain']:.1f} mm")
    w6.metric("Recent 24h rain",f"{r24:.1f} mm")
    w7.metric("Recent 72h rain",f"{r72:.1f} mm")
    w8.metric("Rain probability",f"{max(w['hourly']['precipitation_probability'][:24]):.0f}%")

    st.subheader("🎯 Dynamic Rainfall Trigger")

    d1,d2,d3=st.columns(3)
    d1.metric("24h threshold",f"{th24:.0f} mm")
    d2.metric("72h threshold",f"{th72:.0f} mm")
    d3.metric("Trigger score",f"{trig:.3f}")

    st.caption(threshold_note)

    # Final fusion
    if sus is not None:
        fusion=fuse_risk(sus,trig)
        st.subheader("🚨 Final Operational Risk")
        f1,f2,f3=st.columns(3)
        f1.metric("Static S",f"{sus:.3f}")
        f2.metric("Dynamic T",f"{trig:.3f}")
        f3.metric("Risk R = 0.6S + 0.4T",f"{fusion.risk_score:.3f}")
        st.markdown(f"### {risk_badge(fusion.risk_level)}")

        with st.expander("🧾 Show full risk calculation"):
            for item in fusion.reasoning:
                st.write("•",item)

        st.info(
            "This operational risk map/result combines the RF susceptibility "
            "surface with the current searched-location trigger. It is a "
            "decision-support indicator, not a guaranteed landslide probability."
        )

    # Forecast
    st.subheader("📈 Next 7 Days")
    daily=w["daily"]
    forecast_df=pd.DataFrame({
        "Date":pd.to_datetime(daily["time"]).strftime("%d %b"),
        "Forecast rain (mm)":np.round(daily["precipitation_sum"],1),
        "Rain probability (%)":np.round(daily["precipitation_probability_max"]).astype(int),
    })
    st.dataframe(forecast_df,use_container_width=True,hide_index=True)

except Exception as e:
    st.error(f"Dynamic weather layer failed: {e}")

# ============================================================
# 7. ROUTE MAP — ONLY THREE REQUESTED AREAS
# ============================================================

st.markdown("---")
st.header("🛣️ Landslide-Aware Route Map")

route_districts = {
    "Kamrup Metropolitan, Assam": ("Kamrup Metropolitan","Assam"),
    "East Khasi Hills, Meghalaya": ("East Khasi Hills","Meghalaya"),
    "West Tripura, Tripura": ("West Tripura","Tripura"),
}

# Detect current area.
current_area=None
for label,(district,state) in route_districts.items():
    if str(loc.get("admin2","")).strip().lower()==district.lower() and \
       str(loc.get("admin1","")).strip().lower()==state.lower():
        current_area=label
        break

route_choice=st.selectbox(
    "Route pilot area",
    list(route_districts.keys()),
    index=list(route_districts.keys()).index(current_area) if current_area else 0,
)

district,state=route_districts[route_choice]

if district=="West Tripura":
    st.warning(
        "West Tripura is intentionally listed as a route pilot, but the current "
        "road-risk GeoPackage was built only for the Assam + Meghalaya 10-district "
        "pilot. Do not show a fabricated West Tripura route. Extract West Tripura "
        "OSM roads first, then attach susceptibility and calculate road cost."
    )
    st.markdown("""
**Required West Tripura pipeline:**
`OSM PBF → West Tripura roads → susceptibility sampling → incident matching → road cost → route graph`
""")
else:
    if gpd is None:
        st.error("Install geopandas to display the route network.")
    elif not ROUTE_GPKG.exists():
        st.error(f"Route GeoPackage not found: {ROUTE_GPKG}")
    else:
        try:
            roads=gpd.read_file(ROUTE_GPKG,layer="pilot_roads_cost")
            roads=roads[
                roads["district"].astype(str).str.strip().str.lower()==district.lower()
            ].copy()

            if roads.empty:
                st.warning("No road records found for this district.")
            else:
                # Keep the map responsive.
                roads["geometry"]=roads.geometry.simplify(0.00008,preserve_topology=True)

                if folium is not None:
                    center=STATE_CENTERS[state]
                    rm=folium.Map(location=center,zoom_start=10,control_scale=True)

                    folium.TileLayer(
                        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        attr="Esri World Imagery",
                        name="🛰️ Satellite",
                    ).add_to(rm)

                    colors={
                        "AVAILABLE":"green",
                        "CAUTION":"orange",
                        "AVOID_IF_ALTERNATIVE":"red",
                        "BLOCKED":"black",
                    }

                    # Limit to a manageable number for browser rendering.
                    show=roads.head(7000)

                    for _,r in show.iterrows():
                        status=str(r.get("routing_status","AVAILABLE"))
                        color=colors.get(status,"gray")
                        tooltip=(
                            f"{r.get('road_name','Unnamed')} | "
                            f"{status} | LSI {float(r.get('LSI_mean',0)):.3f}"
                        )
                        folium.GeoJson(
                            r.geometry.__geo_interface__,
                            style_function=lambda feature,c=color:{
                                "color":c,"weight":3,"opacity":0.75
                            },
                            tooltip=tooltip,
                        ).add_to(rm)

                    folium.LayerControl().add_to(rm)
                    st_folium(rm,use_container_width=True,height=600)

                    counts=roads["routing_status"].value_counts()
                    st.dataframe(
                        counts.rename_axis("Routing status").reset_index(name="Road segments"),
                        use_container_width=True,hide_index=True
                    )

                    st.caption(
                        "Green = available, orange = caution, red = avoid if an alternative exists. "
                        "The road layer uses the project's real OSM-derived road network and attached RF LSI."
                    )
        except Exception as e:
            st.error(f"Route map failed: {e}")

# ============================================================
# 8. LIVE 7-POINT DYNAMIC RISK MONITORING
# ============================================================

st.markdown("---")

st.header("📡 Live Dynamic Risk Monitoring")

st.caption(
    "Seven real monitoring locations across Northeast India. "
    "Dynamic trigger is calculated from recent rainfall; "
    "static susceptibility comes from the production RF raster."
)

# ------------------------------------------------------------
# RISK COLORS
# ------------------------------------------------------------

RISK_COLORS = {
    "LOW": "green",
    "MEDIUM": "orange",
    "HIGH": "red",
    "CRITICAL": "darkred",
}

RISK_EMOJI = {
    "LOW": "🟢",
    "MEDIUM": "🟡",
    "HIGH": "🟠",
    "CRITICAL": "🔴",
}

# ------------------------------------------------------------
# GET LIVE DATA FOR ALL 7 LOCATIONS
# ------------------------------------------------------------

@st.cache_data(ttl=600)
def get_all_monitoring_data():

    rows = []

    for label, (lat, lon) in MONITORING.items():

        try:

            # --------------------------------------------
            # REAL RF SUSCEPTIBILITY
            # --------------------------------------------

            sus = get_real_sus(lat, lon)

            # --------------------------------------------
            # REAL WEATHER
            # --------------------------------------------

            w = weather(lat, lon)

            cur = w["current"]

            # Recent rainfall only
            r24, r72 = rain_history(w)

            # --------------------------------------------
            # THRESHOLDS
            # --------------------------------------------

            is_east_khasi = (
                abs(lat - EAST_KHASI_HILLS.station_lat) < 0.15
                and
                abs(lon - EAST_KHASI_HILLS.station_lon) < 0.15
            )

            if is_east_khasi:

                th24 = float(
                    EAST_KHASI_HILLS.threshold_24h_mm
                )

                th72 = float(
                    EAST_KHASI_HILLS.threshold_72h_mm
                )

            else:

                th24 = 150.0
                th72 = 300.0

            # --------------------------------------------
            # DYNAMIC TRIGGER
            # --------------------------------------------

            trig, ratio24, ratio72 = dynamic_trigger(
                r24,
                r72,
                th24,
                th72
            )

            # --------------------------------------------
            # RISK FUSION
            # --------------------------------------------

            if sus is not None:

                fusion = fuse_risk(
                    sus,
                    trig
                )

                risk_score = float(
                    fusion.risk_score
                )

                risk_level = str(
                    fusion.risk_level
                ).upper()

            else:

                risk_score = np.nan
                risk_level = "UNAVAILABLE"

            # --------------------------------------------
            # RAIN PROBABILITY
            # --------------------------------------------

            rain_prob = 0

            if "precipitation_probability" in w["hourly"]:

                probs = w["hourly"][
                    "precipitation_probability"
                ]

                # next 24 hours
                rain_prob = int(
                    max(probs[:24])
                )

            # --------------------------------------------
            # STORE
            # --------------------------------------------

            rows.append({

                "Location": label,

                "Latitude": lat,

                "Longitude": lon,

                "Susceptibility": sus,

                "Trigger": trig,

                "Risk Score": risk_score,

                "Risk Level": risk_level,

                "Rain 24h": r24,

                "Rain 72h": r72,

                "Rain Probability": rain_prob,

                "Temperature": float(
                    cur["temperature_2m"]
                ),

                "Humidity": float(
                    cur["relative_humidity_2m"]
                ),

                "Status": "LIVE"

            })

        except Exception as e:

            rows.append({

                "Location": label,

                "Latitude": lat,

                "Longitude": lon,

                "Susceptibility": np.nan,

                "Trigger": np.nan,

                "Risk Score": np.nan,

                "Risk Level": "UNAVAILABLE",

                "Rain 24h": np.nan,

                "Rain 72h": np.nan,

                "Rain Probability": np.nan,

                "Temperature": np.nan,

                "Humidity": np.nan,

                "Status": f"ERROR: {e}"

            })

    return pd.DataFrame(rows)


monitor_df = get_all_monitoring_data()


# ============================================================
# TOP SUMMARY
# ============================================================

total_locations = len(monitor_df)

high_count = int(
    (monitor_df["Risk Level"] == "HIGH").sum()
)

medium_count = int(
    (monitor_df["Risk Level"] == "MEDIUM").sum()
)

low_count = int(
    (monitor_df["Risk Level"] == "LOW").sum()
)

critical_count = int(
    (monitor_df["Risk Level"] == "CRITICAL").sum()
)

unavailable_count = int(
    (monitor_df["Risk Level"] == "UNAVAILABLE").sum()
)


c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Monitoring Locations",
    total_locations
)

c2.metric(
    "HIGH / CRITICAL",
    high_count + critical_count
)

c3.metric(
    "MEDIUM",
    medium_count
)

c4.metric(
    "LOW",
    low_count
)

c5.metric(
    "Data Unavailable",
    unavailable_count
)


# ============================================================
# LIVE MAP
# ============================================================

st.subheader("🗺️ Live Dynamic Risk Map")

if folium is None:

    st.error(
        "Install folium and streamlit-folium first."
    )

else:

    # Northeast India center
    monitor_map = folium.Map(
        location=[25.5, 92.8],
        zoom_start=6,
        control_scale=True,
        tiles="OpenStreetMap"
    )

    # Satellite layer
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/"
            "ArcGIS/rest/services/World_Imagery/"
            "MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="🛰️ Satellite",
        overlay=False,
        control=True
    ).add_to(monitor_map)

    # OpenStreetMap
    folium.TileLayer(
        tiles=(
            "https://{s}.tile.openstreetmap.org/"
            "{z}/{x}/{y}.png"
        ),
        attr="OpenStreetMap",
        name="🗺️ Roads",
        overlay=False,
        control=True
    ).add_to(monitor_map)


    # --------------------------------------------------------
    # ADD ALL 7 MONITORING POINTS
    # --------------------------------------------------------

    for _, row in monitor_df.iterrows():

        lat = row["Latitude"]
        lon = row["Longitude"]

        level = str(
            row["Risk Level"]
        ).upper()

        color = RISK_COLORS.get(
            level,
            "gray"
        )

        emoji = RISK_EMOJI.get(
            level,
            "⚪"
        )

        # Safe formatting
        s = row["Susceptibility"]

        t = row["Trigger"]

        r = row["Risk Score"]

        r24 = row["Rain 24h"]

        r72 = row["Rain 72h"]


        if np.isfinite(s):

            s_text = f"{s:.4f}"

        else:

            s_text = "Unavailable"


        if np.isfinite(t):

            t_text = f"{t:.4f}"

        else:

            t_text = "Unavailable"


        if np.isfinite(r):

            r_text = f"{r:.4f}"

        else:

            r_text = "Unavailable"


        if np.isfinite(r24):

            r24_text = f"{r24:.1f} mm"

        else:

            r24_text = "Unavailable"


        if np.isfinite(r72):

            r72_text = f"{r72:.1f} mm"

        else:

            r72_text = "Unavailable"


        popup_html = f"""
        <div style="
            width:260px;
            font-family:Arial;
        ">

            <h3 style="margin-bottom:8px;">
                {emoji} {row["Location"]}
            </h3>

            <hr>

            <b>Static Susceptibility S:</b>
            {s_text}
            <br><br>

            <b>Dynamic Trigger T:</b>
            {t_text}
            <br><br>

            <b>Final Risk R:</b>
            {r_text}
            <br><br>

            <b>Risk Level:</b>
            {level}
            <br><br>

            <b>Recent 24h Rain:</b>
            {r24_text}
            <br>

            <b>Recent 72h Rain:</b>
            {r72_text}
            <br><br>

            <b>Data:</b>
            LIVE
            <br>

            <small>
            R = 0.6 × S + 0.4 × T
            </small>

        </div>
        """


        # ----------------------------------------------------
        # CIRCLE MARKER
        # ----------------------------------------------------

        folium.CircleMarker(

            location=[
                lat,
                lon
            ],

            radius=12,

            color=color,

            fill=True,

            fill_color=color,

            fill_opacity=0.85,

            weight=3,

            popup=folium.Popup(
                popup_html,
                max_width=320
            ),

            tooltip=(
                f"{emoji} "
                f"{row['Location']} — "
                f"{level} — "
                f"R={r_text}"
            )

        ).add_to(monitor_map)


        # ----------------------------------------------------
        # LOCATION LABEL
        # ----------------------------------------------------

        folium.Marker(

            location=[
                lat,
                lon
            ],

            icon=folium.DivIcon(
                html=f"""
                <div style="
                    font-size:11px;
                    font-weight:bold;
                    color:#111;
                    background:white;
                    padding:2px 5px;
                    border-radius:4px;
                    border:1px solid #888;
                    white-space:nowrap;
                ">
                    {row["Location"].split(",")[0]}
                </div>
                """
            )

        ).add_to(monitor_map)


    # --------------------------------------------------------
    # LEGEND
    # --------------------------------------------------------

    legend_html = """

    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 190px;
        z-index:9999;
        background:white;
        border:2px solid #555;
        border-radius:8px;
        padding:10px;
        font-size:13px;
    ">

    <b>Dynamic Risk</b><br><br>

    <span style="color:green;">●</span>
    LOW<br>

    <span style="color:orange;">●</span>
    MEDIUM<br>

    <span style="color:red;">●</span>
    HIGH<br>

    <span style="color:darkred;">●</span>
    CRITICAL<br>

    <span style="color:gray;">●</span>
    UNAVAILABLE

    </div>

    """

    monitor_map.get_root().html.add_child(
        folium.Element(legend_html)
    )


    folium.LayerControl().add_to(
        monitor_map
    )


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    st_folium(
        monitor_map,
        use_container_width=True,
        height=650
    )


    # --------------------------------------------------------
    # DOWNLOAD INTERACTIVE MAP
    # --------------------------------------------------------

    map_html = monitor_map.get_root().render()

    st.download_button(

        label="⬇️ Download Dynamic Risk Map",

        data=map_html,

        file_name="giri_rakshak_dynamic_risk_map.html",

        mime="text/html"

    )


# ============================================================
# 7 LOCATION RISK TABLE
# ============================================================

st.subheader("📊 Seven-Point Dynamic Risk Status")

display_df = monitor_df.copy()

display_df["S"] = display_df[
    "Susceptibility"
].map(
    lambda x:
    f"{x:.4f}" if np.isfinite(x)
    else "Unavailable"
)

display_df["T"] = display_df[
    "Trigger"
].map(
    lambda x:
    f"{x:.4f}" if np.isfinite(x)
    else "Unavailable"
)

display_df["Final R"] = display_df[
    "Risk Score"
].map(
    lambda x:
    f"{x:.4f}" if np.isfinite(x)
    else "Unavailable"
)

display_df["Risk"] = display_df[
    "Risk Level"
].map(
    lambda x:
    f"{RISK_EMOJI.get(x, '⚪')} {x}"
)

display_df["24h Rain"] = display_df[
    "Rain 24h"
].map(
    lambda x:
    f"{x:.1f} mm" if np.isfinite(x)
    else "Unavailable"
)

display_df["72h Rain"] = display_df[
    "Rain 72h"
].map(
    lambda x:
    f"{x:.1f} mm" if np.isfinite(x)
    else "Unavailable"
)


st.dataframe(

    display_df[
        [
            "Location",
            "S",
            "T",
            "Final R",
            "Risk",
            "24h Rain",
            "72h Rain"
        ]
    ],

    use_container_width=True,

    hide_index=True

)


st.caption(
    "S = RF susceptibility, T = rainfall-driven dynamic trigger, "
    "R = 0.6S + 0.4T. Dynamic values are fetched from live weather "
    "for the seven monitoring coordinates."
)

# ============================================================
# 9. MODEL VALIDATION + EXPLANATION
# ============================================================

st.markdown("---")
st.header("🎯 Model Validation")
v1,v2,v3,v4=st.columns(4)
v1.metric("Independent ROC-AUC","0.8355")
v2.metric("PR-AUC","0.7535")
v3.metric("Balanced Accuracy","0.7829")
v4.metric("Training Samples","991")
st.caption("Independent evaluation on 74 unseen Meghalaya locations.")

st.header("🧠 What each layer does")
st.markdown("""
**STATIC LAYER**
- Takes 8 real raster predictors.
- RF produces a 0–1 susceptibility score for each valid 250 m cell.
- The searched point shows all 8 actual input values.

**DYNAMIC LAYER**
- Uses current/recent rainfall and forecast weather at the searched coordinate.
- Shows temperature, wind, humidity, pressure, current rain, 24 h rain, 72 h rain and rain probability.
- Computes the rainfall trigger independently of the trained RF.

**RISK FUSION**
- `R = 0.6 × S + 0.4 × T`
- Produces LOW / MEDIUM / HIGH / CRITICAL operational risk.

**SATELLITE**
- The searched-area map uses a real satellite imagery basemap.
- Optional GEE Sentinel-2 / Sentinel-1 disturbance checking can remain available separately.
- Satellite disturbance is corroborating evidence, not proof of a landslide.

**ROUTE**
- Only the three requested pilot areas are offered.
- Existing real OSM + RF road-risk data can be shown for Kamrup Metropolitan and East Khasi Hills.
- West Tripura is listed but deliberately blocked until its road network is actually extracted.
""")

st.success(
    "Data integrity: no synthetic susceptibility or weather values are inserted. "
    "Mizoram remains unavailable where the production predictor stack has no valid coverage."
)
