from pathlib import Path
import json
import math
import requests

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from shapely.geometry import Point
from rasterio.warp import transform


# ============================================================
# GIRI-RAKSHAK
# FOUR-REGION LANDSLIDE-AWARE GPS ROUTING DEMO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR / "pilot_route_data"
)

OUTPUT_DIR.mkdir(
    exist_ok=True
)


# ============================================================
# REAL RF SUSCEPTIBILITY RASTER
# ============================================================

SUSCEPTIBILITY_RASTER = (
    BASE_DIR
    / "processed"
    / "step70_8factor_susceptibility_probability.tif"
)


# ============================================================
# REAL HISTORICAL LANDSLIDE DATA
# ============================================================

HISTORICAL_FILE = (
    OUTPUT_DIR
    / "pilot_historical_landslide_points.csv"
)


# ============================================================
# REAL VALIDATED REGION VALUES
#
# These are the validated outputs already produced
# by your Giri-Rakshak pipeline.
# ============================================================

REGIONS = {

    "East Khasi Hills": {

        "state": "Meghalaya",

        "district": "East Khasi Hills",

        "start": [
            25.5788,
            91.8933
        ],

        "destination": [
            25.6200,
            91.8827
        ],

        "susceptibility": 0.5677,

        "trigger": 0.2069,

    },


    "Guwahati": {

        "state": "Assam",

        "district": "Kamrup Metropolitan",

        "start": [
            26.1445,
            91.7362
        ],

        "destination": [
            26.1760,
            91.7580
        ],

        "susceptibility": 0.6830,

        "trigger": 0.2089,

    },


    "Agartala": {

        "state": "Tripura",

        "district": "West Tripura",

        "start": [
            23.8312,
            91.2824
        ],

        "destination": [
            23.8580,
            91.3060
        ],

        "susceptibility": 0.4184,

        "trigger": 0.2345,

    },


    "Mangan": {

        "state": "Sikkim",

        "district": "Mangan",

        "start": [
            27.5167,
            88.5333
        ],

        "destination": [
            27.5480,
            88.5750
        ],

        "susceptibility": 0.8299,

        "trigger": 0.4363,

    },

}


# ============================================================
# ROUTING WEIGHTS
# ============================================================

STATIC_WEIGHT = 0.60
DYNAMIC_WEIGHT = 0.40

HISTORICAL_PENALTY = 1.20

TRAFFIC_PENALTY = 1.25

DEMO_BLOCKED_INCIDENT_PENALTY = float("inf")


# ============================================================
# REAL ROAD ROUTING
# ============================================================

OSRM_URL = (
    "https://router.project-osrm.org/"
    "route/v1/driving/"
)


# ============================================================
# LOAD HISTORICAL LANDSLIDES
# ============================================================

def load_historical_points():

    if not HISTORICAL_FILE.exists():

        print(
            "Historical landslide file not found."
        )

        return gpd.GeoDataFrame(
            columns=[
                "latitude",
                "longitude"
            ],
            geometry=[],
            crs="EPSG:4326",
        )

    df = pd.read_csv(
        HISTORICAL_FILE
    )

    # --------------------------------------------------------
    # Try common coordinate names
    # --------------------------------------------------------

    lat_col = None
    lon_col = None

    for c in df.columns:

        c_lower = str(c).lower()

        if (
            "lat" in c_lower
            and lat_col is None
        ):

            lat_col = c

        if (
            (
                "lon" in c_lower
                or "lng" in c_lower
            )
            and lon_col is None
        ):

            lon_col = c

    if (
        lat_col is None
        or lon_col is None
    ):

        print(
            "Could not identify "
            "historical coordinates."
        )

        return gpd.GeoDataFrame(
            columns=[
                "latitude",
                "longitude"
            ],
            geometry=[],
            crs="EPSG:4326",
        )

    df["latitude"] = pd.to_numeric(
        df[lat_col],
        errors="coerce"
    )

    df["longitude"] = pd.to_numeric(
        df[lon_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "latitude",
            "longitude"
        ]
    )

    geometry = [
        Point(
            lon,
            lat
        )

        for lat, lon
        in zip(
            df["latitude"],
            df["longitude"]
        )
    ]

    return gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs="EPSG:4326",
    )


historical_points = (
    load_historical_points()
)


# ============================================================
# OSRM ROUTE
# ============================================================

def get_route(
    start,
    destination,
    waypoint=None,
):

    points = [
        start
    ]

    if waypoint is not None:

        points.append(
            waypoint
        )

    points.append(
        destination
    )

    coordinates = ";".join(

        f"{lon},{lat}"

        for lat, lon
        in points

    )

    url = (
        OSRM_URL
        + coordinates
    )

    params = {

        "overview": "full",

        "geometries": "geojson",

        "steps": "true",

        "alternatives": "false",

    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "Ok":

        raise RuntimeError(
            "OSRM could not find "
            "a route."
        )

    route = data[
        "routes"
    ][0]

    geometry = [

        [
            lat,
            lon
        ]

        for lon, lat
        in route[
            "geometry"
        ][
            "coordinates"
        ]

    ]

    return {

        "geometry":
            geometry,

        "distance_km":
            route[
                "distance"
            ] / 1000.0,

        "duration_min":
            route[
                "duration"
            ] / 60.0,

        "raw":
            route,

    }


# ============================================================
# ROAD NAMES
# ============================================================

def get_road_names(
    route
):

    names = []

    for leg in route[
        "raw"
    ].get(
        "legs",
        []
    ):

        for step in leg.get(
            "steps",
            []
        ):

            name = step.get(
                "name",
                ""
            )

            ref = step.get(
                "ref",
                ""
            )

            value = (
                name
                or ref
            )

            if value:

                value = str(
                    value
                ).strip()

                if (
                    value
                    and value
                    not in names
                ):

                    names.append(
                        value
                    )

    if not names:

        return "Unnamed OSM road"

    return " → ".join(
        names[:6]
    )


# ============================================================
# HISTORICAL LANDSLIDE MATCH
# ============================================================

def count_historical_events(
    route_geometry,
    radius_m=1000,
):

    if historical_points.empty:

        return 0, []


    route_gdf = gpd.GeoDataFrame(

        geometry=[
            Point(
                lon,
                lat
            )

            for lat, lon
            in route_geometry
        ],

        crs="EPSG:4326",

    ).to_crs(
        "EPSG:6933"
    )


    route_line = (
        route_gdf
        .union_all()
    )


    historical_projected = (
        historical_points
        .to_crs("EPSG:6933")
    )


    distances = (
        historical_projected
        .geometry
        .distance(
            route_line
        )
    )


    matched = (
        historical_projected[
            distances <= radius_m
        ]
    )


    return (
        len(matched),
        matched
    )


# ============================================================
# REAL RF SAMPLING
# ============================================================

def sample_rf(
    route_geometry
):

    values = []

    with rasterio.open(
        SUSCEPTIBILITY_RASTER
    ) as src:

        raster = src.read(1)

        for lat, lon in route_geometry:

            try:

                xs, ys = transform(

                    "EPSG:4326",

                    src.crs,

                    [lon],

                    [lat],

                )

                row, col = src.index(

                    xs[0],

                    ys[0],

                )

                if (

                    row < 0

                    or row >=
                    raster.shape[0]

                    or col < 0

                    or col >=
                    raster.shape[1]

                ):

                    continue

                value = raster[
                    row,
                    col
                ]

                if (

                    np.isfinite(value)

                    and value !=
                    src.nodata

                    and 0 <= value <= 1

                ):

                    values.append(
                        float(value)
                    )

            except Exception:

                continue


    if not values:

        return None


    return {

        "mean":
            float(
                np.mean(values)
            ),

        "max":
            float(
                np.max(values)
            ),

        "samples":
            len(values),

    }


# ============================================================
# RISK FUSION
# ============================================================

def risk_score(
    susceptibility,
    trigger,
):

    return (

        STATIC_WEIGHT
        * susceptibility

        +

        DYNAMIC_WEIGHT
        * trigger

    )


# ============================================================
# DYNAMIC PENALTY
# ============================================================

def dynamic_penalty(
    trigger
):

    if trigger < 0.35:

        return 1.0

    if trigger < 0.65:

        return 1.3

    if trigger < 0.75:

        return 2.0

    return 2.5


# ============================================================
# ROUTE COST
# ============================================================

def route_cost(

    base_time,

    susceptibility,

    trigger,

    historical_count,

    incident_blocked,

    traffic=False,

):

    p_dynamic = (
        dynamic_penalty(
            trigger
        )
    )


    p_historical = (

        HISTORICAL_PENALTY

        if historical_count > 0

        else 1.0

    )


    p_traffic = (

        TRAFFIC_PENALTY

        if traffic

        else 1.0

    )


    p_incident = (

        DEMO_BLOCKED_INCIDENT_PENALTY

        if incident_blocked

        else 1.0

    )


    cost = (

        base_time

        *

        (
            1
            + STATIC_WEIGHT
            * susceptibility
        )

        *

        p_dynamic

        *

        p_historical

        *

        p_traffic

        *

        p_incident

    )


    return {

        "cost":
            cost,

        "dynamic_penalty":
            p_dynamic,

        "historical_penalty":
            p_historical,

        "traffic_penalty":
            p_traffic,

        "incident_penalty":
            p_incident,

    }


# ============================================================
# BUILD ONE REGION
# ============================================================

def build_region(
    region_name,
    cfg,
):

    print()
    print("=" * 75)

    print(
        f"GIRI-RAKSHAK — "
        f"{region_name.upper()}"
    )

    print("=" * 75)

    print()

    print(
        "State:",
        cfg["state"]
    )

    print(
        "District:",
        cfg["district"]
    )

    print(
        "GPS start:",
        cfg["start"]
    )

    print(
        "GPS destination:",
        cfg["destination"]
    )


    # --------------------------------------------------------
    # ROUTE A
    # --------------------------------------------------------

    print()

    print(
        "Calculating Route A..."
    )

    route_a = get_route(

        cfg["start"],

        cfg["destination"],

    )


    # --------------------------------------------------------
    # ROUTE B
    #
    # Slightly offset waypoint forces
    # another real road corridor.
    # --------------------------------------------------------

    lat1, lon1 = cfg["start"]

    lat2, lon2 = cfg["destination"]

    waypoint = [

        (
            lat1 + lat2
        ) / 2,

        lon1 + 0.015,

    ]


    print(
        "Calculating Route B..."
    )

    route_b = get_route(

        cfg["start"],

        cfg["destination"],

        waypoint,

    )


    # --------------------------------------------------------
    # ROAD NAMES
    # --------------------------------------------------------

    route_a_name = get_road_names(
        route_a
    )

    route_b_name = get_road_names(
        route_b
    )


    # --------------------------------------------------------
    # RF
    # --------------------------------------------------------

    rf_a = sample_rf(
        route_a["geometry"]
    )

    rf_b = sample_rf(
        route_b["geometry"]
    )


    # Use validated regional score if raster
    # sampling is unavailable.

    s_a = (

        rf_a["mean"]

        if rf_a is not None

        else cfg["susceptibility"]

    )


    s_b = (

        rf_b["mean"]

        if rf_b is not None

        else cfg["susceptibility"]

    )


    # --------------------------------------------------------
    # DYNAMIC TRIGGER
    # --------------------------------------------------------

    trigger = cfg[
        "trigger"
    ]


    # --------------------------------------------------------
    # HISTORICAL LANDSLIDES
    # --------------------------------------------------------

    hist_a, hist_points_a = (
        count_historical_events(
            route_a["geometry"]
        )
    )

    hist_b, hist_points_b = (
        count_historical_events(
            route_b["geometry"]
        )
    )


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk_a = risk_score(
        s_a,
        trigger
    )

    risk_b = risk_score(
        s_b,
        trigger
    )


    # --------------------------------------------------------
    # DEMO ACTIVE BLOCK
    #
    # Route A is blocked for demonstration.
    #
    # This does NOT modify the RF model.
    # --------------------------------------------------------

    route_a_blocked = True

    route_b_blocked = False


    # --------------------------------------------------------
    # TRAFFIC
    # --------------------------------------------------------

    route_a_traffic = True

    route_b_traffic = False


    # --------------------------------------------------------
    # COST
    # --------------------------------------------------------

    cost_a = route_cost(

        route_a["duration_min"],

        s_a,

        trigger,

        hist_a,

        route_a_blocked,

        route_a_traffic,

    )


    cost_b = route_cost(

        route_b["duration_min"],

        s_b,

        trigger,

        hist_b,

        route_b_blocked,

        route_b_traffic,

    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print()

    print(
        "ROUTE A"
    )

    print(
        "Road:",
        route_a_name
    )

    print(
        "Distance:",
        f"{route_a['distance_km']:.2f} km"
    )

    print(
        "RF susceptibility:",
        f"{s_a:.4f}"
    )

    print(
        "Dynamic trigger:",
        f"{trigger:.4f}"
    )

    print(
        "Historical events:",
        hist_a
    )

    print(
        "Risk score:",
        f"{risk_a:.4f}"
    )

    print(
        "Traffic:",
        "HEAVY"
    )

    print(
        "Active incident:",
        "BLOCKED"
    )

    print(
        "Route cost:",
        "INFINITY"
    )

    print(
        "Decision:",
        "DO NOT TAKE"
    )


    print()

    print(
        "ROUTE B"
    )

    print(
        "Road:",
        route_b_name
    )

    print(
        "Distance:",
        f"{route_b['distance_km']:.2f} km"
    )

    print(
        "RF susceptibility:",
        f"{s_b:.4f}"
    )

    print(
        "Dynamic trigger:",
        f"{trigger:.4f}"
    )

    print(
        "Historical events:",
        hist_b
    )

    print(
        "Risk score:",
        f"{risk_b:.4f}"
    )

    print(
        "Traffic:",
        "NORMAL"
    )

    print(
        "Active incident:",
        "OPEN"
    )

    print(
        "Route cost:",
        f"{cost_b['cost']:.2f}"
    )

    print(
        "Decision:",
        "TAKE THIS ROUTE"
    )


    # ========================================================
    # MAP
    # ========================================================

    center = [

        (
            cfg["start"][0]
            +
            cfg["destination"][0]
        ) / 2,

        (
            cfg["start"][1]
            +
            cfg["destination"][1]
        ) / 2,

    ]


    m = folium.Map(

        location=center,

        zoom_start=13,

        tiles="OpenStreetMap",

        control_scale=True,

    )


    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    folium.Marker(

        cfg["start"],

        tooltip="CURRENT GPS LOCATION",

        popup=f"""

        <b>GIRI-RAKSHAK</b><br><br>

        <b>State:</b>
        {cfg["state"]}<br>

        <b>District:</b>
        {cfg["district"]}<br>

        <b>GPS:</b>
        {cfg["start"][0]:.5f},
        {cfg["start"][1]:.5f}

        """,

        icon=folium.Icon(

            color="blue",

            icon="car",

            prefix="fa",

        ),

    ).add_to(m)


    # --------------------------------------------------------
    # DESTINATION
    # --------------------------------------------------------

    folium.Marker(

        cfg["destination"],

        tooltip="DESTINATION",

        popup=f"""

        <b>DESTINATION</b><br><br>

        <b>State:</b>
        {cfg["state"]}<br>

        <b>District:</b>
        {cfg["district"]}<br>

        <b>Recommended corridor:</b><br>
        {route_b_name}

        """,

        icon=folium.Icon(

            color="green",

            icon="flag",

            prefix="fa",

        ),

    ).add_to(m)


    # --------------------------------------------------------
    # ROUTE A
    # --------------------------------------------------------

    folium.PolyLine(

        route_a["geometry"],

        color="#d93025",

        weight=7,

        opacity=0.85,

        dash_array="12,10",

        tooltip=(
            "ROUTE A — DO NOT TAKE"
        ),

    ).add_to(m)


    # --------------------------------------------------------
    # ROUTE B
    # --------------------------------------------------------

    folium.PolyLine(

        route_b["geometry"],

        color="#188038",

        weight=8,

        opacity=0.95,

        tooltip=(
            "ROUTE B — TAKE THIS ROUTE"
        ),

    ).add_to(m)


    # --------------------------------------------------------
    # HISTORICAL LANDSLIDE MARKERS
    # --------------------------------------------------------

    all_hist = []

    if not hist_points_a.empty:

        all_hist.append(
            hist_points_a
        )

    if not hist_points_b.empty:

        all_hist.append(
            hist_points_b
        )


    if all_hist:

        combined = pd.concat(
            all_hist
        ).drop_duplicates()


        for _, row in combined.iterrows():

            lat = float(
                row["latitude"]
            )

            lon = float(
                row["longitude"]
            )


            folium.CircleMarker(

                [lat, lon],

                radius=8,

                color="#7b1fa2",

                fill=True,

                fill_color="#9c27b0",

                fill_opacity=0.9,

                popup=f"""

                <b>HISTORICAL LANDSLIDE</b>

                <br><br>

                <b>State:</b>
                {cfg["state"]}<br>

                <b>District:</b>
                {cfg["district"]}<br>

                <b>Latitude:</b>
                {lat:.6f}<br>

                <b>Longitude:</b>
                {lon:.6f}<br>

                <br>

                <b>Status:</b>
                HISTORICAL EVIDENCE

                <br><br>

                <small>
                This event is historical
                and does NOT mean the road
                is currently blocked.
                </small>

                """,

                tooltip=(
                    "Historical landslide"
                ),

            ).add_to(m)


    # --------------------------------------------------------
    # SYNTHETIC ACTIVE BLOCK
    # --------------------------------------------------------

    block_position = route_a[
        "geometry"
    ][
        int(
            len(
                route_a[
                    "geometry"
                ]
            ) * 0.55
        )
    ]


    folium.CircleMarker(

        block_position,

        radius=11,

        color="#b3261e",

        fill=True,

        fill_color="#d93025",

        fill_opacity=0.95,

        popup=f"""

        <b>ACTIVE ROAD BLOCK</b>

        <br><br>

        <b>State:</b>
        {cfg["state"]}<br>

        <b>District:</b>
        {cfg["district"]}<br>

        <b>Road:</b>
        {route_a_name}<br>

        <b>Status:</b>
        BLOCKED<br>

        <b>Routing action:</b>
        REMOVE FROM ROUTING

        <br><br>

        <small>
        SYNTHETIC DEMONSTRATION INCIDENT
        </small>

        """,

        tooltip=(
            "ROAD BLOCKED"
        ),

    ).add_to(m)


    # --------------------------------------------------------
    # PROFESSIONAL PANEL
    # --------------------------------------------------------

    css = """

    <style>

    .giri-panel {

        position: fixed;

        top: 18px;

        left: 55px;

        z-index: 9999;

        width: 430px;

        max-height: 92vh;

        overflow-y: auto;

        background: white;

        border-radius: 12px;

        box-shadow:
        0 4px 24px rgba(0,0,0,.28);

        font-family:
        Arial, Helvetica, sans-serif;

    }


    .header {

        padding: 18px 20px;

        border-bottom:
        1px solid #e5e5e5;

    }


    .title {

        font-size: 22px;

        font-weight: 700;

        color: #202124;

    }


    .subtitle {

        font-size: 10px;

        color: #6b7280;

        margin-top: 4px;

        letter-spacing: .7px;

    }


    .location {

        padding: 13px 20px;

        border-bottom:
        1px solid #eeeeee;

    }


    .label {

        font-size: 10px;

        color: #777;

        text-transform: uppercase;

        letter-spacing: .6px;

    }


    .value {

        font-size: 14px;

        font-weight: 600;

        margin-top: 4px;

    }


    .alert {

        margin: 14px 20px;

        padding: 13px;

        background: #fff4f4;

        border:
        1px solid #ef9a9a;

        border-radius: 8px;

    }


    .alert-title {

        font-size: 15px;

        font-weight: 700;

        color: #b3261e;

    }


    .route {

        margin: 10px 20px;

        padding: 14px;

        border-radius: 9px;

    }


    .safe {

        border:
        2px solid #188038;

    }


    .danger {

        border:
        2px solid #d93025;

    }


    .route-title {

        font-size: 16px;

        font-weight: 700;

    }


    .route-info {

        margin-top: 9px;

        font-size: 12px;

        line-height: 1.7;

        color: #444;

    }


    .safe-button {

        margin-top: 11px;

        padding: 10px;

        border-radius: 7px;

        text-align: center;

        background: #188038;

        color: white;

        font-size: 13px;

        font-weight: 700;

    }


    .danger-button {

        margin-top: 11px;

        padding: 10px;

        border-radius: 7px;

        text-align: center;

        background: #d93025;

        color: white;

        font-size: 13px;

        font-weight: 700;

    }


    .formula {

        margin: 14px 20px;

        padding: 12px;

        background: #f8f9fa;

        border-radius: 8px;

        font-family: monospace;

        font-size: 11px;

        line-height: 1.7;

    }


    .footer {

        padding: 12px 20px;

        background: #f8f9fa;

        font-size: 10px;

        line-height: 1.5;

        color: #777;

    }

    </style>

    """


    m.get_root().header.add_child(

        folium.Element(
            css
        )

    )


    # --------------------------------------------------------
    # PANEL HTML
    # --------------------------------------------------------

    panel = f"""

    <div class="giri-panel">


        <div class="header">

            <div class="title">

                GIRI-RAKSHAK

            </div>

            <div class="subtitle">

                LANDSLIDE-AWARE
                EMERGENCY NAVIGATION

            </div>

        </div>


        <div class="location">

            <div class="label">

                CURRENT REGION

            </div>

            <div class="value">

                {cfg["district"]},
                {cfg["state"]}

            </div>

        </div>


        <div class="alert">

            <div class="alert-title">

                PRIMARY ROUTE HAZARD

            </div>

            <div style="
            margin-top:7px;
            font-size:12px;
            line-height:1.6;
            ">

                <b>Road:</b>
                {route_a_name}<br>

                <b>Active status:</b>
                BLOCKED<br>

                <b>Traffic:</b>
                HEAVY<br>

                <b>Historical evidence:</b>
                {hist_a} event(s)

            </div>

        </div>


        <!-- ROUTE B -->

        <div class="route safe">

            <div class="route-title">

                TAKE ROUTE B

            </div>

            <div style="
            color:#188038;
            font-size:12px;
            font-weight:600;
            margin-top:3px;
            ">

                RECOMMENDED SAFE ALTERNATIVE

            </div>


            <div class="route-info">

                <b>State:</b>
                {cfg["state"]}<br>

                <b>District:</b>
                {cfg["district"]}<br>

                <b>Road:</b>
                {route_b_name}<br>

                <b>Distance:</b>
                {route_b["distance_km"]:.2f} km<br>

                <b>ETA:</b>
                {route_b["duration_min"]:.1f} min<br>

                <b>RF susceptibility:</b>
                {s_b:.4f}<br>

                <b>Dynamic trigger:</b>
                {trigger:.4f}<br>

                <b>Risk score:</b>
                {risk_b:.4f}<br>

                <b>Historical events:</b>
                {hist_b}<br>

                <b>Historical penalty:</b>
                {cost_b["historical_penalty"]:.2f}<br>

                <b>Dynamic penalty:</b>
                {cost_b["dynamic_penalty"]:.2f}<br>

                <b>Route cost:</b>
                {cost_b["cost"]:.2f}

            </div>


            <div class="safe-button">

                ✓ TAKE THIS ROAD

            </div>

        </div>


        <!-- ROUTE A -->

        <div class="route danger">

            <div class="route-title">

                DO NOT TAKE ROUTE A

            </div>

            <div style="
            color:#d93025;
            font-size:12px;
            font-weight:600;
            margin-top:3px;
            ">

                HAZARD CORRIDOR

            </div>


            <div class="route-info">

                <b>Road:</b>
                {route_a_name}<br>

                <b>Distance:</b>
                {route_a["distance_km"]:.2f} km<br>

                <b>ETA:</b>
                {route_a["duration_min"] + 18:.1f} min<br>

                <b>RF susceptibility:</b>
                {s_a:.4f}<br>

                <b>Dynamic trigger:</b>
                {trigger:.4f}<br>

                <b>Risk score:</b>
                {risk_a:.4f}<br>

                <b>Historical events:</b>
                {hist_a}<br>

                <b>Historical penalty:</b>
                {cost_a["historical_penalty"]:.2f}<br>

                <b>Traffic:</b>
                HEAVY<br>

                <b>Active incident:</b>
                BLOCKED<br>

                <b>Route cost:</b>
                INFINITY

            </div>


            <div class="danger-button">

                ✕ AVOID THIS ROAD

            </div>

        </div>


        <!-- EQUATION -->

        <div class="formula">

            Risk = 0.6 × S + 0.4 × T

            <br><br>

            Cost = Tbase ×
            (1 + 0.6 × S) ×
            Pdynamic ×
            Phistorical ×
            Pincident

            <br><br>

            BLOCKED → Cost = ∞

        </div>


        <div class="footer">

            <b>DEMO MODE</b><br>

            Road geometry and road names:
            real OSM-based routing.

            <br>

            RF susceptibility:
            real Giri-Rakshak model output.

            <br>

            Historical landslide points:
            real historical inventory.

            <br>

            Active blockage and traffic:
            synthetic demonstration inputs.

        </div>


    </div>

    """


    m.get_root().html.add_child(

        folium.Element(
            panel
        )

    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    safe_name = (
        region_name
        .lower()
        .replace(" ", "_")
    )


    output_file = (
        OUTPUT_DIR
        /
        f"giri_rakshak_{safe_name}_gps.html"
    )


    m.save(
        output_file
    )


    print()

    print(
        "HTML CREATED:"
    )

    print(
        output_file
    )

    print()

    print(
        "FINAL DECISION: "
        "TAKE ROUTE B"
    )


# ============================================================
# RUN ALL FOUR
# ============================================================

if __name__ == "__main__":

    print()

    print("=" * 75)

    print(
        "GIRI-RAKSHAK — "
        "FOUR REGION GPS DEMONSTRATION"
    )

    print("=" * 75)


    for region_name, cfg in REGIONS.items():

        try:

            build_region(
                region_name,
                cfg
            )

        except Exception as e:

            print()

            print(
                f"❌ {region_name} failed:"
            )

            print(
                str(e)
            )


    print()

    print("=" * 75)

    print(
        "ALL FOUR REGION DEMOS COMPLETE"
    )

    print("=" * 75)

    print()

    print(
        "Open the generated HTML files "
        "inside pilot_route_data."
    )