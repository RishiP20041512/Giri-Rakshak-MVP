from pathlib import Path
import math
import json
import requests

import folium
import geopandas as gpd
import numpy as np
import rasterio

from shapely.geometry import Point
from rasterio.warp import transform


# ============================================================
# GIRI-RAKSHAK
# OFFICIAL GPS + LANDSLIDE SAFE ROUTING DEMO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ROAD_FILE = (
    BASE_DIR
    / "pilot_route_data"
    / "pilot_roads_dynamic_cost.gpkg"
)

ROAD_LAYER = "pilot_roads_dynamic_cost"

SUSCEPTIBILITY_RASTER = (
    BASE_DIR
    / "processed"
    / "step70_8factor_susceptibility_probability.tif"
)

OUTPUT = (
    BASE_DIR
    / "pilot_route_data"
    / "giri_rakshak_final_gps_demo.html"
)


# ============================================================
# LOCATION
# ============================================================

STATE = "Meghalaya"
DISTRICT = "East Khasi Hills"

START = [25.5788, 91.8933]
DESTINATION = [25.6200, 91.8827]


# ============================================================
# REAL DYNAMIC TRIGGER
# ============================================================

REAL_DYNAMIC_TRIGGER = 0.1108

STATIC_WEIGHT = 0.60
DYNAMIC_WEIGHT = 0.40


# ============================================================
# DEMO TRAFFIC
# ============================================================

TRAFFIC_DELAY_MIN = 18


# ============================================================
# SYNTHETIC DEMO INCIDENTS
# ============================================================

DEMO_INCIDENTS = [
    {
        "id": "DEMO-LS-001",
        "severity": "HIGH",
        "type": "Early landslide movement",
    },
    {
        "id": "DEMO-LS-002",
        "severity": "HIGH",
        "type": "Slope failure warning",
    },
    {
        "id": "DEMO-LS-003",
        "severity": "CRITICAL",
        "type": "Early ground movement",
    },
]


print("=" * 75)
print("GIRI-RAKSHAK — OFFICIAL GPS SAFE ROUTING DEMO")
print("=" * 75)

print()
print("State:", STATE)
print("District:", DISTRICT)
print("Start GPS:", START)
print("Destination:", DESTINATION)


# ============================================================
# LOAD REAL OSM ROAD DATA
# ============================================================

print()
print("Loading real OSM road network...")

roads = gpd.read_file(
    ROAD_FILE,
    layer=ROAD_LAYER,
)

print(
    f"Road segments: {len(roads):,}"
)


roads_projected = roads.to_crs(
    "EPSG:6933"
)


# ============================================================
# ROAD NAME
# ============================================================

def nearest_road(lat, lon):

    point = gpd.GeoSeries(
        [Point(lon, lat)],
        crs="EPSG:4326",
    ).to_crs(
        "EPSG:6933"
    ).iloc[0]

    distances = (
        roads_projected.geometry
        .distance(point)
    )

    idx = distances.idxmin()

    row = roads.loc[idx]

    name = row.get("name", "")
    ref = row.get("ref", "")
    highway = row.get("highway", "")
    osm_id = row.get("osm_id", "")

    if (
        name is None
        or str(name).strip() == ""
        or str(name).lower() == "nan"
    ):
        name = ""

    if ref is None:
        ref = ""

    if highway is None:
        highway = ""

    if osm_id is None:
        osm_id = ""

    if not name and ref:
        name = str(ref)

    if not name and highway:
        name = (
            str(highway)
            .replace("_", " ")
            .title()
        )

    if not name:
        name = "Unnamed OSM road"

    return {
        "name": str(name),
        "ref": str(ref),
        "highway": str(highway),
        "osm_id": str(osm_id),
        "distance_m": float(
            distances.loc[idx]
        ),
    }


start_road = nearest_road(
    START[0],
    START[1],
)

destination_road = nearest_road(
    DESTINATION[0],
    DESTINATION[1],
)


# ============================================================
# REAL OSM ROUTING
# ============================================================

def get_osrm_route(points):

    coordinates = ";".join(
        f"{lon},{lat}"
        for lat, lon in points
    )

    url = (
        "https://router.project-osrm.org/"
        "route/v1/driving/"
        + coordinates
    )

    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
    }

    print()
    print("Requesting real road route...")

    response = requests.get(
        url,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "Ok":
        raise RuntimeError(
            "OSRM could not find a driving route."
        )

    route = data["routes"][0]

    coordinates = (
        route["geometry"]["coordinates"]
    )

    geometry = [
        [lat, lon]
        for lon, lat in coordinates
    ]

    return {
        "geometry": geometry,
        "distance_km":
            route["distance"] / 1000,
        "duration_min":
            route["duration"] / 60,
        "raw": route,
    }


# ============================================================
# ROUTE A
# ============================================================

print()
print("Calculating primary route...")

route_a = get_osrm_route(
    [
        START,
        DESTINATION,
    ]
)


# ============================================================
# ROUTE B
# ============================================================

ALTERNATIVE_WAYPOINT = [
    25.6000,
    91.9000,
]

print()
print("Calculating alternative route...")

route_b = get_osrm_route(
    [
        START,
        ALTERNATIVE_WAYPOINT,
        DESTINATION,
    ]
)


# ============================================================
# EXTRACT ACTUAL ROAD NAMES FROM OSRM
# ============================================================

def get_route_road_names(route):

    names = []

    for leg in route["raw"].get(
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
                    and value not in names
                ):

                    names.append(
                        value
                    )

    return names


route_a_roads = get_route_road_names(
    route_a
)

route_b_roads = get_route_road_names(
    route_b
)


def route_name(
    names,
    fallback,
):

    if names:

        return " → ".join(
            names[:5]
        )

    return fallback


route_a_road_display = route_name(
    route_a_roads,
    start_road["name"],
)

route_b_road_display = route_name(
    route_b_roads,
    destination_road["name"],
)


print()
print("ROUTE A ROAD(S):")
print(route_a_road_display)

print()
print("ROUTE B ROAD(S):")
print(route_b_road_display)


# ============================================================
# REAL RF SUSCEPTIBILITY
# ============================================================

def sample_susceptibility(
    geometry
):

    values = []

    with rasterio.open(
        SUSCEPTIBILITY_RASTER
    ) as src:

        raster = src.read(1)

        for lat, lon in geometry:

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
                    or row >= raster.shape[0]
                    or col < 0
                    or col >= raster.shape[1]
                ):
                    continue

                value = raster[
                    row,
                    col
                ]

                if (
                    np.isfinite(value)
                    and value != src.nodata
                    and 0 <= value <= 1
                ):

                    values.append(
                        float(value)
                    )

            except Exception:
                continue

    if not values:

        raise RuntimeError(
            "No valid susceptibility "
            "values found on route."
        )

    return {
        "mean": float(
            np.mean(values)
        ),

        "max": float(
            np.max(values)
        ),

        "samples": len(values),
    }


print()
print(
    "Sampling REAL RF susceptibility..."
)

route_a_s = sample_susceptibility(
    route_a["geometry"]
)

route_b_s = sample_susceptibility(
    route_b["geometry"]
)


# ============================================================
# RISK EQUATION
#
# Risk = 0.6 × S + 0.4 × T
# ============================================================

route_a_risk = (
    STATIC_WEIGHT
    * route_a_s["mean"]
    +
    DYNAMIC_WEIGHT
    * REAL_DYNAMIC_TRIGGER
)

route_b_risk = (
    STATIC_WEIGHT
    * route_b_s["mean"]
    +
    DYNAMIC_WEIGHT
    * REAL_DYNAMIC_TRIGGER
)


# ============================================================
# DYNAMIC PENALTY
# ============================================================

def dynamic_penalty(trigger):

    if trigger < 0.35:
        return 1.0

    elif trigger < 0.65:
        return 1.3

    elif trigger < 0.75:
        return 2.0

    else:
        return 2.5


P_DYNAMIC = dynamic_penalty(
    REAL_DYNAMIC_TRIGGER
)


# ============================================================
# ROUTE COST
#
# C = Tbase × (1 + 0.6 × LSI)
#     × P_dynamic × P_incident
# ============================================================

def calculate_cost(
    base_time,
    lsi,
    p_dynamic,
    p_incident,
):

    return (
        base_time
        *
        (
            1
            + STATIC_WEIGHT * lsi
        )
        *
        p_dynamic
        *
        p_incident
    )


# Route A has synthetic active blockage.
route_a_cost = calculate_cost(
    route_a["duration_min"],
    route_a_s["mean"],
    P_DYNAMIC,
    float("inf"),
)


# Route B is available.
route_b_cost = calculate_cost(
    route_b["duration_min"],
    route_b_s["mean"],
    P_DYNAMIC,
    1.0,
)


# ============================================================
# ETA
# ============================================================

route_a_eta = (
    route_a["duration_min"]
    + TRAFFIC_DELAY_MIN
)

route_b_eta = (
    route_b["duration_min"]
)


# ============================================================
# TERMINAL RESULTS
# ============================================================

print()
print("=" * 75)
print("REAL GIRI-RAKSHAK ROUTE DECISION")
print("=" * 75)

print()

print(
    "Dynamic trigger:",
    f"{REAL_DYNAMIC_TRIGGER:.4f}"
)

print(
    "Dynamic penalty:",
    f"{P_DYNAMIC:.2f}"
)

print()

print("ROUTE A")
print(
    "Road:",
    route_a_road_display
)

print(
    "Distance:",
    f"{route_a['distance_km']:.2f} km"
)

print(
    "RF susceptibility:",
    f"{route_a_s['mean']:.4f}"
)

print(
    "Maximum LSI:",
    f"{route_a_s['max']:.4f}"
)

print(
    "RiskScore:",
    f"{route_a_risk:.4f}"
)

print(
    "ETA with traffic:",
    f"{route_a_eta:.1f} min"
)

print(
    "Route cost:",
    "INFINITY — BLOCKED"
)

print(
    "Decision:",
    "DO NOT TAKE"
)

print()

print("ROUTE B")
print(
    "Road:",
    route_b_road_display
)

print(
    "Distance:",
    f"{route_b['distance_km']:.2f} km"
)

print(
    "RF susceptibility:",
    f"{route_b_s['mean']:.4f}"
)

print(
    "Maximum LSI:",
    f"{route_b_s['max']:.4f}"
)

print(
    "RiskScore:",
    f"{route_b_risk:.4f}"
)

print(
    "ETA:",
    f"{route_b_eta:.1f} min"
)

print(
    "Route cost:",
    f"{route_b_cost:.2f}"
)

print(
    "Decision:",
    "TAKE THIS ROUTE"
)

print()
print("=" * 75)


# ============================================================
# MAP
# ============================================================

center = [
    (
        START[0]
        + DESTINATION[0]
    ) / 2,

    (
        START[1]
        + DESTINATION[1]
    ) / 2,
]


m = folium.Map(
    location=center,
    zoom_start=13,
    tiles="OpenStreetMap",
    control_scale=True,
)


# ============================================================
# START
# ============================================================

folium.Marker(
    START,

    tooltip="CURRENT GPS LOCATION",

    popup=f"""
    <b>GIRI-RAKSHAK GPS</b><br><br>

    State: {STATE}<br>

    District: {DISTRICT}<br>

    Current road:
    {start_road["name"]}<br>

    Road type:
    {start_road["highway"]}<br>

    OSM ID:
    {start_road["osm_id"]}
    """,

    icon=folium.Icon(
        color="blue",
        icon="car",
        prefix="fa",
    ),

).add_to(m)


# ============================================================
# DESTINATION
# ============================================================

folium.Marker(
    DESTINATION,

    tooltip="DESTINATION",

    popup=f"""
    <b>DESTINATION</b><br><br>

    State: {STATE}<br>

    District: {DISTRICT}<br>

    Road:
    {destination_road["name"]}<br>

    Road type:
    {destination_road["highway"]}
    """,

    icon=folium.Icon(
        color="green",
        icon="flag",
        prefix="fa",
    ),

).add_to(m)


# ============================================================
# ROUTE A
# ============================================================

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


# ============================================================
# ROUTE B
# ============================================================

folium.PolyLine(
    route_b["geometry"],

    color="#188038",

    weight=8,

    opacity=0.95,

    tooltip=(
        "ROUTE B — TAKE THIS ROUTE"
    ),

).add_to(m)


# ============================================================
# LANDSLIDES ON REAL ROUTE A
# ============================================================

def route_point(
    geometry,
    fraction,
):

    index = int(
        fraction
        *
        (len(geometry) - 1)
    )

    index = max(
        0,
        min(
            index,
            len(geometry) - 1
        )
    )

    return geometry[index]


fractions = [
    0.32,
    0.53,
    0.73,
]


for incident, fraction in zip(
    DEMO_INCIDENTS,
    fractions,
):

    position = route_point(
        route_a["geometry"],
        fraction,
    )

    lat, lon = position

    folium.CircleMarker(

        [lat, lon],

        radius=10,

        color="#b3261e",

        fill=True,

        fill_color="#d93025",

        fill_opacity=0.95,

        popup=f"""
        <b>LANDSLIDE EARLY WARNING</b>
        <br><br>

        Incident:
        {incident["id"]}<br>

        State:
        {STATE}<br>

        District:
        {DISTRICT}<br>

        Affected road:
        {route_a_road_display}<br>

        Type:
        {incident["type"]}<br>

        Severity:
        {incident["severity"]}<br>

        Status:
        <b>ROAD BLOCKED</b>

        <br><br>

        <b>ROUTING ACTION</b><br>

        Do not continue on Route A.
        Use Route B.

        <br><br>

        <small>
        SYNTHETIC DEMONSTRATION INCIDENT
        </small>
        """,

        tooltip=(
            f"⚠ {incident['severity']} "
            "LANDSLIDE"
        ),

    ).add_to(m)


# ============================================================
# TRAFFIC JAM
# ============================================================

traffic_position = route_point(
    route_a["geometry"],
    0.62,
)


folium.CircleMarker(

    traffic_position,

    radius=9,

    color="#e37400",

    fill=True,

    fill_color="#f9ab00",

    fill_opacity=0.95,

    popup=f"""
    <b>TRAFFIC CONDITION</b>
    <br><br>

    State:
    {STATE}<br>

    District:
    {DISTRICT}<br>

    Affected road:
    {route_a_road_display}<br>

    Severity:
    HEAVY<br>

    Estimated delay:
    +{TRAFFIC_DELAY_MIN} minutes

    <br><br>

    <small>
    SYNTHETIC DEMONSTRATION TRAFFIC
    </small>
    """,

    tooltip="HEAVY TRAFFIC",

).add_to(m)


# ============================================================
# PROFESSIONAL CSS
# ============================================================

css = """

<style>

.giri-panel {

position: fixed;

top: 18px;

left: 60px;

z-index: 9999;

width: 410px;

background: white;

border-radius: 12px;

box-shadow:
0 4px 20px rgba(0,0,0,.28);

font-family:
Arial, Helvetica, sans-serif;

overflow: hidden;

}


.header {

padding: 18px 20px;

border-bottom:
1px solid #e5e5e5;

}


.title {

font-size: 21px;

font-weight: 700;

color: #202124;

}


.subtitle {

font-size: 11px;

color: #6b7280;

margin-top: 4px;

letter-spacing: .4px;

}


.section {

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

color: #202124;

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


.route-sub {

font-size: 12px;

font-weight: 600;

margin-top: 3px;

}


.info {

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


.footer {

padding: 10px 20px;

background: #f8f9fa;

border-top:
1px solid #eeeeee;

font-size: 10px;

line-height: 1.5;

color: #777;

}

</style>

"""

m.get_root().header.add_child(
    folium.Element(css)
)


# ============================================================
# PANEL
# ============================================================

panel = f"""

<div class="giri-panel">


<div class="header">

<div class="title">

GIRI-RAKSHAK

</div>

<div class="subtitle">

LANDSLIDE-AWARE EMERGENCY NAVIGATION

</div>

</div>


<div class="section">

<div class="label">

CURRENT LOCATION

</div>

<div class="value">

{DISTRICT}, {STATE}

</div>

<div style="
font-size:12px;
margin-top:6px;
color:#555;
">

Current road:
<b>{start_road["name"]}</b>

</div>

</div>


<div class="section">

<div class="label">

DESTINATION

</div>

<div class="value">

{DISTRICT}, {STATE}

</div>

<div style="
font-size:12px;
margin-top:6px;
color:#555;
">

Road:
<b>{route_b_road_display}</b>

</div>

</div>


<div class="alert">

<div class="alert-title">

PRIMARY ROUTE HAZARD ALERT

</div>

<div style="
font-size:12px;
margin-top:7px;
color:#444;
line-height:1.6;
">

<b>State:</b>
{STATE}<br>

<b>District:</b>
{DISTRICT}<br>

<b>Affected corridor:</b>
{route_a_road_display}<br>

<b>Hazard:</b>
Landslide / slope movement<br>

<b>Traffic:</b>
Heavy congestion<br>

<b>Routing status:</b>
PRIMARY ROUTE BLOCKED

</div>

</div>


<!-- ROUTE B -->

<div class="route safe">

<div class="route-title">

TAKE ROUTE B

</div>

<div class="route-sub"
style="color:#188038;">

RECOMMENDED SAFE ALTERNATIVE

</div>

<div class="info">

<b>State:</b>
{STATE}<br>

<b>District:</b>
{DISTRICT}<br>

<b>Road corridor:</b>
{route_b_road_display}<br>

<b>Distance:</b>
{route_b["distance_km"]:.2f} km<br>

<b>ETA:</b>
{route_b_eta:.1f} min<br>

<b>RF susceptibility:</b>
{route_b_s["mean"]:.3f}<br>

<b>Dynamic trigger:</b>
{REAL_DYNAMIC_TRIGGER:.3f}<br>

<b>Risk score:</b>
{route_b_risk:.3f}<br>

<b>Route cost:</b>
{route_b_cost:.2f}<br>

<b>Network:</b>
REAL OSM ROAD ROUTE

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

<div class="route-sub"
style="color:#d93025;">

HAZARD CORRIDOR

</div>

<div class="info">

<b>State:</b>
{STATE}<br>

<b>District:</b>
{DISTRICT}<br>

<b>Road corridor:</b>
{route_a_road_display}<br>

<b>Distance:</b>
{route_a["distance_km"]:.2f} km<br>

<b>ETA:</b>
{route_a_eta:.1f} min<br>

<b>RF susceptibility:</b>
{route_a_s["mean"]:.3f}<br>

<b>Dynamic trigger:</b>
{REAL_DYNAMIC_TRIGGER:.3f}<br>

<b>Risk score:</b>
{route_a_risk:.3f}<br>

<b>Traffic delay:</b>
+{TRAFFIC_DELAY_MIN} min<br>

<b>Route cost:</b>
INFINITY<br>

<b>Status:</b>
BLOCKED / AVOID

</div>

<div class="danger-button">

✕ AVOID THIS ROAD

</div>

</div>


<div class="footer">

<b>Decision:</b>
Risk = 0.6 × susceptibility +
0.4 × dynamic trigger.

Route cost incorporates travel time,
susceptibility, dynamic penalty and
incident status.

<br><br>

DEMO MODE — Landslide and traffic
events are synthetic. Road geometry
and road metadata are based on
real OSM data.

</div>


</div>

"""

m.get_root().html.add_child(
    folium.Element(panel)
)


# ============================================================
# BOTTOM STATUS
# ============================================================

status = """

<div id="nav-status"

style="
position:fixed;

bottom:22px;

left:50%;

transform:translateX(-50%);

z-index:9999;

background:white;

padding:12px 26px;

border-radius:24px;

box-shadow:
0 3px 16px rgba(0,0,0,.25);

font-family:Arial;

font-size:14px;

font-weight:700;

color:#188038;
">

✓ SAFE ALTERNATIVE ROUTE AVAILABLE

</div>

"""

m.get_root().html.add_child(
    folium.Element(status)
)


# ============================================================
# GPS ANIMATION
# ============================================================

route_a_js = json.dumps(
    route_a["geometry"]
)

route_b_js = json.dumps(
    route_b["geometry"]
)


javascript = f"""

<script>

var map = {m.get_name()};

var routeA = {route_a_js};

var routeB = {route_b_js};

var currentRoute = routeA;

var car = null;

var index = 0;

var rerouted = false;


// ==========================================================
// GPS VEHICLE
// ==========================================================

function createCar(position) {{

    car = L.circleMarker(

        position,

        {{

            radius: 8,

            color: "#1557b0",

            fillColor: "#1a73e8",

            fillOpacity: 1,

            weight: 3

        }}

    ).addTo(map);

}}


// ==========================================================
// VEHICLE MOVEMENT
// ==========================================================

function moveCar() {{

    if (!car) {{

        createCar(
            currentRoute[0]
        );

    }}


    if (
        index >=
        currentRoute.length
    ) {{

        index = 0;

    }}


    var position =
        currentRoute[index];


    car.setLatLng(
        position
    );


    // ------------------------------------------------------
    // LANDSLIDE WARNING
    // ------------------------------------------------------

    if (
        !rerouted
        &&
        index >
        currentRoute.length * 0.25
        &&
        index <
        currentRoute.length * 0.55
    ) {{

        document.getElementById(
            "nav-status"
        ).innerHTML =
            "⚠ LANDSLIDE AHEAD — SLOW DOWN";

        document.getElementById(
            "nav-status"
        ).style.color =
            "#b3261e";

    }}


    // ------------------------------------------------------
    // REROUTE
    // ------------------------------------------------------

    if (
        !rerouted
        &&
        index >=
        currentRoute.length * 0.55
    ) {{

        document.getElementById(
            "nav-status"
        ).innerHTML =
            "↻ REROUTING — PRIMARY ROAD BLOCKED";

        document.getElementById(
            "nav-status"
        ).style.color =
            "#d93025";


        setTimeout(

            function() {{

                rerouted = true;

                currentRoute = routeB;

                index = 0;


                document.getElementById(
                    "nav-status"
                ).innerHTML =
                    "✓ TAKE ROUTE B — SAFER ALTERNATIVE";

                document.getElementById(
                    "nav-status"
                ).style.color =
                    "#188038";

            }},

            2500

        );

        return;

    }}


    // ------------------------------------------------------
    // SAFE ROUTE
    // ------------------------------------------------------

    if (rerouted) {{

        document.getElementById(
            "nav-status"
        ).innerHTML =
            "✓ ROUTE B ACTIVE — SAFE ALTERNATIVE";

        document.getElementById(
            "nav-status"
        ).style.color =
            "#188038";

    }}


    // ------------------------------------------------------
    // DESTINATION
    // ------------------------------------------------------

    if (
        rerouted
        &&
        index ===
        currentRoute.length - 1
    ) {{

        document.getElementById(
            "nav-status"
        ).innerHTML =
            "✓ DESTINATION REACHED";

    }}


    index++;

}}


// ==========================================================
// START GPS
// ==========================================================

setTimeout(

    function() {{

        createCar(
            routeA[0]
        );

        setInterval(
            moveCar,
            1000
        );

    }},

    1000

);

</script>

"""

m.get_root().html.add_child(
    folium.Element(javascript)
)


# ============================================================
# SAVE
# ============================================================

m.save(
    OUTPUT
)


# ============================================================
# FINAL
# ============================================================

print()

print("=" * 75)

print(
    "✅ FINAL GIRI-RAKSHAK GPS DEMO CREATED"
)

print("=" * 75)

print()

print("HTML:")
print(OUTPUT)

print()

print("State:", STATE)
print("District:", DISTRICT)

print()

print(
    "Current road:",
    start_road["name"]
)

print(
    "Destination road:",
    route_b_road_display
)
print()

print(
    "Route A road:",
    route_a_road_display
)

print(
    "Route B road:",
    route_b_road_display
)

print()

print(
    f"Route A: "
    f"{route_a['distance_km']:.2f} km"
)

print(
    f"Route B: "
    f"{route_b['distance_km']:.2f} km"
)

print()

print(
    f"Route A risk: "
    f"{route_a_risk:.3f}"
)

print(
    f"Route B risk: "
    f"{route_b_risk:.3f}"
)

print()

print(
    "Route A cost: INFINITY — BLOCKED"
)

print(
    f"Route B cost: "
    f"{route_b_cost:.2f}"
)

print()

print(
    "DECISION: TAKE ROUTE B"
)

print()

print("=" * 75)