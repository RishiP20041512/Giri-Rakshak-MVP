# ============================================================
# GIRI-RAKSHAK — SIKKIM GPS LANDSLIDE ROUTING DEMO
# ============================================================
#
# RED  = Blocked / unsafe primary route
# GREEN = Recommended alternative route
#
# REAL LANDSLIDE EVENTS:
# 1. 05–06 Sep 2026 — Naga–Tosa / Theng Tunnel
# 2. 24 May 2020 — Kabi Village
# 3. 24 May 2020 — Dhare / Sitala
#
# DEMO LOCATION:
# Mangan -> Chungthang
#
# ============================================================

import os
import json
import math
import requests
import folium

from folium.plugins import AntPath

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "pilot_route_data"

OUTPUT_HTML = os.path.join(
    OUTPUT_DIR,
    "sikkim_giri_rakshak_gps.html"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# REAL SIKKIM LOCATIONS
# ============================================================

# Mangan
MANGAN = (
    27.50115,
    88.53553
)

# Chungthang
CHUNGTHANG = (
    27.60590,
    88.64469
)

# Phidang
PHIDANG = (
    27.411641,
    88.516016
)

# Sankalang / Sangkalang
SANKALANG = (
    27.509306,
    88.526417
)

# ============================================================
# REAL RECENT LANDSLIDE
# ============================================================

RECENT_LANDSLIDE = (
    27.599862,
    88.649398
)


# ============================================================
# REAL HISTORICAL LANDSLIDES FROM YOUR CSV
# ============================================================

KABI_VILLAGE = (
    27.393056,
    88.635278
)

DHARE_SITALA = (
    27.398667,
    88.623583
)


# ============================================================
# VALIDATED SIKKIM / MANGAN MODEL VALUES
# ============================================================

STATIC_SUSCEPTIBILITY = 0.8299

DYNAMIC_TRIGGER = 0.4363

RISK_SCORE = (
    0.60 * STATIC_SUSCEPTIBILITY
    +
    0.40 * DYNAMIC_TRIGGER
)

RISK_CLASS = "HIGH"


# ============================================================
# ROUTING SETTINGS
# ============================================================

OSRM_URL = (
    "https://router.project-osrm.org/route/v1/driving/"
)

REQUEST_TIMEOUT = 30


# ============================================================
# HELPER
# ============================================================

def reverse_coordinates(point):
    """
    Convert:
        (latitude, longitude)

    into:
        longitude,latitude

    for OSRM.
    """

    lat, lon = point

    return f"{lon},{lat}"


# ============================================================
# OSRM ROUTING
# ============================================================

def get_route(points):
    """
    Request a real driving route from OSRM.

    points:
        [(lat, lon), (lat, lon), ...]

    Returns:
        distance_km
        duration_min
        geometry
    """

    coordinates = ";".join(
        reverse_coordinates(p)
        for p in points
    )

    url = (
        OSRM_URL
        +
        coordinates
    )

    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true"
    }

    print()
    print("Requesting route:")
    print(url)

    response = requests.get(
        url,
        params=params,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    if data.get("code") != "Ok":
        raise RuntimeError(
            "OSRM routing failed: "
            + str(data)
        )

    if not data.get("routes"):
        raise RuntimeError(
            "No route returned by OSRM."
        )

    route = data["routes"][0]

    geometry = route["geometry"]["coordinates"]

    # OSRM gives:
    # [longitude, latitude]

    # Folium needs:
    # [latitude, longitude]

    folium_geometry = [
        [
            coord[1],
            coord[0]
        ]
        for coord in geometry
    ]

    distance_km = (
        route["distance"] / 1000.0
    )

    duration_min = (
        route["duration"] / 60.0
    )

    return {
        "geometry": folium_geometry,
        "distance_km": distance_km,
        "duration_min": duration_min
    }


# ============================================================
# ROUTE A
# ============================================================

def build_blocked_route():

    print()
    print("=" * 70)
    print("BUILDING ROUTE A — BLOCKED")
    print("=" * 70)

    # Route A deliberately follows the
    # Mangan -> recent landslide -> Chungthang corridor.

    route = get_route(
        [
            MANGAN,
            RECENT_LANDSLIDE,
            CHUNGTHANG
        ]
    )

    return route


# ============================================================
# ROUTE B
# ============================================================

def build_alternative_route():

    print()
    print("=" * 70)
    print("BUILDING ROUTE B — RECOMMENDED")
    print("=" * 70)

    # Official alternative corridor:
    #
    # Mangan
    #    ↓
    # Phidang
    #    ↓
    # Sankalang
    #    ↓
    # Chungthang

    route = get_route(
        [
            MANGAN,
            PHIDANG,
            SANKALANG,
            CHUNGTHANG
        ]
    )

    return route


# ============================================================
# FORMAT TIME
# ============================================================

def format_minutes(minutes):

    if minutes < 60:

        return f"{minutes:.0f} min"

    hours = int(minutes // 60)

    mins = int(round(minutes % 60))

    if mins == 60:

        hours += 1
        mins = 0

    return (
        f"{hours} hr {mins} min"
    )


# ============================================================
# HTML POPUP — ROUTE A
# ============================================================

def route_a_popup(route):

    return f"""
    <div style="
        font-family: Arial, sans-serif;
        width: 330px;
    ">

        <h2 style="
            margin-bottom: 5px;
            color: #d93025;
        ">
            ✕ ROUTE A — BLOCKED
        </h2>

        <hr>

        <b>Corridor</b><br>
        Mangan → Naga–Tosa /
        Theng Tunnel → Chungthang

        <br><br>

        <b>Status</b><br>

        <span style="
            color:#d93025;
            font-size:18px;
            font-weight:bold;
        ">
            BLOCKED
        </span>

        <br><br>

        <b>Reason</b><br>
        Recent landslide near the
        Naga–Tosa / Theng Tunnel corridor.

        <br><br>

        <b>Route distance</b><br>
        {route["distance_km"]:.2f} km

        <br><br>

        <b>Estimated travel time</b><br>
        {format_minutes(route["duration_min"])}

        <br><br>

        <div style="
            background:#fde8e7;
            padding:10px;
            border-radius:8px;
            color:#b42318;
            font-weight:bold;
        ">
            ⚠ DO NOT TAKE THIS ROUTE
        </div>

    </div>
    """


# ============================================================
# HTML POPUP — ROUTE B
# ============================================================

def route_b_popup(route):

    return f"""
    <div style="
        font-family: Arial, sans-serif;
        width: 330px;
    ">

        <h2 style="
            margin-bottom: 5px;
            color: #188038;
        ">
            ✓ ROUTE B — RECOMMENDED
        </h2>

        <hr>

        <b>Corridor</b><br>
        Mangan → Phidang →
        Sankalang → Chungthang

        <br><br>

        <b>Status</b><br>

        <span style="
            color:#188038;
            font-size:18px;
            font-weight:bold;
        ">
            ALTERNATIVE ROUTE
        </span>

        <br><br>

        <b>Route distance</b><br>
        {route["distance_km"]:.2f} km

        <br><br>

        <b>Estimated travel time</b><br>
        {format_minutes(route["duration_min"])}

        <br><br>

        <div style="
            background:#e8f5e9;
            padding:10px;
            border-radius:8px;
            color:#137333;
            font-weight:bold;
        ">
            ✓ USE THIS ROUTE
        </div>

    </div>
    """


# ============================================================
# CREATE MAP
# ============================================================

def create_map(route_a, route_b):

    print()
    print("=" * 70)
    print("CREATING SIKKIM GPS MAP")
    print("=" * 70)

    m = folium.Map(

        location=[
            27.555,
            88.595
        ],

        zoom_start=11,

        tiles=None,

        control_scale=True
    )


    # ========================================================
    # MAP TILES
    # ========================================================

    folium.TileLayer(
        "OpenStreetMap",
        name="Street Map",
        control=True
    ).add_to(m)

    folium.TileLayer(
        "CartoDB positron",
        name="Light Map",
        control=True
    ).add_to(m)


    # ========================================================
    # ROUTE LAYERS
    # ========================================================

    blocked_layer = folium.FeatureGroup(
        name="🔴 Blocked Route A",
        show=True
    )

    recommended_layer = folium.FeatureGroup(
        name="🟢 Recommended Route B",
        show=True
    )


    # ========================================================
    # ROUTE A — RED
    # ========================================================

    folium.PolyLine(

        locations=route_a["geometry"],

        color="#d93025",

        weight=9,

        opacity=0.95,

        dash_array="18,10",

        line_cap="round",

        line_join="round",

        tooltip=(
            "✕ ROUTE A — BLOCKED — DO NOT TAKE"
        ),

        popup=folium.Popup(
            route_a_popup(route_a),
            max_width=380
        )

    ).add_to(blocked_layer)


    # ========================================================
    # RED WARNING ZONE
    # ========================================================

    folium.Circle(

        location=RECENT_LANDSLIDE,

        radius=900,

        color="#d93025",

        weight=3,

        fill=True,

        fill_opacity=0.12,

        tooltip=(
            "⚠ LANDSLIDE HAZARD ZONE"
        )

    ).add_to(blocked_layer)


    # ========================================================
    # ROUTE B — GREEN
    # ========================================================

    folium.PolyLine(

        locations=route_b["geometry"],

        color="#188038",

        weight=9,

        opacity=0.95,

        line_cap="round",

        line_join="round",

        tooltip=(
            "✓ ROUTE B — RECOMMENDED"
        ),

        popup=folium.Popup(
            route_b_popup(route_b),
            max_width=380
        )

    ).add_to(recommended_layer)


    # ========================================================
    # ADD ROUTE LAYERS
    # ========================================================

    blocked_layer.add_to(m)

    recommended_layer.add_to(m)


    # ========================================================
    # START — MANGAN
    # ========================================================

    folium.Marker(

        location=MANGAN,

        tooltip="🚗 START — MANGAN",

        popup="""
        <div style="
            font-family:Arial;
            width:280px;
        ">

        <h3 style="color:#1565c0;">
            🚗 GIRI-RAKSHAK
        </h3>

        <b>Current location</b><br>
        Mangan, Sikkim

        <br><br>

        <b>Destination</b><br>
        Chungthang

        <br><br>

        <b>Routing system</b><br>
        Landslide-aware GPS

        </div>
        """,

        icon=folium.Icon(
            color="blue",
            icon="car",
            prefix="fa"
        )

    ).add_to(m)


    # ========================================================
    # DESTINATION — CHUNGTHANG
    # ========================================================

    folium.Marker(

        location=CHUNGTHANG,

        tooltip="🏁 DESTINATION — CHUNGTHANG",

        popup="""
        <div style="
            font-family:Arial;
            width:260px;
        ">

        <h3 style="color:#188038;">
            🏁 DESTINATION
        </h3>

        Chungthang<br>
        North Sikkim

        </div>
        """,

        icon=folium.Icon(
            color="green",
            icon="flag",
            prefix="fa"
        )

    ).add_to(m)


    # ========================================================
    # PHIDANG MARKER
    # ========================================================

    folium.CircleMarker(

        location=PHIDANG,

        radius=6,

        color="#188038",

        fill=True,

        fill_opacity=1,

        tooltip="Phidang — Alternative Corridor"

    ).add_to(m)


    # ========================================================
    # SANKALANG MARKER
    # ========================================================

    folium.CircleMarker(

        location=SANKALANG,

        radius=6,

        color="#188038",

        fill=True,

        fill_opacity=1,

        tooltip="Sankalang — Alternative Corridor"

    ).add_to(m)


    # ========================================================
    # RECENT LANDSLIDE MARKER
    # ========================================================

    folium.Marker(

        location=RECENT_LANDSLIDE,

        tooltip=(
            "⚠ 05–06 SEP 2026 — RECENT LANDSLIDE"
        ),

        popup="""
        <div style="
            font-family:Arial;
            width:340px;
        ">

        <h2 style="color:#d93025;">
            ⚠ RECENT LANDSLIDE
        </h2>

        <hr>

        <b>Date</b><br>
        05–06 September 2026

        <br><br>

        <b>Location</b><br>
        Naga–Tosa Road /
        Theng Tunnel area

        <br><br>

        <b>District</b><br>
        Mangan, Sikkim

        <br><br>

        <b>Routing decision</b><br>

        <span style="
            color:#d93025;
            font-size:18px;
            font-weight:bold;
        ">
            ROUTE A — AVOID
        </span>

        <br><br>

        <span style="
            color:#188038;
            font-size:18px;
            font-weight:bold;
        ">
            ROUTE B — USE ALTERNATIVE
        </span>

        </div>
        """,

        icon=folium.Icon(
            color="red",
            icon="warning-sign"
        )

    ).add_to(m)


    # ========================================================
    # HISTORICAL LANDSLIDE LAYER
    # ========================================================

    historical_layer = folium.FeatureGroup(
        name="🟠 Historical Landslides",
        show=False
    )


    # --------------------------------------------------------
    # KABI
    # --------------------------------------------------------

    folium.Marker(

        location=KABI_VILLAGE,

        tooltip="🟠 24 May 2020 — Kabi Village",

        popup="""
        <div style="
            font-family:Arial;
            width:300px;
        ">

        <h3 style="color:#e37400;">
            🟠 HISTORICAL LANDSLIDE
        </h3>

        <b>Date:</b><br>
        24 May 2020

        <br><br>

        <b>Location:</b><br>
        Kabi Village, North Sikkim

        <br><br>

        Historical event — not a
        current blockage.

        </div>
        """,

        icon=folium.Icon(
            color="orange",
            icon="exclamation-sign"
        )

    ).add_to(historical_layer)


    # --------------------------------------------------------
    # DHARE / SITALA
    # --------------------------------------------------------

    folium.Marker(

        location=DHARE_SITALA,

        tooltip="🟠 24 May 2020 — Dhare / Sitala",

        popup="""
        <div style="
            font-family:Arial;
            width:300px;
        ">

        <h3 style="color:#e37400;">
            🟠 HISTORICAL LANDSLIDE
        </h3>

        <b>Date:</b><br>
        24 May 2020

        <br><br>

        <b>Location:</b><br>
        Dhare / Sitala,
        Kabi Village, North Sikkim

        <br><br>

        Historical event — not a
        current blockage.

        </div>
        """,

        icon=folium.Icon(
            color="orange",
            icon="exclamation-sign"
        )

    ).add_to(historical_layer)


    historical_layer.add_to(m)


    # ========================================================
    # GPS CAR MARKER
    # ========================================================

    car = folium.Marker(

        location=MANGAN,

        tooltip="🚗 LIVE GPS",

        icon=folium.Icon(
            color="blue",
            icon="car",
            prefix="fa"
        )

    )

    car.add_to(m)


    # ========================================================
    # GPS CAR ANIMATION
    # ========================================================

    route_points = route_b["geometry"]

    # Limit animation points so browser remains smooth.

    if len(route_points) > 500:

        step = math.ceil(
            len(route_points) / 500
        )

        animation_points = (
            route_points[::step]
        )

        # Always include destination.

        if animation_points[-1] != route_points[-1]:

            animation_points.append(
                route_points[-1]
            )

    else:

        animation_points = route_points


    animation_json = json.dumps(
        animation_points
    )


    # ========================================================
    # JAVASCRIPT GPS ANIMATION
    # ========================================================

    animation_script = f"""
    <script>

    document.addEventListener(
        "DOMContentLoaded",
        function() {{

            var route =
                {animation_json};

            var marker =
                document.querySelector(
                    '.leaflet-marker-icon'
                );

            var i = 0;

            function moveCar() {{

                if (i >= route.length) {{
                    i = 0;
                }}

                var point = route[i];

                // Find Leaflet marker objects
                // through the map instance.

                if (
                    typeof map !== "undefined"
                    &&
                    map
                ) {{
                    // handled by Leaflet below
                }}

                i++;

                setTimeout(
                    moveCar,
                    80
                );
            }}

            // Animation is also implemented
            // through a Leaflet marker object.

        }}
    );

    </script>
    """


    # ========================================================
    # MORE RELIABLE LEAFLET GPS ANIMATION
    # ========================================================

    gps_script = f"""

    <script>

    var gpsRoute = {animation_json};

    var gpsIndex = 0;

    var gpsMarker = null;

    function startGPSAnimation() {{

        if (
            typeof map === "undefined"
        ) {{
            setTimeout(
                startGPSAnimation,
                500
            );

            return;
        }}

        gpsMarker = L.marker(
            gpsRoute[0],
            {{
                icon: L.divIcon({{
                    className: "gps-car-icon",
                    html: `
                        <div style="
                            width:34px;
                            height:34px;
                            border-radius:50%;
                            background:#1a73e8;
                            border:4px solid white;
                            box-shadow:
                                0 2px 8px
                                rgba(0,0,0,.35);
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            font-size:18px;
                        ">
                            🚗
                        </div>
                    `,
                    iconSize:[42,42],
                    iconAnchor:[21,21]
                }})
            }}
        ).addTo(map);

        function moveGPS() {{

            if (
                gpsIndex >= gpsRoute.length
            ) {{
                gpsIndex = 0;
            }}

            gpsMarker.setLatLng(
                gpsRoute[gpsIndex]
            );

            gpsIndex++;

            setTimeout(
                moveGPS,
                70
            );
        }}

        moveGPS();
    }}

    setTimeout(
        startGPSAnimation,
        1000
    );

    </script>

    """


    m.get_root().html.add_child(
        folium.Element(
            animation_script
            +
            gps_script
        )
    )


    # ========================================================
    # INFORMATION PANEL
    # ========================================================

    panel_html = f"""

    <div style="
        position: fixed;
        top: 20px;
        left: 60px;
        z-index: 9999;

        background: white;

        padding: 16px 20px;

        border-radius: 14px;

        box-shadow:
            0 3px 15px
            rgba(0,0,0,.25);

        font-family: Arial;

        width: 330px;
    ">

        <div style="
            font-size:22px;
            font-weight:bold;
            color:#202124;
        ">
            🏔 GIRI-RAKSHAK
        </div>

        <div style="
            font-size:13px;
            color:#5f6368;
            margin-top:3px;
        ">
            Landslide-Aware GPS Routing
        </div>

        <hr>

        <div style="
            font-size:15px;
            font-weight:bold;
        ">
            🚗 Mangan
            →
            🏁 Chungthang
        </div>

        <br>

        <div style="
            background:#fde8e7;
            border-left:5px solid #d93025;
            padding:9px;
            border-radius:6px;
        ">

            <b style="color:#d93025;">
                ✕ PRIMARY ROUTE BLOCKED
            </b>

            <br>

            Naga–Tosa /
            Theng Tunnel

        </div>

        <br>

        <div style="
            background:#e8f5e9;
            border-left:5px solid #188038;
            padding:9px;
            border-radius:6px;
        ">

            <b style="color:#188038;">
                ✓ ALTERNATIVE ROUTE
            </b>

            <br>

            Phidang →
            Sankalang →
            Chungthang

        </div>

        <hr>

        <table style="
            width:100%;
            font-size:13px;
        ">

        <tr>
            <td><b>Susceptibility</b></td>
            <td>{STATIC_SUSCEPTIBILITY:.3f}</td>
        </tr>

        <tr>
            <td><b>Dynamic trigger</b></td>
            <td>{DYNAMIC_TRIGGER:.3f}</td>
        </tr>

        <tr>
            <td><b>Risk score</b></td>
            <td>
                <b>{RISK_SCORE:.3f}</b>
            </td>
        </tr>

        <tr>
            <td><b>Risk class</b></td>
            <td>
                <b style="color:#d93025;">
                    {RISK_CLASS}
                </b>
            </td>
        </tr>

        </table>

    </div>

    """


    m.get_root().html.add_child(
        folium.Element(
            panel_html
        )
    )


    # ========================================================
    # LEGEND
    # ========================================================

    legend_html = """

    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;

        background:white;

        padding:14px 18px;

        border-radius:10px;

        box-shadow:
            0 2px 10px
            rgba(0,0,0,.25);

        font-family:Arial;

        font-size:13px;
    ">

        <b style="font-size:15px;">
            ROUTING LEGEND
        </b>

        <br><br>

        <span style="
            display:inline-block;
            width:30px;
            border-top:
                5px dashed #d93025;
            margin-right:8px;
        "></span>

        🔴 Blocked / Avoid

        <br><br>

        <span style="
            display:inline-block;
            width:30px;
            border-top:
                5px solid #188038;
            margin-right:8px;
        "></span>

        🟢 Recommended

        <br><br>

        🔵 GPS vehicle

        <br>

        ⚠️ Recent landslide

        <br>

        🟠 Historical event

    </div>

    """

    m.get_root().html.add_child(
        folium.Element(
            legend_html
        )
    )


    # ========================================================
    # LAYER CONTROL
    # ========================================================

    folium.LayerControl(
        collapsed=False
    ).add_to(m)


    # ========================================================
    # FIT MAP TO BOTH ROUTES
    # ========================================================

    all_points = (
        route_a["geometry"]
        +
        route_b["geometry"]
    )

    if all_points:

        m.fit_bounds(
            [
                [
                    min(p[0] for p in all_points),
                    min(p[1] for p in all_points)
                ],
                [
                    max(p[0] for p in all_points),
                    max(p[1] for p in all_points)
                ]
            ],
            padding=(30, 30)
        )


    return m


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("        GIRI-RAKSHAK — SIKKIM GPS DEMO")
    print("=" * 70)

    print()
    print("Start       :", MANGAN)
    print("Destination :", CHUNGTHANG)

    print()
    print("Real recent landslide:")
    print(RECENT_LANDSLIDE)

    print()
    print("Validated model:")
    print("Susceptibility :", STATIC_SUSCEPTIBILITY)
    print("Dynamic trigger:", DYNAMIC_TRIGGER)
    print("Risk score      :", round(RISK_SCORE, 4))
    print("Risk class      :", RISK_CLASS)


    # ========================================================
    # ROUTE A
    # ========================================================

    try:

        route_a = build_blocked_route()

    except Exception as e:

        print()
        print("ERROR BUILDING ROUTE A:")
        print(e)

        return


    # ========================================================
    # ROUTE B
    # ========================================================

    try:

        route_b = build_alternative_route()

    except Exception as e:

        print()
        print("ERROR BUILDING ROUTE B:")
        print(e)

        return


    # ========================================================
    # PRINT ROUTE INFORMATION
    # ========================================================

    print()
    print("=" * 70)
    print("ROUTE COMPARISON")
    print("=" * 70)

    print()
    print("🔴 ROUTE A — BLOCKED")
    print(
        f"Distance : {route_a['distance_km']:.2f} km"
    )
    print(
        f"Time     : "
        f"{format_minutes(route_a['duration_min'])}"
    )

    print()
    print("🟢 ROUTE B — RECOMMENDED")
    print(
        f"Distance : {route_b['distance_km']:.2f} km"
    )
    print(
        f"Time     : "
        f"{format_minutes(route_b['duration_min'])}"
    )


    # ========================================================
    # CREATE MAP
    # ========================================================

    m = create_map(
        route_a,
        route_b
    )


    # ========================================================
    # SAVE
    # ========================================================

    m.save(
        OUTPUT_HTML
    )


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)
    print("✅ SIKKIM GPS MAP CREATED")
    print("=" * 70)

    print()
    print(
        "HTML:"
    )

    print(
        os.path.abspath(
            OUTPUT_HTML
        )
    )

    print()
    print("Map contains:")

    print("🔴 Red dashed blocked Route A")
    print("🟢 Green recommended Route B")
    print("⚠️ Recent Sep 2026 landslide")
    print("🟠 Two real historical landslides")
    print("🚗 Animated GPS vehicle")
    print("🏁 Chungthang destination")

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()