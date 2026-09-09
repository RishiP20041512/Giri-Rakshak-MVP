import geopandas as gpd
import requests
import json
import time
from shapely.geometry import LineString
from shapely.ops import unary_union


# ============================================================
# LOAD CONFIG
# ============================================================

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

bbox = config["bbox"]

south = bbox["south"]
north = bbox["north"]
west = bbox["west"]
east = bbox["east"]

print("Bounding box:")
print("South:", south)
print("North:", north)
print("West :", west)
print("East :", east)


# ============================================================
# LOAD TRAINING FEATURES
# ============================================================

input_file = "processed/training_features_weather.geojson"

points = gpd.read_file(input_file)

print("\nTraining points:", len(points))


# ============================================================
# OVERPASS SETTINGS
# ============================================================

endpoints = [
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass-api.de/api/interpreter"
]

headers = {
    "User-Agent":
        "GiriRakshak/1.0 educational-landslide-project"
}


# ============================================================
# FUNCTION: QUERY OVERPASS
# ============================================================

def query_overpass(query):

    last_error = None

    for endpoint in endpoints:

        print("\nTrying:", endpoint)

        try:

            response = requests.post(
                endpoint,
                data={"data": query},
                headers=headers,
                timeout=180
            )

            print(
                "HTTP status:",
                response.status_code
            )

            if response.status_code == 200:

                return response.json()

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        except Exception as e:

            last_error = str(e)

        print("This endpoint failed.")
        time.sleep(5)

    raise RuntimeError(
        "All Overpass endpoints failed.\n"
        + str(last_error)
    )


# ============================================================
# ROAD QUERY
# ============================================================

road_query = f"""
[out:json][timeout:90];

way["highway"](
    {south},
    {west},
    {north},
    {east}
);

out geom;
"""

print("\n========================================")
print("DOWNLOADING ROADS")
print("========================================")

road_osm = query_overpass(road_query)

print(
    "Road elements:",
    len(road_osm.get("elements", []))
)


# ============================================================
# RIVER QUERY
# ============================================================

river_query = f"""
[out:json][timeout:90];

way["waterway"~"river|stream"](
    {south},
    {west},
    {north},
    {east}
);

out geom;
"""

print("\n========================================")
print("DOWNLOADING RIVERS / STREAMS")
print("========================================")

river_osm = query_overpass(river_query)

print(
    "River elements:",
    len(river_osm.get("elements", []))
)


# ============================================================
# CONVERT OSM WAYS TO LINES
# ============================================================

def extract_lines(osm_data):

    lines = []

    for element in osm_data.get("elements", []):

        if element.get("type") != "way":
            continue

        geometry = element.get("geometry")

        if not geometry:
            continue

        coords = [
            (p["lon"], p["lat"])
            for p in geometry
        ]

        if len(coords) >= 2:

            lines.append(
                LineString(coords)
            )

    return lines


road_lines = extract_lines(road_osm)

river_lines = extract_lines(river_osm)


print("\nRoad geometries:", len(road_lines))
print("River geometries:", len(river_lines))


# ============================================================
# CHECK DATA
# ============================================================

if len(road_lines) == 0:

    raise RuntimeError(
        "No road geometries found."
    )

if len(river_lines) == 0:

    raise RuntimeError(
        "No river/stream geometries found."
    )


# ============================================================
# PROJECT TO METRES
# ============================================================

projected_crs = points.estimate_utm_crs()

print(
    "\nProjected CRS:",
    projected_crs
)

points_projected = points.to_crs(
    projected_crs
)


roads_gdf = gpd.GeoDataFrame(
    geometry=road_lines,
    crs="EPSG:4326"
).to_crs(projected_crs)


rivers_gdf = gpd.GeoDataFrame(
    geometry=river_lines,
    crs="EPSG:4326"
).to_crs(projected_crs)


# ============================================================
# UNION
# ============================================================

print("\nCombining road geometries...")

roads_union = unary_union(
    roads_gdf.geometry
)

print("Combining river geometries...")

rivers_union = unary_union(
    rivers_gdf.geometry
)


# ============================================================
# DISTANCE TO ROAD
# ============================================================

print("\nCalculating distance to roads...")

points_projected["dist_to_road"] = (
    points_projected.geometry.apply(
        lambda p: p.distance(roads_union)
    )
)


# ============================================================
# DISTANCE TO RIVER
# ============================================================

print(
    "Calculating distance to rivers..."
)

points_projected["dist_to_river"] = (
    points_projected.geometry.apply(
        lambda p: p.distance(rivers_union)
    )
)


# ============================================================
# BACK TO WGS84
# ============================================================

points_final = points_projected.to_crs(
    "EPSG:4326"
)


# ============================================================
# STATISTICS
# ============================================================

print("\n========================================")
print("DISTANCE STATISTICS")
print("========================================")

print(
    points_final[
        [
            "dist_to_road",
            "dist_to_river"
        ]
    ].describe()
)


print("\nSample values:")

print(
    points_final[
        [
            "label",
            "dist_to_road",
            "dist_to_river"
        ]
    ]
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    points_final[
        [
            "dist_to_road",
            "dist_to_river"
        ]
    ].isnull().sum()
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "processed/training_features_complete.geojson"
)

points_final.to_file(
    output_file,
    driver="GeoJSON"
)


print("\n========================================")
print("STEP 15 COMPLETE")
print("========================================")

print(
    "Saved:",
    output_file
)