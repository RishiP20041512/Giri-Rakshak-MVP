from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import Point


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

INPUT_GPKG = (
    ROOT
    / "pilot_route_data"
    / "pilot_roads_dynamic_cost.gpkg"
)

INPUT_LAYER = "pilot_roads_dynamic_cost"


# ============================================================
# DEMO LOCATIONS
# ============================================================
#
# IMPORTANT:
# These are coordinates inside the pilot road network.
#
# Change these later from the Streamlit user's
# selected START and DESTINATION.
#
# ============================================================

START_LAT = 25.5788
START_LON = 91.8933

DEST_LAT = 25.6200
DEST_LON = 91.8827


# ============================================================
# GRAPH SETTINGS
# ============================================================

SNAP_DISTANCE_M = 1000


# ============================================================
# LOAD ROAD NETWORK
# ============================================================

def load_roads():

    print("\n[1/6] Loading road-cost network...")

    if not INPUT_GPKG.exists():
        raise FileNotFoundError(
            f"Missing input:\n{INPUT_GPKG}"
        )

    roads = gpd.read_file(
        INPUT_GPKG,
        layer=INPUT_LAYER,
    )

    print(
        f"Road segments loaded: {len(roads):,}"
    )

    return roads


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(roads):

    print("\n[2/6] Building NetworkX graph...")

    # Work in projected CRS so node coordinates
    # and distances are numerically stable.
    roads_projected = roads.to_crs(
        "EPSG:6933"
    )

    G = nx.MultiDiGraph()

    for idx, row in roads_projected.iterrows():

        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        # ----------------------------------------------------
        # Convert line to coordinate sequence
        # ----------------------------------------------------

        if geometry.geom_type == "MultiLineString":

            lines = list(
                geometry.geoms
            )

            coords = []

            for line in lines:
                coords.extend(
                    list(line.coords)
                )

        else:

            coords = list(
                geometry.coords
            )

        if len(coords) < 2:
            continue

        start_xy = coords[0]
        end_xy = coords[-1]

        # Rounded coordinates prevent tiny floating-point
        # differences from creating unnecessary nodes.
        start_node = (
            round(start_xy[0], 2),
            round(start_xy[1], 2),
        )

        end_node = (
            round(end_xy[0], 2),
            round(end_xy[1], 2),
        )

        # ----------------------------------------------------
        # Attributes
        # ----------------------------------------------------

        length_m = float(
            row.geometry.length
        )

        base_time = float(
            row.get(
                "base_travel_time_min",
                0,
            )
        )

        routing_cost = float(
            row.get(
                "routing_cost_min",
                base_time,
            )
        )

        lsi = row.get(
            "LSI_mean",
            np.nan,
        )

        risk_class = row.get(
            "LSI_risk_class",
            "NO_DATA",
        )

        routing_status = row.get(
            "routing_status",
            "AVAILABLE",
        )

        historical_count = row.get(
            "historical_landslide_count",
            0,
        )

        active_blocked = bool(
            row.get(
                "active_blocked",
                False,
            )
        )

        # ----------------------------------------------------
        # Add edge
        # ----------------------------------------------------

        G.add_edge(
            start_node,
            end_node,

            road_id=str(
                row.get(
                    "road_id",
                    idx,
                )
            ),

            length_m=length_m,

            base_travel_time_min=base_time,

            routing_cost_min=routing_cost,

            LSI_mean=lsi,

            LSI_risk_class=risk_class,

            routing_status=routing_status,

            historical_landslide_count=historical_count,

            active_blocked=active_blocked,

            geometry=row.geometry,
        )

    print(
        f"Graph nodes: {G.number_of_nodes():,}"
    )

    print(
        f"Graph edges: {G.number_of_edges():,}"
    )

    return G, roads_projected


# ============================================================
# SNAP COORDINATE TO ROAD
# ============================================================

def nearest_graph_node(
    G,
    lat,
    lon,
):

    # Transform WGS84 → EPSG:6933
    point = gpd.GeoSeries(
        [
            Point(
                lon,
                lat,
            )
        ],
        crs="EPSG:4326",
    ).to_crs(
        "EPSG:6933"
    ).iloc[0]

    x = point.x
    y = point.y

    nearest = min(
        G.nodes,
        key=lambda node:
        (node[0] - x) ** 2
        +
        (node[1] - y) ** 2
    )

    distance = (
        (
            nearest[0] - x
        ) ** 2
        +
        (
            nearest[1] - y
        ) ** 2
    ) ** 0.5

    return nearest, distance


# ============================================================
# FIND ROUTE
# ============================================================

def find_route(
    G,
    start_node,
    destination_node,
):

    print(
        "\n[4/6] Finding lowest-cost route..."
    )

    try:

        route = nx.shortest_path(
            G,
            source=start_node,
            target=destination_node,
            weight="routing_cost_min",
        )

    except nx.NetworkXNoPath:

        print(
            "No route available."
        )

        return None

    return route


# ============================================================
# SUMMARIZE ROUTE
# ============================================================

def summarize_route(
    G,
    route,
):

    total_distance_km = 0.0
    total_base_time = 0.0
    total_cost = 0.0

    high_risk_edges = 0
    historical_edges = 0
    blocked_edges = 0

    road_ids = []

    for u, v in zip(
        route[:-1],
        route[1:],
    ):

        edge_data = G.get_edge_data(
            u,
            v,
        )

        if edge_data is None:
            continue

        # MultiDiGraph can contain multiple edges.
        edge = min(
            edge_data.values(),
            key=lambda d:
            d.get(
                "routing_cost_min",
                np.inf,
            ),
        )

        total_distance_km += (
            edge["length_m"]
            / 1000
        )

        total_base_time += (
            edge["base_travel_time_min"]
        )

        total_cost += (
            edge["routing_cost_min"]
        )

        road_ids.append(
            edge["road_id"]
        )

        if edge["LSI_risk_class"] in [
            "HIGH",
            "VERY HIGH",
        ]:
            high_risk_edges += 1

        if edge[
            "historical_landslide_count"
        ] > 0:
            historical_edges += 1

        if edge[
            "active_blocked"
        ]:
            blocked_edges += 1

    return {
        "distance_km":
            total_distance_km,

        "base_time_min":
            total_base_time,

        "routing_cost_min":
            total_cost,

        "high_risk_segments":
            high_risk_edges,

        "historical_segments":
            historical_edges,

        "blocked_segments":
            blocked_edges,

        "road_ids":
            road_ids,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 6 — NETWORKX SAFE ROUTING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    roads = load_roads()

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    G, roads_projected = build_graph(
        roads
    )

    # --------------------------------------------------------
    # SNAP START
    # --------------------------------------------------------

    print(
        "\n[3/6] Snapping start and destination..."
    )

    start_node, start_distance = (
        nearest_graph_node(
            G,
            START_LAT,
            START_LON,
        )
    )

    destination_node, destination_distance = (
        nearest_graph_node(
            G,
            DEST_LAT,
            DEST_LON,
        )
    )

    print(
        f"\nSTART"
    )

    print(
        f"  Latitude  : {START_LAT}"
    )

    print(
        f"  Longitude : {START_LON}"
    )

    print(
        f"  Snap distance: "
        f"{start_distance:.2f} m"
    )

    print(
        f"\nDESTINATION"
    )

    print(
        f"  Latitude  : {DEST_LAT}"
    )

    print(
        f"  Longitude : {DEST_LON}"
    )

    print(
        f"  Snap distance: "
        f"{destination_distance:.2f} m"
    )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    route = find_route(
        G,
        start_node,
        destination_node,
    )

    if route is None:
        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print(
        "\n[5/6] Calculating route summary..."
    )

    summary = summarize_route(
        G,
        route,
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("🚗 RECOMMENDED ROUTE")
    print("=" * 70)

    print(
        f"\nRoute nodes          : "
        f"{len(route):,}"
    )

    print(
        f"Distance             : "
        f"{summary['distance_km']:.2f} km"
    )

    print(
        f"Base travel time     : "
        f"{summary['base_time_min']:.2f} min"
    )

    print(
        f"Risk-adjusted cost   : "
        f"{summary['routing_cost_min']:.2f}"
        f" cost-min"
    )

    print(
        f"High-risk segments   : "
        f"{summary['high_risk_segments']}"
    )

    print(
        f"Historical segments  : "
        f"{summary['historical_segments']}"
    )

    print(
        f"Blocked segments     : "
        f"{summary['blocked_segments']}"
    )

    print(
        "\n[6/6] Routing complete."
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "STEP 6 COMPLETE!"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is the real NetworkX route "
        "using your real OSM road network."
    )

    print(
        "No current road closure is being invented."
    )


if __name__ == "__main__":
    main()