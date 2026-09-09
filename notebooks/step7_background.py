import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

LANDSLIDE_FILE = "raw_data/landslides/landslides.csv"
OUTPUT_FILE = "raw_data/landslides/background.csv"

# Minimum distance from a known landslide
MIN_DISTANCE_METERS = 500

# Number of background points per landslide
BACKGROUND_RATIO = 3

# --------------------------------------------------
# LOAD LANDSLIDE POINTS
# --------------------------------------------------

landslides = pd.read_csv("raw_data/landslides/expanded_landslides.csv")
print("Loaded landslide points:", len(landslides))

# Create GeoDataFrame
geometry = [
    Point(lon, lat)
    for lon, lat in zip(
        landslides["lon"],
        landslides["lat"]
    )
]

gdf = gpd.GeoDataFrame(
    landslides,
    geometry=geometry,
    crs="EPSG:4326"
)

# --------------------------------------------------
# CREATE A PROJECTED VERSION FOR DISTANCE CALCULATION
# --------------------------------------------------

# Automatically choose an appropriate UTM CRS
projected = gdf.to_crs(gdf.estimate_utm_crs())

print("Using projected CRS:", projected.crs)

# --------------------------------------------------
# PILOT AREA BOUNDING BOX
# --------------------------------------------------

south = 28.2
north = 28.9
west = 95.6
east = 96.1

# --------------------------------------------------
# NUMBER OF BACKGROUND POINTS
# --------------------------------------------------

target_count = len(gdf) * BACKGROUND_RATIO

print("Target background points:", target_count)

# --------------------------------------------------
# GENERATE RANDOM POINTS
# --------------------------------------------------

np.random.seed(42)

background_points = []

# Create bounding box in geographic coordinates
bbox = gpd.GeoSeries(
    [Point(west, south), Point(east, north)],
    crs="EPSG:4326"
).to_crs(projected.crs)

min_x = bbox.iloc[0].x
min_y = bbox.iloc[0].y
max_x = bbox.iloc[1].x
max_y = bbox.iloc[1].y

attempts = 0
max_attempts = target_count * 100

while len(background_points) < target_count and attempts < max_attempts:

    attempts += 1

    x = np.random.uniform(min_x, max_x)
    y = np.random.uniform(min_y, max_y)

    candidate = Point(x, y)

    # Calculate distance to every known landslide
    distances = projected.geometry.distance(candidate)

    # Accept only if at least 500 m from ALL landslides
    if distances.min() >= MIN_DISTANCE_METERS:
        background_points.append(candidate)

print("Generated background points:", len(background_points))
print("Attempts:", attempts)

# --------------------------------------------------
# CONVERT BACK TO LAT/LON
# --------------------------------------------------

background_gdf = gpd.GeoDataFrame(
    geometry=background_points,
    crs=projected.crs
).to_crs("EPSG:4326")

# Create CSV
background_df = pd.DataFrame({
    "lat": background_gdf.geometry.y,
    "lon": background_gdf.geometry.x,
    "label": 0
})

# --------------------------------------------------
# SAVE
# --------------------------------------------------

background_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nBackground points saved to:")
print(OUTPUT_FILE)

print("\nFirst 5 points:")
print(background_df.head())