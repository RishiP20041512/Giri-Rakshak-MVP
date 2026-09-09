import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# --------------------------------------------------
# FILES
# --------------------------------------------------

LANDSLIDE_FILE =  "raw_data/landslides/expanded_landslides.csv"
BACKGROUND_FILE = "raw_data/landslides/background.csv"

OUTPUT_FILE = "processed/training_points.geojson"

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

landslides = pd.read_csv(LANDSLIDE_FILE)
background = pd.read_csv(BACKGROUND_FILE)

print("Landslide points:", len(landslides))
print("Background points:", len(background))

# --------------------------------------------------
# ADD LABELS
# --------------------------------------------------

landslides["label"] = 1
background["label"] = 0

# --------------------------------------------------
# KEEP COMMON COLUMNS
# --------------------------------------------------
if "date" not in landslides.columns:
    landslides["date"] = ""
landslides = landslides[["lat", "lon", "date", "source", "label"]]

background = background[["lat", "lon", "label"]]

# Background points don't have event dates/sources
background["date"] = ""
background["source"] = "random_background"

# Make column order identical
background = background[
    ["lat", "lon", "date", "source", "label"]
]

# --------------------------------------------------
# COMBINE
# --------------------------------------------------

df = pd.concat(
    [landslides, background],
    ignore_index=True
)

print("\nTotal training points:", len(df))

print("\nLabel counts:")
print(df["label"].value_counts())

# --------------------------------------------------
# CREATE GEOMETRY
# --------------------------------------------------

geometry = [
    Point(lon, lat)
    for lon, lat in zip(
        df["lon"],
        df["lat"]
    )
]

gdf = gpd.GeoDataFrame(
    df,
    geometry=geometry,
    crs="EPSG:4326"
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

gdf.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\nFirst 5 records:")
print(gdf.head())