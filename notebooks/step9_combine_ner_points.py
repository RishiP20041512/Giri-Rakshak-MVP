import pandas as pd
import geopandas as gpd

LANDSLIDE_FILE = "raw_data/landslides/landslides_ner_clean.csv"
BACKGROUND_FILE = "raw_data/landslides/background_ner_clean.csv"

OUTPUT_FILE = "processed/ner_training_points.geojson"

# Load
landslides = pd.read_csv(LANDSLIDE_FILE)
background = pd.read_csv(BACKGROUND_FILE)

# Assign labels
landslides["label"] = 1
background["label"] = 0

# Keep common columns
columns = ["lat", "lon", "date", "source", "region", "label"]

landslides = landslides[columns]
background = background[columns]

# Combine
df = pd.concat(
    [landslides, background],
    ignore_index=True
)

# Remove duplicate coordinates
df = df.drop_duplicates(
    subset=["lat", "lon"]
).reset_index(drop=True)

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(
        df["lon"],
        df["lat"]
    ),
    crs="EPSG:4326"
)

# Save
gdf.to_file(
    OUTPUT_FILE,
    driver="GeoJSON"
)

print("=" * 50)
print("STEP 9 - NER TRAINING POINTS")
print("=" * 50)

print("Landslides:", (gdf["label"] == 1).sum())
print("Background:", (gdf["label"] == 0).sum())
print("Total:", len(gdf))

print("\nSaved:")
print(OUTPUT_FILE)