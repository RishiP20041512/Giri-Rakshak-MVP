import geopandas as gpd


# ============================================================
# LOAD COMPLETE FEATURES
# ============================================================

input_file = (
    "processed/training_features_complete.geojson"
)

gdf = gpd.read_file(input_file)

print("Training points:", len(gdf))


# ============================================================
# SELECT MACHINE LEARNING FEATURES
# ============================================================

columns = [
    "elevation",
    "slope",
    "aspect",
    "ndvi",
    "rainfall_3day",
    "soil_moisture",
    "dist_to_road",
    "dist_to_river",
    "label"
]

df = gdf[columns].copy()


# ============================================================
# CHECK DATA
# ============================================================

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nFeature statistics:")
print(df.describe())


# ============================================================
# REMOVE MISSING ROWS
# ============================================================

before = len(df)

df = df.dropna()

after = len(df)

print(
    f"\nRemoved {before - after} rows containing "
    "missing values."
)


# ============================================================
# SAVE CSV
# ============================================================

output_file = "processed/training_data.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n========================================")
print("STEP 16 COMPLETE")
print("========================================")

print("Saved:", output_file)
print("Final shape:", df.shape)