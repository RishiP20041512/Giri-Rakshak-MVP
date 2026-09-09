import geopandas as gpd
import rasterio

# Load the combined training points
points = gpd.read_file(
    "processed/training_points.geojson"
)

print("Training points:", len(points))

# Raster files
rasters = {
    "elevation": "raw_data/dem/dem.tif",
    "slope": "raw_data/dem/slope.tif",
    "aspect": "raw_data/dem/aspect.tif",
    "ndvi": "processed/ndvi.tif"
}

# Sample each raster
for column, raster_path in rasters.items():

    print(f"\nSampling {column}...")
    print("File:", raster_path)

    with rasterio.open(raster_path) as src:

        # Convert points to the raster CRS
        points_projected = points.to_crs(src.crs)

        coordinates = [
            (point.x, point.y)
            for point in points_projected.geometry
        ]

        values = []

        for value in src.sample(coordinates):
            values.append(value[0])

        points[column] = values

        print("CRS:", src.crs)
        print("Raster size:", src.width, "x", src.height)

# Display results
print("\n================================")
print("RASTER SAMPLING COMPLETE")
print("================================")

print(
    points[
        [
            "label",
            "elevation",
            "slope",
            "aspect",
            "ndvi"
        ]
    ]
)

# Statistics
print("\nStatistics:")
print(
    points[
        [
            "elevation",
            "slope",
            "aspect",
            "ndvi"
        ]
    ].describe()
)

# Check missing values
print("\nMissing values:")
print(
    points[
        [
            "elevation",
            "slope",
            "aspect",
            "ndvi"
        ]
    ].isnull().sum()
)

# Save
points.to_file(
    "processed/training_features.geojson",
    driver="GeoJSON"
)

print("\nSaved:")
print("processed/training_features.geojson")