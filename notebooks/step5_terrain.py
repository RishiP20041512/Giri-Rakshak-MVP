import rasterio
import numpy as np
import math

dem_file = "raw_data/dem/dem.tif"

with rasterio.open(dem_file) as src:
    dem = src.read(1).astype("float32")
    profile = src.profile
    transform = src.transform

    # DEM is in EPSG:4326, so pixel sizes are in degrees.
    # Convert approximately to metres at the centre latitude.
    mean_lat = (src.bounds.top + src.bounds.bottom) / 2

    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = (
        111320.0 * math.cos(math.radians(mean_lat))
    )

    xres_m = abs(transform.a) * meters_per_degree_lon
    yres_m = abs(transform.e) * meters_per_degree_lat

print("Pixel size:")
print("X:", xres_m, "metres")
print("Y:", yres_m, "metres")

# Calculate elevation gradients in metres/metre
dy, dx = np.gradient(
    dem,
    yres_m,
    xres_m
)

# Slope in degrees
slope = np.degrees(
    np.arctan(
        np.sqrt(dx**2 + dy**2)
    )
)

# Aspect
aspect = np.degrees(
    np.arctan2(-dx, dy)
)

aspect = 90.0 - aspect
aspect = np.where(aspect < 0, aspect + 360, aspect)
aspect = np.where(aspect >= 360, aspect - 360, aspect)

# Output settings
profile.update(
    dtype="float32",
    count=1,
    compress="lzw"
)

# Save slope
with rasterio.open(
    "raw_data/dem/slope.tif",
    "w",
    **profile
) as dst:
    dst.write(slope.astype("float32"), 1)

# Save aspect
with rasterio.open(
    "raw_data/dem/aspect.tif",
    "w",
    **profile
) as dst:
    dst.write(aspect.astype("float32"), 1)

print("\nCreated:")
print("raw_data/dem/slope.tif")
print("raw_data/dem/aspect.tif")

print("\nSlope statistics:")
print("Min:", np.nanmin(slope))
print("Mean:", np.nanmean(slope))
print("Max:", np.nanmax(slope))

print("\nAspect statistics:")
print("Min:", np.nanmin(aspect))
print("Mean:", np.nanmean(aspect))
print("Max:", np.nanmax(aspect))