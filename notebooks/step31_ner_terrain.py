import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path
import numpy as np
from pyproj import CRS

dem_path = Path("raw_data/dem_ner/dem_ner.tif")
out_dir = Path("raw_data/dem_ner")
out_dir.mkdir(parents=True, exist_ok=True)

projected_dem = out_dir / "dem_ner_projected.tif"
slope_path = out_dir / "slope_ner.tif"

# NER spans multiple UTM zones, so use a metric CRS centered
# on the study region for consistent regional terrain calculation.
target_crs = "EPSG:32645"   # WGS84 / UTM zone 45N

print("Reading NER DEM...")

with rasterio.open(dem_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs,
        target_crs,
        src.width,
        src.height,
        *src.bounds
    )

    profile = src.profile.copy()
    profile.update(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        dtype="float32",
        compress="deflate"
    )

    print("Reprojecting DEM...")
    
    with rasterio.open(projected_dem, "w", **profile) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

print("Calculating slope...")

with rasterio.open(projected_dem) as src:
    dem = src.read(1).astype("float32")
    pixel_size_x = abs(src.transform.a)
    pixel_size_y = abs(src.transform.e)

    nodata = src.nodata

    if nodata is not None:
        valid = dem != nodata
    else:
        valid = np.isfinite(dem)

    # Gradient in metres/metre
    dz_dy, dz_dx = np.gradient(
        dem,
        pixel_size_y,
        pixel_size_x
    )

    slope = np.degrees(
        np.arctan(
            np.sqrt(dz_dx ** 2 + dz_dy ** 2)
        )
    ).astype("float32")

    slope[~valid] = -9999

    slope_profile = src.profile.copy()
    slope_profile.update(
        dtype="float32",
        nodata=-9999,
        compress="deflate"
    )

    with rasterio.open(slope_path, "w", **slope_profile) as dst:
        dst.write(slope, 1)

print("\nSUCCESS")
print("Projected DEM:", projected_dem)
print("Slope:", slope_path)

valid_slope = slope[slope != -9999]

print("Slope minimum:", float(valid_slope.min()))
print("Slope maximum:", float(valid_slope.max()))
print("Slope mean:", float(valid_slope.mean()))