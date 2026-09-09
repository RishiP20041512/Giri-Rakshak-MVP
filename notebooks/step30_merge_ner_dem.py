import rasterio
from rasterio.merge import merge
from pathlib import Path

dem_dir = Path("raw_data/dem_ner")

tiles = [
    dem_dir / "dem_ner_tile1.tif",
    dem_dir / "dem_ner_tile2.tif",
    dem_dir / "dem_ner_tile3.tif",
    dem_dir / "dem_ner_tile4.tif",
]

output = dem_dir / "dem_ner.tif"

print("Checking DEM tiles...")

for tile in tiles:
    if not tile.exists():
        raise FileNotFoundError(f"Missing: {tile}")

    with rasterio.open(tile) as src:
        print(
            f"{tile.name}: "
            f"{src.width} x {src.height}, "
            f"CRS={src.crs}, "
            f"bounds={src.bounds}"
        )

print("\nMerging tiles...")

src_files = [rasterio.open(tile) for tile in tiles]

mosaic, transform = merge(src_files)

profile = src_files[0].profile.copy()

profile.update(
    driver="GTiff",
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=transform,
    compress="deflate",
    tiled=True
)

with rasterio.open(output, "w", **profile) as dst:
    dst.write(mosaic)

for src in src_files:
    src.close()

print("\nSUCCESS!")
print(f"Saved: {output}")
print(f"Size: {mosaic.shape[2]} x {mosaic.shape[1]}")