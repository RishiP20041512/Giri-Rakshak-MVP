import os
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt


# ============================================================
# STEP 71 — CREATE FINAL SUSCEPTIBILITY MAP
# ============================================================

print("=" * 70)
print("STEP 71 — FINAL 8-FACTOR SUSCEPTIBILITY MAP")
print("=" * 70)


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(
    BASE,
    "processed",
    "step70_8factor_susceptibility_probability.tif"
)

OUTPUT_TIF = os.path.join(
    BASE,
    "processed",
    "step71_final_8factor_susceptibility.tif"
)

OUTPUT_PNG = os.path.join(
    BASE,
    "processed",
    "step71_final_8factor_susceptibility_map.png"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not os.path.exists(INPUT):
    raise FileNotFoundError(f"Input not found:\n{INPUT}")

print("\nLoading Step 70 susceptibility raster...")
print(INPUT)


# ============================================================
# READ RASTER
# ============================================================

with rasterio.open(INPUT) as src:

    data = src.read(1)

    profile = src.profile.copy()

    print("\nRaster information:")
    print(f"CRS        : {src.crs}")
    print(f"Dimensions : {src.width} x {src.height}")
    print(f"Resolution : {src.res}")
    print(f"NoData     : {src.nodata}")


# ============================================================
# VALID DATA
# ============================================================

nodata = -9999

valid = (
    np.isfinite(data) &
    (data != nodata) &
    (data >= 0) &
    (data <= 1)
)

valid_values = data[valid]

print("\nFinal susceptibility statistics:")
print(f"Valid cells : {valid_values.size:,}")
print(f"Minimum     : {valid_values.min():.6f}")
print(f"Maximum     : {valid_values.max():.6f}")
print(f"Mean        : {valid_values.mean():.6f}")
print(f"Median      : {np.median(valid_values):.6f}")


# ============================================================
# WRITE CLEAN FINAL TIFF
# ============================================================

profile.update(
    dtype="float32",
    count=1,
    nodata=nodata,
    compress="deflate"
)

with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:

    clean_data = data.astype(np.float32)

    clean_data[~valid] = nodata

    dst.write(clean_data, 1)


print("\nFinal GeoTIFF created:")
print(OUTPUT_TIF)


# ============================================================
# CREATE VISUAL MAP
# ============================================================

print("\nCreating presentation map...")

display_data = data.astype(float)
display_data[~valid] = np.nan

plt.figure(figsize=(12, 10))

im = plt.imshow(
    display_data,
    vmin=0,
    vmax=1,
    cmap="RdYlGn_r"
)

plt.colorbar(
    im,
    label="Landslide Susceptibility Probability"
)

plt.title(
    "North-East India\n"
    "8-Factor Landslide Susceptibility"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nPNG map created:")
print(OUTPUT_PNG)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("STEP 71 COMPLETE")
print("=" * 70)

print("\nOutputs:")
print(OUTPUT_TIF)
print(OUTPUT_PNG)
