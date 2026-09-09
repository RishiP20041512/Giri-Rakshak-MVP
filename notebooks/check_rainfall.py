import xarray as xr
import glob
import os

files = sorted(
    glob.glob("raw_data/rainfall/*.nc4")
)

lat = 28.394017
lon = 95.923317

total = 0.0

print("Checking rainfall at:")
print("Latitude:", lat)
print("Longitude:", lon)
print()

for file in files:

    ds = xr.open_dataset(file)

    rain = ds["precipitation"]

    value = rain.sel(
        lon=lon,
        lat=lat,
        method="nearest"
    ).mean().item()

    print(
        os.path.basename(file),
        "→",
        value,
        "mm/day"
    )

    total += value

    ds.close()

print()
print("3-day rainfall:", total, "mm")