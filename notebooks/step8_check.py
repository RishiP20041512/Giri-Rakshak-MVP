import xarray as xr
import glob
files = glob.glob("raw_data/rainfall/*.nc4")
print(f"Found {len(files)} rainfall files")
if files:
    ds = xr.open_dataset(files[0])
    print(ds)
    # Sample rainfall at Anini (28.638, 95.856) as a sanity check
    val = ds["precipitationCal"].sel(lat=28.638, lon=95.856, method="nearest").values
    print(f"Sample rainfall value near Anini: {val}")
else:
    print("No files found - check earthaccess login or try Option B (data.gov.in CSV)")
