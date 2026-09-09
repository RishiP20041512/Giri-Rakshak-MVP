"""
STEP 64C — AUDIT ENVIRONMENTAL SOURCES
======================================

Checks the currently available rainfall, soil-moisture,
and NDVI source files before preparing them for the
final 250 m EPSG:6933 raster grid.

This step does NOT modify any source data.
"""

from pathlib import Path

import h5py
import netCDF4
from pyhdf.SD import SD


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAINFALL_DIR = PROJECT_ROOT / "raw_data" / "rainfall"
SOIL_DIR = PROJECT_ROOT / "raw_data" / "soil_moisture"
NDVI_DIR = PROJECT_ROOT / "raw_data" / "ndvi"


print("=" * 70)
print("STEP 64C — AUDIT ENVIRONMENTAL SOURCES")
print("=" * 70)


# ================================================================
# RAINFALL
# ================================================================

print("\n[1] RAINFALL SOURCE FILES")
print("-" * 70)

rainfall_files = sorted(
    RAINFALL_DIR.glob("*")
)

if not rainfall_files:
    raise FileNotFoundError(
        f"No rainfall files found in:\n{RAINFALL_DIR}"
    )

for file in rainfall_files:

    if file.suffix.lower() not in [
        ".nc",
        ".nc4",
        ".netcdf",
    ]:
        continue

    print(f"\n  File: {file.name}")

    with netCDF4.Dataset(
        file,
        "r"
    ) as ds:

        print(
            f"    Dimensions: "
            f"{dict(ds.dimensions)}"
        )

        print(
            f"    Variables: "
            f"{list(ds.variables.keys())}"
        )

        if "precipitation" in ds.variables:

            var = ds.variables[
                "precipitation"
            ]

            print(
                f"    precipitation shape: "
                f"{var.shape}"
            )

            print(
                f"    precipitation units: "
                f"{getattr(var, 'units', 'N/A')}"
            )

            print(
                f"    long_name: "
                f"{getattr(var, 'long_name', 'N/A')}"
            )

        for coord in [
            "lat",
            "latitude",
            "lon",
            "longitude",
        ]:

            if coord in ds.variables:

                v = ds.variables[
                    coord
                ]

                print(
                    f"    {coord}: "
                    f"shape={v.shape}, "
                    f"units={getattr(v, 'units', 'N/A')}"
                )


# ================================================================
# SOIL MOISTURE
# ================================================================

print("\n\n[2] SOIL MOISTURE SOURCE FILES")
print("-" * 70)

soil_files = sorted(
    list(SOIL_DIR.glob("*.h5"))
    +
    list(SOIL_DIR.glob("*.hdf5"))
)

if not soil_files:

    raise FileNotFoundError(
        f"No soil-moisture HDF5 files found in:\n"
        f"{SOIL_DIR}"
    )

for file in soil_files:

    print(
        f"\n  File: {file.name}"
    )

    with h5py.File(
        file,
        "r"
    ) as h5:

        group_name = (
            "Soil_Moisture_Retrieval_Data_AM"
        )

        if group_name not in h5:

            print(
                "    WARNING: expected SMAP group "
                "not found."
            )

            continue

        group = h5[
            group_name
        ]

        print(
            "    Datasets in expected group:"
        )

        for name in group.keys():

            obj = group[name]

            if hasattr(
                obj,
                "shape"
            ):

                print(
                    f"      {name}: "
                    f"shape={obj.shape}"
                )

        if (
            "soil_moisture"
            in group
        ):

            sm = group[
                "soil_moisture"
            ]

            print(
                f"    soil_moisture shape: "
                f"{sm.shape}"
            )

            print(
                f"    soil_moisture units: "
                f"{sm.attrs.get('units', 'N/A')}"
            )

        if (
            "latitude"
            in group
        ):

            lat = group[
                "latitude"
            ]

            print(
                f"    latitude shape: "
                f"{lat.shape}"
            )

        if (
            "longitude"
            in group
        ):

            lon = group[
                "longitude"
            ]

            print(
                f"    longitude shape: "
                f"{lon.shape}"
            )


# ================================================================
# NDVI
# ================================================================

print("\n\n[3] NDVI SOURCE FILES")
print("-" * 70)

ndvi_files = sorted(
    NDVI_DIR.glob("*.hdf")
)

if not ndvi_files:

    raise FileNotFoundError(
        f"No MODIS HDF files found in:\n"
        f"{NDVI_DIR}"
    )

for file in ndvi_files:

    print(
        f"\n  File: {file.name}"
    )

    hdf = SD(
        str(file)
    )

    datasets = hdf.datasets()

    print(
        f"    Number of datasets: "
        f"{len(datasets)}"
    )

    for name, info in datasets.items():

        print(
            f"      {name}: "
            f"shape={info[2]}"
        )

    if "250m 16 days NDVI" in datasets:

        data = hdf.select(
            "250m 16 days NDVI"
        )

        print(
            "    NDVI dataset found."
        )

        print(
            f"    NDVI shape: "
            f"{data[:].shape}"
        )

        print(
            f"    Scale factor: "
            f"{data.attributes().get('scale_factor', 'N/A')}"
        )

    hdf.end()


# ================================================================
# SUMMARY
# ================================================================

print("\n\n[4] SUMMARY")
print("-" * 70)

print(
    f"  Rainfall files      : "
    f"{len(rainfall_files)}"
)

print(
    f"  Soil-moisture files : "
    f"{len(soil_files)}"
)

print(
    f"  NDVI HDF files      : "
    f"{len(ndvi_files)}"
)

print("\nNo source files were modified.")

print("=" * 70)
print("STEP 64C SOURCE AUDIT COMPLETED")
print("=" * 70)