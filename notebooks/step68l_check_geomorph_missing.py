from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import pandas as pd


PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

POINTS = (
    PROJECT
    / "processed"
    / "ner_training_points.geojson"
)

ORIGINAL_GEOM = (
    PROJECT
    / "processed"
    / "ner_geomorphology.csv"
)

RASTER = (
    PROJECT
    / "processed"
    / "predictors"
    / "geomorph_origin_250m.tif"
)


print("=" * 75)
print("STEP 68L — CHECK GEOMORPHOLOGY MISSING POINTS")
print("=" * 75)


# ------------------------------------------------------------
# LOAD ORIGINAL GEOMORPHOLOGY EXTRACTION
# ------------------------------------------------------------

geom_df = pd.read_csv(
    ORIGINAL_GEOM
)

print(
    f"\nOriginal geomorphology records: "
    f"{len(geom_df)}"
)

print(
    f"Original valid geomorphology: "
    f"{geom_df['geomorphology'].notna().sum()}"
)

print(
    f"Original missing geomorphology: "
    f"{geom_df['geomorphology'].isna().sum()}"
)


# ------------------------------------------------------------
# LOAD TRAINING POINTS
# ------------------------------------------------------------

points = gpd.read_file(
    POINTS
)

print(
    f"\nTraining points: {len(points)}"
)


# ------------------------------------------------------------
# SAMPLE CURRENT RASTER
# ------------------------------------------------------------

with rasterio.open(RASTER) as src:

    pts = points.to_crs(
        src.crs
    )

    coords = [
        (geom.x, geom.y)
        for geom in pts.geometry
    ]

    raster_values = np.array(
        list(
            src.sample(coords)
        )
    )[:, 0]

    nodata = src.nodata

    if nodata is not None:

        raster_valid = (
            np.isfinite(raster_values)
            &
            (raster_values != nodata)
        )

    else:

        raster_valid = np.isfinite(
            raster_values
        )


# ------------------------------------------------------------
# ALIGN ORIGINAL DATA
# ------------------------------------------------------------

# Match by longitude/latitude because both datasets
# originate from the same training points.

merged = points.copy()

merged["lon"] = merged.geometry.x
merged["lat"] = merged.geometry.y

original = geom_df.copy()

# Standardize coordinate names
if "longitude" in original.columns:
    original_lon = "longitude"
elif "lon" in original.columns:
    original_lon = "lon"
else:
    raise ValueError(
        "Longitude column not found."
    )

if "latitude" in original.columns:
    original_lat = "latitude"
elif "lat" in original.columns:
    original_lat = "lat"
else:
    raise ValueError(
        "Latitude column not found."
    )


merged["original_geomorphology"] = pd.Series(
    [None] * len(merged),
    dtype="object"
)

for i, row in merged.iterrows():

    lon = row["lon"]
    lat = row["lat"]

    distances = (
        (original[original_lon] - lon) ** 2
        +
        (original[original_lat] - lat) ** 2
    )

    idx = distances.idxmin()

    nearest = original.loc[
        idx
    ]

    if (
        abs(
            nearest[original_lon]
            - lon
        ) < 1e-5
        and
        abs(
            nearest[original_lat]
            - lat
        ) < 1e-5
    ):

        merged.loc[
            i,
            "original_geomorphology"
        ] = nearest[
            "geomorphology"
        ]


merged["raster_valid"] = (
    raster_valid
)

merged["raster_value"] = (
    raster_values
)


# ------------------------------------------------------------
# MISSING CASES
# ------------------------------------------------------------

missing_raster = merged[
    ~merged["raster_valid"]
].copy()


print("\n" + "=" * 75)
print("CURRENT RASTER MISSING POINTS")
print("=" * 75)

print(
    f"Missing raster values: "
    f"{len(missing_raster)}"
)


# ------------------------------------------------------------
# STATE BREAKDOWN
# ------------------------------------------------------------

if "state" in missing_raster.columns:

    print(
        "\nMissing points by state:"
    )

    print(
        missing_raster[
            "state"
        ]
        .value_counts()
        .to_string()
    )


# ------------------------------------------------------------
# ORIGINAL SOURCE AVAILABILITY
# ------------------------------------------------------------

print(
    "\nOriginal Bhuvan extraction "
    "availability among current raster-missing points:"
)

source_available = (
    missing_raster[
        "original_geomorphology"
    ].notna()
)

print(
    f"Original source available: "
    f"{source_available.sum()}"
)

print(
    f"Original source missing: "
    f"{(~source_available).sum()}"
)


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

output = (
    PROJECT
    / "processed"
    / "step68l_geomorphology_missing_points.csv"
)

missing_raster[
    [
        "lon",
        "lat",
        "label",
        "original_geomorphology",
        "raster_valid",
        "raster_value",
    ]
    + (
        ["state"]
        if "state" in missing_raster.columns
        else []
    )
].to_csv(
    output,
    index=False
)


print(
    f"\nSaved:\n{output}"
)

print("\n" + "=" * 75)
print("STEP 68L COMPLETE")
print("=" * 75)