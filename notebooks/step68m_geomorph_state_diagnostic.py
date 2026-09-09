from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT = Path(
    r"C:\Users\Adrija\OneDrive\Desktop\landslide prj\Giri-rakshak-main"
)

MISSING_FILE = (
    PROJECT
    / "processed"
    / "step68l_geomorphology_missing_points.csv"
)

BOUNDARY_FILE = (
    PROJECT
    / "raw_data"
    / "boundaries"
    / "geoBoundaries-IND-ADM1.geojson"
)

OUTPUT = (
    PROJECT
    / "processed"
    / "step68m_geomorph_state_diagnostic.csv"
)


print("=" * 75)
print("STEP 68M — GEOMORPHOLOGY STATE DIAGNOSTIC")
print("=" * 75)


# ============================================================
# LOAD MISSING POINTS
# ============================================================

d = pd.read_csv(
    MISSING_FILE
)

print(
    f"\nCurrent raster-missing points: {len(d)}"
)


# ============================================================
# CONVERT POINTS TO GEODATAFRAME
# ============================================================

points = gpd.GeoDataFrame(
    d.copy(),
    geometry=gpd.points_from_xy(
        d["lon"],
        d["lat"]
    ),
    crs="EPSG:4326"
)


# ============================================================
# LOAD INDIA BOUNDARY
# ============================================================

boundary = gpd.read_file(
    BOUNDARY_FILE
)

print(
    f"Boundary administrative units: "
    f"{len(boundary)}"
)


# ============================================================
# SELECT NER STATES USING ACTUAL FILE NAMES
# ============================================================

NER_STATES = [
    "Arunāchal Pradesh",
    "Assam",
    "Manipur",
    "Meghālaya",
    "Mizoram",
    "Nāgāland",
    "Sikkim",
    "Tripura",
]

ner = boundary[
    boundary["shapeName"].isin(
        NER_STATES
    )
].copy()

print(
    f"NER states found: {len(ner)}"
)


# ============================================================
# SPATIAL JOIN
# ============================================================

joined = gpd.sjoin(
    points,
    ner[
        [
            "shapeName",
            "geometry"
        ]
    ],
    how="left",
    predicate="within"
)

joined = joined.rename(
    columns={
        "shapeName": "state"
    }
)


# ============================================================
# STATE BREAKDOWN
# ============================================================

print("\n" + "=" * 75)
print("ALL 258 RASTER-MISSING POINTS BY STATE")
print("=" * 75)

print(
    joined["state"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# RECOVERABLE ONLY
# ============================================================

recoverable = joined[
    joined[
        "original_geomorphology"
    ].notna()
].copy()

unavailable = joined[
    joined[
        "original_geomorphology"
    ].isna()
].copy()


print("\n" + "=" * 75)
print("RECOVERABLE POINTS — ORIGINAL BHUVAN VALUE AVAILABLE")
print("=" * 75)

print(
    f"Recoverable: {len(recoverable)}"
)

print(
    recoverable["state"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# SOURCE-MISSING ONLY
# ============================================================

print("\n" + "=" * 75)
print("SOURCE-MISSING POINTS")
print("=" * 75)

print(
    f"Source missing: {len(unavailable)}"
)

print(
    unavailable["state"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# LABEL BREAKDOWN
# ============================================================

print("\n" + "=" * 75)
print("RECOVERABLE POINTS BY LABEL")
print("=" * 75)

print(
    recoverable["label"]
    .value_counts()
    .to_string()
)


# ============================================================
# SOURCE CLASS BREAKDOWN
# ============================================================

print("\n" + "=" * 75)
print("RECOVERABLE ORIGINAL GEOMORPHOLOGY CLASSES")
print("=" * 75)

print(
    recoverable[
        "original_geomorphology"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

output_columns = [
    "lon",
    "lat",
    "label",
    "original_geomorphology",
    "state",
]

joined[
    output_columns
].to_csv(
    OUTPUT,
    index=False
)


print(
    f"\nSaved:\n{OUTPUT}"
)

print("\n" + "=" * 75)
print("STEP 68M COMPLETE")
print("=" * 75)