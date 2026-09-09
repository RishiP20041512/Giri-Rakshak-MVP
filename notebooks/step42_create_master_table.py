import pandas as pd
import geopandas as gpd
from pathlib import Path


# ============================================================
# STEP 42 — CREATE MASTER FACTOR TABLE
# Giri Rakshak Project
# ============================================================

print("=" * 70)
print("STEP 42 — CREATE MASTER FACTOR TABLE")
print("=" * 70)


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "processed"

TRAINING_FILE = PROCESSED_DIR / "ner_training_points.geojson"

RAINFALL_FILE = PROCESSED_DIR / "ner_rainfall.csv"
SOIL_FILE = PROCESSED_DIR / "ner_soil_moisture.csv"
GEOMORPH_FILE = PROCESSED_DIR / "ner_geomorphology.csv"
LULC_FILE = PROCESSED_DIR / "ner_lulc.csv"
NDVI_FILE = PROCESSED_DIR / "ner_ndvi.csv"
ROAD_FILE = PROCESSED_DIR / "ner_distance_to_road.csv"
LINEAMENT_FILE = PROCESSED_DIR / "ner_lineament_density.csv"

OUTPUT_FILE = PROCESSED_DIR / "ner_master_factors.csv"


# ------------------------------------------------------------
# 2. CHECK INPUT FILES
# ------------------------------------------------------------

print("\n[1] Checking input files...")

input_files = {
    "Training points": TRAINING_FILE,
    "Rainfall": RAINFALL_FILE,
    "Soil moisture": SOIL_FILE,
    "Geomorphology": GEOMORPH_FILE,
    "LULC": LULC_FILE,
    "NDVI": NDVI_FILE,
    "Distance to road": ROAD_FILE,
    "Lineament density": LINEAMENT_FILE,
}

for name, path in input_files.items():
    if not path.exists():
        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )

    print(f"  OK: {name}")


# ------------------------------------------------------------
# 3. LOAD TRAINING POINTS
# ------------------------------------------------------------

print("\n[2] Loading training points...")

gdf = gpd.read_file(TRAINING_FILE)

print(f"  Training points: {len(gdf)}")
print(f"  CRS: {gdf.crs}")

if len(gdf) != 1071:
    raise ValueError(
        f"Expected 1071 training points, found {len(gdf)}"
    )


# ------------------------------------------------------------
# 4. CREATE BASE TABLE
# ------------------------------------------------------------

print("\n[3] Creating base table...")

# Coordinates from geometry
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

base_cols = [
    "lat",
    "lon",
    "date",
    "source",
    "region",
    "label",
]

missing_base = [c for c in base_cols if c not in gdf.columns]

if missing_base:
    raise ValueError(
        f"Missing columns in training points: {missing_base}"
    )

master = gdf[base_cols].copy()

print(f"  Base rows: {len(master)}")


# ------------------------------------------------------------
# 5. HELPER FUNCTION
# ------------------------------------------------------------

def load_factor(filename, name):
    """
    Load a factor CSV and verify that coordinates are present.
    """

    path = PROCESSED_DIR / filename

    df = pd.read_csv(path)

    print(f"\n  {name}")
    print(f"    Rows: {len(df)}")
    print(f"    Columns: {df.columns.tolist()}")

    if len(df) != 1071:
        raise ValueError(
            f"{name}: expected 1071 rows, found {len(df)}"
        )

    return df


# ------------------------------------------------------------
# 6. LOAD RAINFALL
# ------------------------------------------------------------

print("\n[4] Loading rainfall...")

rainfall = load_factor(
    "ner_rainfall.csv",
    "Rainfall"
)

if "rainfall_3day" not in rainfall.columns:
    raise ValueError(
        "Column 'rainfall_3day' not found in rainfall file."
    )

rainfall = rainfall[
    ["lat", "lon", "rainfall_3day"]
].copy()


# ------------------------------------------------------------
# 7. LOAD SOIL MOISTURE
# ------------------------------------------------------------

print("\n[5] Loading soil moisture...")

soil = load_factor(
    "ner_soil_moisture.csv",
    "Soil moisture"
)

if "soil_moisture" not in soil.columns:
    raise ValueError(
        "Column 'soil_moisture' not found in soil moisture file."
    )

soil = soil[
    ["lat", "lon", "soil_moisture"]
].copy()


# ------------------------------------------------------------
# 8. LOAD GEOMORPHOLOGY
# ------------------------------------------------------------

print("\n[6] Loading geomorphology...")

geomorph = pd.read_csv(GEOMORPH_FILE)

print(f"  Rows: {len(geomorph)}")
print(f"  Columns: {geomorph.columns.tolist()}")

required_geomorph = [
    "longitude",
    "latitude",
    "geomorphology",
]

for col in required_geomorph:
    if col not in geomorph.columns:
        raise ValueError(
            f"Column '{col}' not found in geomorphology file."
        )

if len(geomorph) != 1071:
    raise ValueError(
        f"Geomorphology: expected 1071 rows, found {len(geomorph)}"
    )

geomorph = geomorph.rename(
    columns={
        "longitude": "lon",
        "latitude": "lat",
    }
)

geomorph = geomorph[
    ["lat", "lon", "geomorphology"]
].copy()


# ------------------------------------------------------------
# 9. LOAD LULC
# ------------------------------------------------------------

print("\n[7] Loading LULC...")

lulc = load_factor(
    "ner_lulc.csv",
    "LULC"
)

if "Level_I" not in lulc.columns:
    raise ValueError(
        "Column 'Level_I' not found in LULC file."
    )

lulc = lulc[
    ["lat", "lon", "Level_I"]
].copy()


# ------------------------------------------------------------
# 10. LOAD NDVI
# ------------------------------------------------------------

print("\n[8] Loading NDVI...")

ndvi = load_factor(
    "ner_ndvi.csv",
    "NDVI"
)

if "ndvi" not in ndvi.columns:
    raise ValueError(
        "Column 'ndvi' not found in NDVI file."
    )

ndvi = ndvi[
    ["lat", "lon", "ndvi"]
].copy()


# ------------------------------------------------------------
# 11. LOAD DISTANCE TO ROAD
# ------------------------------------------------------------

print("\n[9] Loading distance to road...")

road = load_factor(
    "ner_distance_to_road.csv",
    "Distance to road"
)

if "distance_to_road_m" not in road.columns:
    raise ValueError(
        "Column 'distance_to_road_m' not found in road file."
    )

road = road[
    ["lat", "lon", "distance_to_road_m"]
].copy()


# ------------------------------------------------------------
# 12. LOAD LINEAMENT DENSITY
# ------------------------------------------------------------

print("\n[10] Loading lineament density...")

lineament = load_factor(
    "ner_lineament_density.csv",
    "Lineament density"
)

if "lineament_density" not in lineament.columns:
    raise ValueError(
        "Column 'lineament_density' not found in lineament file."
    )

lineament = lineament[
    ["lat", "lon", "lineament_density"]
].copy()


# ------------------------------------------------------------
# 13. CHECK COORDINATES BEFORE MERGING
# ------------------------------------------------------------

print("\n[11] Checking coordinate consistency...")

def check_coordinates(df, name):
    """
    Check whether factor coordinates match the base training points.
    """

    base_coords = set(
        zip(
            master["lat"].round(8),
            master["lon"].round(8)
        )
    )

    factor_coords = set(
        zip(
            df["lat"].round(8),
            df["lon"].round(8)
        )
    )

    missing = base_coords - factor_coords
    extra = factor_coords - base_coords

    print(f"  {name}:")
    print(f"    Base coordinates:   {len(base_coords)}")
    print(f"    Factor coordinates: {len(factor_coords)}")
    print(f"    Missing:            {len(missing)}")
    print(f"    Extra:              {len(extra)}")

    if missing or extra:
        raise ValueError(
            f"Coordinate mismatch detected in {name}."
        )


check_coordinates(rainfall, "Rainfall")
check_coordinates(soil, "Soil moisture")
check_coordinates(geomorph, "Geomorphology")
check_coordinates(lulc, "LULC")
check_coordinates(ndvi, "NDVI")
check_coordinates(road, "Distance to road")
check_coordinates(lineament, "Lineament density")


# ------------------------------------------------------------
# 14. MERGE FACTORS
# ------------------------------------------------------------

print("\n[12] Merging factor tables...")

factor_tables = [
    (rainfall, "rainfall_3day"),
    (soil, "soil_moisture"),
    (geomorph, "geomorphology"),
    (lulc, "Level_I"),
    (ndvi, "ndvi"),
    (road, "distance_to_road_m"),
    (lineament, "lineament_density"),
]


for df, factor_name in factor_tables:

    before = len(master)

    master = master.merge(
        df,
        on=["lat", "lon"],
        how="left",
        validate="one_to_one"
    )

    after = len(master)

    print(
        f"  Added {factor_name:<25} "
        f"rows: {before} -> {after}"
    )

    if after != 1071:
        raise ValueError(
            f"Row count changed after merging {factor_name}."
        )


# ------------------------------------------------------------
# 15. REORDER COLUMNS
# ------------------------------------------------------------

print("\n[13] Reordering columns...")

final_columns = [
    "lat",
    "lon",
    "date",
    "source",
    "region",
    "label",

    # Final conditioning factors
    "rainfall_3day",
    "soil_moisture",
    "geomorphology",
    "Level_I",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

master = master[final_columns]


# ------------------------------------------------------------
# 16. REMOVE ACCIDENTAL DUPLICATES
# ------------------------------------------------------------

print("\n[14] Checking duplicate coordinates...")

duplicate_count = master.duplicated(
    subset=["lat", "lon"]
).sum()

print(f"  Duplicate coordinate rows: {duplicate_count}")

if duplicate_count > 0:
    raise ValueError(
        "Duplicate training-point coordinates detected."
    )


# ------------------------------------------------------------
# 17. MISSING-VALUE AUDIT
# ------------------------------------------------------------

print("\n[15] Missing-value audit...")

factor_columns = [
    "rainfall_3day",
    "soil_moisture",
    "geomorphology",
    "Level_I",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

missing_summary = []

for col in factor_columns:

    missing = master[col].isna().sum()
    percentage = (missing / len(master)) * 100

    missing_summary.append(
        {
            "factor": col,
            "missing": missing,
            "missing_percent": percentage,
        }
    )

    print(
        f"  {col:<25} "
        f"missing = {missing:>3} "
        f"({percentage:.2f}%)"
    )


missing_df = pd.DataFrame(missing_summary)


# ------------------------------------------------------------
# 18. CONTINUOUS FACTOR STATISTICS
# ------------------------------------------------------------

print("\n[16] Continuous factor statistics...")

continuous_columns = [
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

print()

print(
    master[continuous_columns]
    .describe()
    .T
    .to_string()
)


# ------------------------------------------------------------
# 19. CATEGORICAL FACTOR SUMMARY
# ------------------------------------------------------------

print("\n[17] Categorical factor summaries...")

print("\nGeomorphology:")
print(
    master["geomorphology"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nLULC Level_I:")
print(
    master["Level_I"]
    .value_counts(dropna=False)
    .to_string()
)


# ------------------------------------------------------------
# 20. LABEL DISTRIBUTION
# ------------------------------------------------------------

print("\n[18] Landslide/background label distribution...")

print(
    master["label"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)


# ------------------------------------------------------------
# 21. SAVE MASTER TABLE
# ------------------------------------------------------------

print("\n[19] Saving master factor table...")

master.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"  Saved: {OUTPUT_FILE}")


# ------------------------------------------------------------
# 22. SAVE MISSING-VALUE SUMMARY
# ------------------------------------------------------------

missing_output = (
    PROCESSED_DIR /
    "ner_master_missing_summary.csv"
)

missing_df.to_csv(
    missing_output,
    index=False
)

print(
    f"  Saved missing-value summary: "
    f"{missing_output}"
)


# ------------------------------------------------------------
# 23. FINAL VALIDATION
# ------------------------------------------------------------

print("\n[20] FINAL VALIDATION")

print(f"  Rows:    {len(master)}")
print(f"  Columns: {len(master.columns)}")

print("\nColumns:")
for i, col in enumerate(master.columns, start=1):
    print(f"  {i:>2}. {col}")


if len(master) != 1071:
    raise ValueError(
        "FINAL ERROR: Master table does not contain 1071 rows."
    )

if master["label"].isna().any():
    raise ValueError(
        "FINAL ERROR: Missing label values detected."
    )


print("\nFirst 5 rows:")
print(
    master.head()
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("STEP 42 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nImportant:")
print("- Geomorphology remains categorical.")
print("- LULC Level_I remains categorical.")
print("- Missing values are preserved as NaN.")
print("- No arbitrary numeric ranking has been assigned.")
print("- Lithology is not included because it is currently unavailable.")
print("=" * 70)