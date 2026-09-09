import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 49 — INTEGRATE ELEVATION AND SLOPE")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

MASTER_FILE = (
    BASE_DIR
    / "processed"
    / "ner_master_factors.csv"
)

TERRAIN_FILE = (
    BASE_DIR
    / "processed"
    / "ner_elevation_slope.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_master_factors_9factor.csv"
)

# ---------------------------------------------------------
# 1. LOAD FILES
# ---------------------------------------------------------
print("\n[1] Loading files...")

master = pd.read_csv(MASTER_FILE)
terrain = pd.read_csv(TERRAIN_FILE)

print(f"  Master rows : {len(master)}")
print(f"  Terrain rows: {len(terrain)}")

if len(master) != 1071:
    raise ValueError(
        f"Expected 1071 master rows, found {len(master)}"
    )

if len(terrain) != 1071:
    raise ValueError(
        f"Expected 1071 terrain rows, found {len(terrain)}"
    )

# ---------------------------------------------------------
# 2. CHECK COORDINATES
# ---------------------------------------------------------
print("\n[2] Checking coordinate alignment...")

master_coords = master[["lat", "lon"]].copy()
terrain_coords = terrain[["lat", "lon"]].copy()

master_coords = master_coords.sort_values(
    ["lat", "lon"]
).reset_index(drop=True)

terrain_coords = terrain_coords.sort_values(
    ["lat", "lon"]
).reset_index(drop=True)

lat_match = (
    master_coords["lat"]
    .round(8)
    .equals(
        terrain_coords["lat"].round(8)
    )
)

lon_match = (
    master_coords["lon"]
    .round(8)
    .equals(
        terrain_coords["lon"].round(8)
    )
)

print(f"  Latitude coordinates match : {lat_match}")
print(f"  Longitude coordinates match: {lon_match}")

if not lat_match or not lon_match:
    raise ValueError(
        "Master and terrain coordinates do not match."
    )

# ---------------------------------------------------------
# 3. CHECK TERRAIN VALUES
# ---------------------------------------------------------
print("\n[3] Checking terrain values...")

if terrain["elevation_m"].isna().any():
    raise ValueError(
        "Missing elevation values detected."
    )

if terrain["slope_deg"].isna().any():
    raise ValueError(
        "Missing slope values detected."
    )

invalid_slope = terrain[
    (terrain["slope_deg"] < 0)
    | (terrain["slope_deg"] > 90)
]

if len(invalid_slope) > 0:
    raise ValueError(
        "Invalid slope values detected."
    )

print("  Elevation: 1071 valid")
print("  Slope    : 1071 valid")

# ---------------------------------------------------------
# 4. REMOVE OLD TERRAIN COLUMNS IF PRESENT
# ---------------------------------------------------------
print("\n[4] Preparing master table...")

for column in ["elevation_m", "slope_deg"]:
    if column in master.columns:
        master = master.drop(columns=[column])

# ---------------------------------------------------------
# 5. MERGE BY COORDINATES
# ---------------------------------------------------------
print("\n[5] Merging terrain factors...")

terrain_subset = terrain[
    [
        "lat",
        "lon",
        "elevation_m",
        "slope_deg"
    ]
].copy()

merged = master.merge(
    terrain_subset,
    on=["lat", "lon"],
    how="left",
    validate="one_to_one"
)

print(f"  Rows after merge: {len(merged)}")

if len(merged) != len(master):
    raise ValueError(
        "Row count changed during merge."
    )

# ---------------------------------------------------------
# 6. FINAL FACTOR LIST
# ---------------------------------------------------------
print("\n[6] Checking 9-factor structure...")

factors = [
    "elevation_m",
    "slope_deg",
    "rainfall_3day",
    "soil_moisture",
    "geomorphology",
    "Level_I",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
]

for factor in factors:

    if factor not in merged.columns:
        raise ValueError(
            f"Missing final factor: {factor}"
        )

    print(f"  ✓ {factor}")

# ---------------------------------------------------------
# 7. MISSINGNESS
# ---------------------------------------------------------
print("\n[7] Final missing-value audit...")

for factor in factors:

    missing = merged[factor].isna().sum()

    print(
        f"  {factor}: "
        f"{missing} missing"
    )

# ---------------------------------------------------------
# 8. CHECK DUPLICATES
# ---------------------------------------------------------
print("\n[8] Checking duplicate coordinates...")

duplicates = merged.duplicated(
    subset=["lat", "lon"]
).sum()

print(f"  Duplicate coordinates: {duplicates}")

if duplicates > 0:
    raise ValueError(
        "Duplicate coordinates detected."
    )

# ---------------------------------------------------------
# 9. SAVE
# ---------------------------------------------------------
print("\n[9] Saving 9-factor master table...")

merged.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"  Saved: {OUTPUT_FILE}")

# ---------------------------------------------------------
# 10. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 49 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal available conditioning factors: 9")

for i, factor in enumerate(factors, start=1):
    print(f"  {i}. {factor}")

print("\nTemporarily unavailable:")
print("  10. lithology")

print("\nObservations:")
print(f"  Total rows: {len(merged)}")

print("\nLabels:")
print(
    merged["label"]
    .value_counts()
    .sort_index()
)

print("\nOutput:")
print(f"  {OUTPUT_FILE}")

print("=" * 70)