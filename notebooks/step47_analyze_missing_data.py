import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 47 — MISSING-DATA IMPACT ANALYSIS")
print("=" * 70)

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = BASE_DIR / "processed" / "ner_master_factors.csv"

df = pd.read_csv(INPUT_FILE)

factors = [
    "rainfall_3day",
    "soil_moisture",
    "ndvi",
    "distance_to_road_m",
    "lineament_density",
    "geomorphology",
    "Level_I",
]

print("\n[1] Original dataset")
print("-" * 70)
print(f"Total observations: {len(df)}")
print("\nLabels:")
print(df["label"].value_counts().sort_index())

# ---------------------------------------------------------
# COMPLETE CASE
# ---------------------------------------------------------

complete_mask = df[factors].notna().all(axis=1)

df["complete_case"] = complete_mask

print("\n[2] Complete-case comparison")
print("-" * 70)

for label_value, label_name in [(0, "Background"), (1, "Landslide")]:

    total = (df["label"] == label_value).sum()
    complete = (
        (df["label"] == label_value)
        & df["complete_case"]
    ).sum()

    excluded = total - complete

    print(f"\n{label_name}")
    print(f"  Original : {total}")
    print(f"  Complete : {complete}")
    print(f"  Excluded : {excluded}")
    print(f"  Retained : {complete / total * 100:.2f}%")

# ---------------------------------------------------------
# MISSINGNESS BY FACTOR
# ---------------------------------------------------------

print("\n[3] Missingness by factor")
print("-" * 70)

for factor in factors:

    missing = df[factor].isna()

    total_missing = missing.sum()

    if total_missing == 0:
        print(f"\n{factor}")
        print("  Missing: 0")
        continue

    landslide_missing = (
        missing & (df["label"] == 1)
    ).sum()

    background_missing = (
        missing & (df["label"] == 0)
    ).sum()

    print(f"\n{factor}")
    print(f"  Total missing     : {total_missing}")
    print(f"  Landslide missing : {landslide_missing}")
    print(f"  Background missing: {background_missing}")

# ---------------------------------------------------------
# STATE INFORMATION
# ---------------------------------------------------------

print("\n[4] Determining state distribution")
print("-" * 70)

boundary_file = BASE_DIR / "raw_data" / "boundaries" / "geoBoundaries-IND-ADM1.geojson"

try:

    import geopandas as gpd

    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(
            df["lon"],
            df["lat"]
        ),
        crs="EPSG:4326"
    )

    boundary = gpd.read_file(boundary_file)

    boundary = boundary[
        boundary["shapeName"].isin([
            "Arunachal Pradesh",
            "Assam",
            "Manipur",
            "Meghalaya",
            "Mizoram",
            "Nagaland",
            "Sikkim",
            "Tripura"
        ])
    ].copy()

    joined = gpd.sjoin(
        points,
        boundary[["shapeName", "geometry"]],
        how="left",
        predicate="within"
    )

    joined["state"] = joined["shapeName"]

    print("\nState distribution of all observations:")
    print(
        joined["state"]
        .value_counts(dropna=False)
    )

    print("\nState distribution of excluded observations:")
    print(
        joined.loc[
            ~joined["complete_case"],
            "state"
        ]
        .value_counts(dropna=False)
    )

    print("\nExcluded observations by state and label:")
    print(
        pd.crosstab(
            joined.loc[
                ~joined["complete_case"
                ],
                "state"
            ],
            joined.loc[
                ~joined["complete_case"
                ],
                "label"
            ]
        )
    )

except Exception as e:

    print("\nState analysis could not be completed.")
    print(f"Reason: {e}")

# ---------------------------------------------------------
# EXACT EXCLUDED RECORDS
# ---------------------------------------------------------

print("\n[5] Excluded observations")
print("-" * 70)

excluded = df.loc[
    ~df["complete_case"]
].copy()

print(f"Total excluded: {len(excluded)}")

print("\nExcluded records by missing factor combination:")

missing_pattern = (
    excluded[factors]
    .isna()
    .astype(int)
    .astype(str)
    .agg("".join, axis=1)
)

print(
    missing_pattern.value_counts()
)

# ---------------------------------------------------------
# SAVE AUDIT
# ---------------------------------------------------------

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_missing_data_impact_audit.csv"
)

excluded.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n[6] Saved audit")
print("-" * 70)
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STEP 47 COMPLETED")
print("=" * 70)