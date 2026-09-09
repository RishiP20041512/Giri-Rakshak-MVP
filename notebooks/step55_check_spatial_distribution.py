import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("STEP 55 — CHECK SPATIAL DISTRIBUTION")
print("=" * 70)

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_final_modeling_table.csv"
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("\n[1] Loading final modeling table...")

df = pd.read_csv(INPUT_FILE)

print(f"  Observations: {len(df)}")

# ---------------------------------------------------------
# 2. COORDINATE EXTENT
# ---------------------------------------------------------
print("\n[2] Coordinate extent")
print("-" * 70)

print(
    f"  Latitude : "
    f"{df['lat'].min():.6f} "
    f"to "
    f"{df['lat'].max():.6f}"
)

print(
    f"  Longitude: "
    f"{df['lon'].min():.6f} "
    f"to "
    f"{df['lon'].max():.6f}"
)

# ---------------------------------------------------------
# 3. LABEL DISTRIBUTION
# ---------------------------------------------------------
print("\n[3] Label distribution")
print("-" * 70)

print(
    df["label"]
    .value_counts()
    .sort_index()
)

# ---------------------------------------------------------
# 4. UNIQUE COORDINATES
# ---------------------------------------------------------
print("\n[4] Coordinate uniqueness")
print("-" * 70)

duplicates = df.duplicated(
    subset=["lat", "lon"]
).sum()

print(f"  Duplicate coordinates: {duplicates}")

# ---------------------------------------------------------
# 5. APPROXIMATE SPATIAL GRID COUNTS
# ---------------------------------------------------------
print("\n[5] Testing candidate spatial grid sizes")
print("-" * 70)

# These are geographic-degree grid sizes.
# They are only used to inspect spatial grouping.
# Final spatial validation will use a metric CRS.

grid_sizes = [
    0.10,
    0.25,
    0.50,
    1.00
]

for grid_size in grid_sizes:

    grid_lat = np.floor(
        df["lat"] / grid_size
    ).astype(int)

    grid_lon = np.floor(
        df["lon"] / grid_size
    ).astype(int)

    groups = (
        grid_lat.astype(str)
        + "_"
        + grid_lon.astype(str)
    )

    n_groups = groups.nunique()

    group_sizes = (
        groups
        .value_counts()
    )

    print(
        f"\n  Grid size: "
        f"{grid_size:.2f}°"
    )

    print(
        f"    Spatial groups : "
        f"{n_groups}"
    )

    print(
        f"    Largest group  : "
        f"{group_sizes.max()}"
    )

    print(
        f"    Median group   : "
        f"{group_sizes.median():.1f}"
    )

    print(
        f"    Groups with 1 point: "
        f"{(group_sizes == 1).sum()}"
    )

# ---------------------------------------------------------
# 6. APPROXIMATE POINT SPACING
# ---------------------------------------------------------
print("\n[6] Approximate coordinate spacing")
print("-" * 70)

lat_sorted = np.sort(
    df["lat"].unique()
)

lon_sorted = np.sort(
    df["lon"].unique()
)

lat_diffs = np.diff(
    lat_sorted
)

lon_diffs = np.diff(
    lon_sorted
)

lat_diffs = lat_diffs[
    lat_diffs > 0
]

lon_diffs = lon_diffs[
    lon_diffs > 0
]

if len(lat_diffs) > 0:

    print(
        f"  Median latitude spacing : "
        f"{np.median(lat_diffs):.6f}°"
    )

if len(lon_diffs) > 0:

    print(
        f"  Median longitude spacing: "
        f"{np.median(lon_diffs):.6f}°"
    )

# ---------------------------------------------------------
# 7. STATE-LIKE REGIONAL DISTRIBUTION USING BBOX
# ---------------------------------------------------------
print("\n[7] Basic regional distribution")
print("-" * 70)

# Approximate broad Northeast zones for diagnostic purposes only.
# This does NOT replace the official state boundary.

regions = {
    "Western_NER": (
        (df["lon"] < 91.0)
    ),
    "Central_NER": (
        (df["lon"] >= 91.0)
        & (df["lon"] < 94.0)
    ),
    "Eastern_NER": (
        (df["lon"] >= 94.0)
    )
}

for region, mask in regions.items():

    subset = df.loc[mask]

    print(
        f"\n  {region}: "
        f"{len(subset)} observations"
    )

    if len(subset) > 0:

        print(
            subset["label"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

# ---------------------------------------------------------
# 8. FINAL
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 55 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNo train/test split was created yet.")
print("This step only evaluates spatial grouping options.")

print("=" * 70)