import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 52 — CREATE GEOMORPHOLOGICAL ORIGIN FACTOR")
print("=" * 70)

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_master_factors_9factor.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "processed"
    / "ner_master_factors_9factor_geomorph_origin.csv"
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("\n[1] Loading 9-factor master table...")

df = pd.read_csv(INPUT_FILE)

print(f"  Rows: {len(df)}")

if len(df) != 1071:
    raise ValueError(
        f"Expected 1071 rows, found {len(df)}"
    )

# ---------------------------------------------------------
# 2. CHECK GEOMORPHOLOGY COLUMN
# ---------------------------------------------------------
print("\n[2] Checking geomorphology column...")

if "geomorphology" not in df.columns:
    raise ValueError(
        "geomorphology column not found."
    )

print(
    f"  Valid geomorphology: "
    f"{df['geomorphology'].notna().sum()}"
)

print(
    f"  Missing geomorphology: "
    f"{df['geomorphology'].isna().sum()}"
)

# ---------------------------------------------------------
# 3. CREATE ORIGIN GROUP
# ---------------------------------------------------------
print("\n[3] Creating geomorphological origin...")

def extract_origin(value):

    if pd.isna(value):
        return pd.NA

    value = str(value)

    if "-" not in value:
        return pd.NA

    origin = value.split("-", 1)[0].strip()

    return origin


df["geomorph_origin"] = (
    df["geomorphology"]
    .apply(extract_origin)
)

# ---------------------------------------------------------
# 4. CHECK ORIGIN VALUES
# ---------------------------------------------------------
print("\n[4] Origin categories...")

origin_counts = (
    df["geomorph_origin"]
    .value_counts(dropna=False)
)

for origin, count in origin_counts.items():

    if pd.isna(origin):
        origin_name = "MISSING"
    else:
        origin_name = str(origin)

    percentage = (
        count / len(df) * 100
    )

    print(
        f"  {origin_name}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )

# ---------------------------------------------------------
# 5. VERIFY EXPECTED ORIGINS
# ---------------------------------------------------------
expected_origins = {
    "Structural Origin",
    "Fluvial Origin",
    "Denudational Origin",
    "Glacial Origin",
    "Water Bodies",
    "Lacustrine Origin"
}

observed_origins = set(
    df["geomorph_origin"]
    .dropna()
    .unique()
)

unexpected = observed_origins - expected_origins

if unexpected:

    raise ValueError(
        f"Unexpected geomorphological origins found: "
        f"{unexpected}"
    )

print("\n  ✓ All origins match the expected Bhuvan categories.")

# ---------------------------------------------------------
# 6. CHECK CONSISTENCY
# ---------------------------------------------------------
print("\n[5] Checking source-to-origin consistency...")

valid_geomorph = df[
    df["geomorphology"].notna()
].copy()

inconsistent = valid_geomorph[
    valid_geomorph["geomorph_origin"].isna()
]

print(
    f"  Inconsistent records: "
    f"{len(inconsistent)}"
)

if len(inconsistent) > 0:
    raise ValueError(
        "Some valid geomorphology records could not "
        "be assigned an origin."
    )

# ---------------------------------------------------------
# 7. LABEL CROSS-TABULATION
# ---------------------------------------------------------
print("\n[6] Geomorphological origin vs label...")
print("-" * 70)

origin_label = pd.crosstab(
    df["geomorph_origin"],
    df["label"],
    dropna=False
)

origin_label.columns = [
    "Background_0",
    "Landslide_1"
]

print(origin_label)

# ---------------------------------------------------------
# 8. SAVE
# ---------------------------------------------------------
print("\n[7] Saving updated master table...")

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"  Saved: {OUTPUT_FILE}")

# ---------------------------------------------------------
# 9. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 52 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nNew categorical factor:")
print("  geomorph_origin")

print("\nOriginal factor retained:")
print("  geomorphology")

print("\nNo numerical ranking was applied.")
print("No observations were deleted.")
print("No source geomorphology values were overwritten.")

print("\nOutput:")
print(f"  {OUTPUT_FILE}")

print("=" * 70)