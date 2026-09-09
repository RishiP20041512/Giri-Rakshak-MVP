import pandas as pd
from pathlib import Path

print("=" * 70)
print("STEP 51 — REVIEW CATEGORICAL FACTORS")
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
    / "ner_categorical_frequency_audit.csv"
)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
print("\n[1] Loading 9-factor master table...")

df = pd.read_csv(INPUT_FILE)

print(f"  Total observations: {len(df)}")

if len(df) != 1071:
    raise ValueError(
        f"Expected 1071 observations, found {len(df)}"
    )

# ---------------------------------------------------------
# 2. DEFINE CATEGORICAL FACTORS
# ---------------------------------------------------------
categorical_factors = [
    "geomorphology",
    "Level_I",
]

print("\n[2] Categorical factors:")

for factor in categorical_factors:
    print(f"  - {factor}")

# ---------------------------------------------------------
# 3. FREQUENCY ANALYSIS
# ---------------------------------------------------------
audit_rows = []

for factor in categorical_factors:

    print("\n" + "-" * 70)
    print(f"FACTOR: {factor}")
    print("-" * 70)

    counts = (
        df[factor]
        .value_counts(dropna=False)
    )

    total_valid = df[factor].notna().sum()
    total_missing = df[factor].isna().sum()

    print(f"Valid observations : {total_valid}")
    print(f"Missing observations: {total_missing}")

    print("\nCategory frequencies:")

    for category, count in counts.items():

        if pd.isna(category):
            category_name = "MISSING"
        else:
            category_name = str(category)

        percentage = (
            count / len(df) * 100
        )

        print(
            f"  {category_name}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

        audit_rows.append({
            "factor": factor,
            "category": category_name,
            "count": int(count),
            "percentage": round(
                percentage,
                4
            ),
            "rare_less_than_5": bool(
                count < 5
            ),
            "rare_less_than_10": bool(
                count < 10
            )
        })

# ---------------------------------------------------------
# 4. GEOMORPHOLOGY RARE-CLASS REVIEW
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("GEOMORPHOLOGY RARE-CLASS REVIEW")
print("=" * 70)

geom_counts = (
    df["geomorphology"]
    .value_counts()
)

print("\nClasses with fewer than 5 observations:")

rare_under_5 = geom_counts[
    geom_counts < 5
]

if len(rare_under_5) == 0:

    print("  None")

else:

    for category, count in rare_under_5.items():
        print(
            f"  {category}: {count}"
        )

print("\nClasses with fewer than 10 observations:")

rare_under_10 = geom_counts[
    geom_counts < 10
]

if len(rare_under_10) == 0:

    print("  None")

else:

    for category, count in rare_under_10.items():
        print(
            f"  {category}: {count}"
        )

# ---------------------------------------------------------
# 5. LUCC RARE-CLASS REVIEW
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("LUCC RARE-CLASS REVIEW")
print("=" * 70)

lucc_counts = (
    df["Level_I"]
    .value_counts()
)

print("\nLUCC classes with fewer than 10 observations:")

lucc_rare = lucc_counts[
    lucc_counts < 10
]

if len(lucc_rare) == 0:

    print("  None")

else:

    for category, count in lucc_rare.items():
        print(
            f"  {category}: {count}"
        )

# ---------------------------------------------------------
# 6. CROSS-TABULATION WITH LABEL
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY VS LANDSLIDE LABEL")
print("=" * 70)

for factor in categorical_factors:

    print("\n" + "-" * 70)
    print(f"{factor} vs label")
    print("-" * 70)

    crosstab = pd.crosstab(
        df[factor],
        df["label"],
        dropna=False
    )

    crosstab.columns = [
        "Background_0",
        "Landslide_1"
    ]

    print(crosstab)

# ---------------------------------------------------------
# 7. GEOMORPHOLOGY ORIGIN REVIEW
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("GEOMORPHOLOGY ORIGIN REVIEW")
print("=" * 70)

geom_valid = df[
    df["geomorphology"].notna()
].copy()

geom_valid["origin"] = (
    geom_valid["geomorphology"]
    .str.split("-", n=1)
    .str[0]
)

print("\nGeomorphological origin groups:")

origin_counts = (
    geom_valid["origin"]
    .value_counts()
)

for origin, count in origin_counts.items():

    percentage = (
        count / len(geom_valid) * 100
    )

    print(
        f"  {origin}: "
        f"{count} "
        f"({percentage:.2f}%)"
    )

# ---------------------------------------------------------
# 8. SAVE AUDIT
# ---------------------------------------------------------
print("\n[8] Saving categorical frequency audit...")

audit_df = pd.DataFrame(
    audit_rows
)

audit_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"  Saved: {OUTPUT_FILE}")

# ---------------------------------------------------------
# 9. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 51 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nImportant:")
print("  No categorical classes were modified.")
print("  No categorical values were numerically ranked.")
print("  No observations were deleted.")
print("  This step is review only.")

print("\nOutput:")
print(f"  {OUTPUT_FILE}")

print("=" * 70)