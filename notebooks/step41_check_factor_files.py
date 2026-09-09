# ============================================================
# STEP 41A — CHECK ALL NER CONDITIONING FACTOR FILES
# ============================================================
#
# Purpose:
#   Audit all currently available conditioning-factor datasets
#   before creating the master modelling table.
#
# Factors currently included:
#
#   1. Elevation
#   2. Slope
#   3. Rainfall
#   4. Soil moisture
#   5. Geomorphology
#   6. LUCC
#   7. NDVI
#   8. Distance to road
#   9. Lineament density
#
# Lithology:
#   EXCLUDED FOR NOW because the required GSI/NGDR data
#   has not yet been obtained.
#
# Aspect:
#   NOT INCLUDED.
#
# Distance to river:
#   NOT INCLUDED.
#
# Important:
#   This script ONLY audits the existing files.
#   It does NOT modify any factor data.
#
# ============================================================

from pathlib import Path

import pandas as pd
import geopandas as gpd


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED = PROJECT_ROOT / "processed"


# ============================================================
# 2. EXPECTED FILES
# ============================================================

FILES = {
    "training_points": (
        PROCESSED / "ner_training_points.geojson"
    ),

    "rainfall": (
        PROCESSED / "ner_rainfall.csv"
    ),

    "soil_moisture": (
        PROCESSED / "ner_soil_moisture.csv"
    ),

    "geomorphology": (
        PROCESSED / "ner_geomorphology.csv"
    ),

    "lulc": (
        PROCESSED / "ner_lulc.csv"
    ),

    "ndvi": (
        PROCESSED / "ner_ndvi.csv"
    ),

    "distance_to_road": (
        PROCESSED / "ner_distance_to_road.csv"
    ),

    "lineament_density": (
        PROCESSED / "ner_lineament_density.csv"
    ),
}


# ============================================================
# 3. EXPECTED FACTOR COLUMNS
# ============================================================
#
# These are the ACTUAL column names found in the current
# processed CSV files.
#
# ============================================================

EXPECTED_COLUMNS = {

    "rainfall":
        "rainfall_3day",

    "soil_moisture":
        "soil_moisture",

    "geomorphology":
        "Level_I",

    "lulc":
        "Level_I",

    "ndvi":
        "ndvi",

    "distance_to_road":
        "distance_to_road_m",

    "lineament_density":
        "lineament_density",
}


# ============================================================
# 4. CONTINUOUS FACTORS
# ============================================================

CONTINUOUS_FACTORS = {

    "rainfall":
        "rainfall_3day",

    "soil_moisture":
        "soil_moisture",

    "ndvi":
        "ndvi",

    "distance_to_road":
        "distance_to_road_m",

    "lineament_density":
        "lineament_density",
}


# ============================================================
# 5. CATEGORICAL FACTORS
# ============================================================

CATEGORICAL_FACTORS = {

    "geomorphology":
        "Level_I",

    "lulc":
        "Level_I",
}


# ============================================================
# 6. EXPECTED NUMBER OF TRAINING POINTS
# ============================================================

EXPECTED_ROWS = 1071


# ============================================================
# 7. HEADER
# ============================================================

print("=" * 75)
print("STEP 41A — CONDITIONING FACTOR FILE AUDIT")
print("=" * 75)


# ============================================================
# 8. CHECK FILE EXISTENCE
# ============================================================

print("\n" + "-" * 75)
print("FILE EXISTENCE CHECK")
print("-" * 75)

all_files_found = True

for name, path in FILES.items():

    if path.exists():

        print(
            f"[FOUND]   {name:<22} {path.name}"
        )

    else:

        print(
            f"[MISSING] {name:<22} {path}"
        )

        all_files_found = False


if not all_files_found:

    raise FileNotFoundError(
        "\nOne or more required files are missing. "
        "Check the paths above."
    )


# ============================================================
# 9. READ TRAINING POINTS
# ============================================================

print("\n" + "-" * 75)
print("TRAINING POINT CHECK")
print("-" * 75)

training_points = gpd.read_file(
    FILES["training_points"]
)

print(
    "Training points:",
    len(training_points)
)

print(
    "CRS:",
    training_points.crs
)

if len(training_points) != EXPECTED_ROWS:

    print(
        f"WARNING: Expected {EXPECTED_ROWS} "
        f"training points but found "
        f"{len(training_points)}."
    )

else:

    print(
        "Training-point count: [OK]"
    )


# ============================================================
# 10. READ ALL FACTOR CSVs
# ============================================================

print("\n" + "-" * 75)
print("READING FACTOR FILES")
print("-" * 75)

data = {}

for name, path in FILES.items():

    if name == "training_points":
        continue

    print(
        f"\nReading {name}..."
    )

    df = pd.read_csv(path)

    data[name] = df

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {df.columns.tolist()}"
    )


# ============================================================
# 11. ROW COUNT CHECK
# ============================================================

print("\n" + "-" * 75)
print("ROW-COUNT CHECK")
print("-" * 75)

for name, df in data.items():

    if len(df) == EXPECTED_ROWS:

        status = "OK"

    else:

        status = "MISMATCH"

    print(
        f"{name:<22} "
        f"{len(df):>5} rows   [{status}]"
    )


# ============================================================
# 12. EXPECTED COLUMN CHECK
# ============================================================

print("\n" + "-" * 75)
print("EXPECTED COLUMN CHECK")
print("-" * 75)

column_check_passed = True

for name, expected_column in EXPECTED_COLUMNS.items():

    df = data[name]

    if expected_column in df.columns:

        print(
            f"{name:<22} "
            f"{expected_column:<30} [FOUND]"
        )

    else:

        print(
            f"{name:<22} "
            f"{expected_column:<30} [MISSING]"
        )

        column_check_passed = False


# ============================================================
# 13. MISSING VALUE CHECK
# ============================================================

print("\n" + "-" * 75)
print("MISSING-VALUE CHECK")
print("-" * 75)

missing_summary = []

for name, column in EXPECTED_COLUMNS.items():

    df = data[name]

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if column not in df.columns:

        print(
            f"{name:<22} "
            f"[COLUMN NOT FOUND — SKIPPED]"
        )

        missing_summary.append(
            {
                "factor": name,
                "column": column,
                "valid": None,
                "missing": None,
                "missing_percent": None,
            }
        )

        continue

    # --------------------------------------------------------
    # Calculate missing values
    # --------------------------------------------------------

    missing = int(
        df[column].isna().sum()
    )

    valid = int(
        len(df) - missing
    )

    percentage = (
        missing / len(df) * 100
        if len(df) > 0
        else 0
    )

    missing_summary.append(
        {
            "factor": name,
            "column": column,
            "valid": valid,
            "missing": missing,
            "missing_percent": percentage,
        }
    )

    print(
        f"{name:<22} "
        f"valid={valid:>5} "
        f"missing={missing:>5} "
        f"({percentage:>6.2f}%)"
    )


# ============================================================
# 14. COORDINATE CONSISTENCY CHECK
# ============================================================

print("\n" + "-" * 75)
print("COORDINATE CONSISTENCY CHECK")
print("-" * 75)

# We use rainfall as the reference factor because it contains
# all 1071 records and has lat/lon columns.

reference_name = "rainfall"

reference_df = data[reference_name]

if not {
    "lat",
    "lon"
}.issubset(reference_df.columns):

    print(
        "Rainfall does not contain lat/lon columns."
    )

else:

    reference_coords = list(
        zip(
            reference_df["lat"].round(8),
            reference_df["lon"].round(8)
        )
    )

    print(
        "Reference dataset: rainfall"
    )

    for name, df in data.items():

        if not {
            "lat",
            "lon"
        }.issubset(df.columns):

            print(
                f"{name:<22} "
                "[NO LAT/LON COLUMNS]"
            )

            continue

        coords = list(
            zip(
                df["lat"].round(8),
                df["lon"].round(8)
            )
        )

        if coords == reference_coords:

            print(
                f"{name:<22} [MATCH]"
            )

        else:

            print(
                f"{name:<22} [MISMATCH]"
            )


# ============================================================
# 15. DUPLICATE COORDINATE CHECK
# ============================================================

print("\n" + "-" * 75)
print("DUPLICATE COORDINATE CHECK")
print("-" * 75)

for name, df in data.items():

    if not {
        "lat",
        "lon"
    }.issubset(df.columns):

        print(
            f"{name:<22} "
            "[NO LAT/LON COLUMNS]"
        )

        continue

    duplicates = int(
        df.duplicated(
            subset=[
                "lat",
                "lon"
            ]
        ).sum()
    )

    if duplicates == 0:

        status = "OK"

    else:

        status = "CHECK"


    print(
        f"{name:<22} "
        f"duplicate coordinates={duplicates:<5} "
        f"[{status}]"
    )


# ============================================================
# 16. CONTINUOUS FACTOR STATISTICS
# ============================================================

print("\n" + "-" * 75)
print("CONTINUOUS FACTOR STATISTICS")
print("-" * 75)

for name, column in CONTINUOUS_FACTORS.items():

    df = data[name]

    if column not in df.columns:

        print(
            f"\n{name}"
        )

        print(
            "  Column not found."
        )

        continue

    series = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    print(
        f"\n{name}"
    )

    print(
        f"  column = {column}"
    )

    print(
        f"  valid  = {series.notna().sum()}"
    )

    print(
        f"  missing = {series.isna().sum()}"
    )

    if series.notna().any():

        print(
            f"  min    = {series.min()}"
        )

        print(
            f"  max    = {series.max()}"
        )

        print(
            f"  mean   = {series.mean()}"
        )

        print(
            f"  median = {series.median()}"
        )

        print(
            f"  std    = {series.std()}"
        )


# ============================================================
# 17. CATEGORICAL FACTOR CLASS COUNTS
# ============================================================

print("\n" + "-" * 75)
print("CATEGORICAL FACTOR CLASS COUNTS")
print("-" * 75)

for name, column in CATEGORICAL_FACTORS.items():

    df = data[name]

    print(
        f"\n{name} — {column}"
    )

    if column not in df.columns:

        print(
            "  Column not found."
        )

        continue

    counts = (
        df[column]
        .value_counts(
            dropna=False
        )
    )

    print(
        counts.to_string()
    )


# ============================================================
# 18. MISSING VALUE SUMMARY TABLE
# ============================================================

print("\n" + "-" * 75)
print("MISSING VALUE SUMMARY TABLE")
print("-" * 75)

missing_df = pd.DataFrame(
    missing_summary
)

print(
    missing_df.to_string(
        index=False
    )
)


# ============================================================
# 19. SAVE AUDIT SUMMARY
# ============================================================

audit_output = (
    PROCESSED
    / "ner_factor_audit_summary.csv"
)

missing_df.to_csv(
    audit_output,
    index=False
)

print(
    "\nAudit summary saved:"
)

print(
    audit_output
)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL AUDIT SUMMARY")
print("=" * 75)

print(
    f"Expected training points : {EXPECTED_ROWS}"
)

print(
    f"Actual training points   : {len(training_points)}"
)

print(
    "Factors currently included: 9"
)

print(
    "Lithology                : EXCLUDED FOR NOW"
)

print(
    "Aspect                   : NOT INCLUDED"
)

print(
    "Distance to river        : NOT INCLUDED"
)

print(
    "Mizoram lineament data   : 74 NoData values"
)

print(
    "LULC Level_I             : Used as final LUCC variable"
)

print(
    "Geomorphology Level_I    : Used as categorical variable"
)


# ============================================================
# 21. OVERALL STATUS
# ============================================================

print("\n" + "-" * 75)
print("OVERALL STATUS")
print("-" * 75)

if (
    len(training_points) == EXPECTED_ROWS
    and column_check_passed
):

    print(
        "[PASS] Required files and factor columns found."
    )

else:

    print(
        "[CHECK REQUIRED] One or more structural "
        "checks need attention."
    )


print("\nNext stage:")
print(
    "Create the master factor table."
)

print(
    "Then perform missing-data and multicollinearity checks "
    "before Random Forest modelling."
)

print("=" * 75)
print("STEP 41A COMPLETED")
print("=" * 75)