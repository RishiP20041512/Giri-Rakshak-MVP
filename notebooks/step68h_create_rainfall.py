from pathlib import Path

import ee
import geemap


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

BOUNDARY = (
    ROOT
    / "raw_data"
    / "boundaries"
    / "ner_8states.geojson"
)

OUTPUT_DIR = (
    ROOT
    / "processed"
    / "predictors"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = (
    OUTPUT_DIR
    / "rainfall_3day_250m_raw.tif"
)


# ============================================================
# SETTINGS
# ============================================================

# Last 3 complete CHIRPS daily images currently confirmed
START_DATE = "2026-07-29"
END_DATE = "2026-08-01"

# These are used later when the rainfall raster
# is aligned to the exact 250 m model grid.
TARGET_CRS = "EPSG:6933"
TARGET_SCALE = 250


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 68H — REAL CHIRPS 3-DAY RAINFALL")
print("=" * 70)

print("Dataset:")
print("UCSB-CHG/CHIRPS/DAILY")

print()

print("Rainfall period:")
print(START_DATE, "to", "2026-07-31")

print()

print("Target model CRS:", TARGET_CRS)
print("Target model resolution:", TARGET_SCALE, "m")

print()


# ============================================================
# INITIALIZE GOOGLE EARTH ENGINE
# ============================================================

print("Initializing Google Earth Engine...")

try:

    ee.Initialize(
        project="land-slide-research"
    )

except Exception:

    print("Earth Engine is not initialized.")
    print("Starting authentication...")

    ee.Authenticate()

    ee.Initialize(
        project="land-slide-research"
    )

print("Earth Engine initialized successfully.")


# ============================================================
# LOAD NER BOUNDARY
# ============================================================

print()
print("Loading NER boundary...")

if not BOUNDARY.exists():

    raise FileNotFoundError(
        f"NER boundary not found:\n{BOUNDARY}"
    )

ner = geemap.geojson_to_ee(
    str(BOUNDARY)
)

print("NER boundary loaded.")


# ============================================================
# LOAD REAL CHIRPS DAILY DATA
# ============================================================

print()
print("Loading CHIRPS rainfall...")

chirps = (
    ee.ImageCollection(
        "UCSB-CHG/CHIRPS/DAILY"
    )
    .filterDate(
        START_DATE,
        END_DATE
    )
    .filterBounds(
        ner
    )
    .select(
        "precipitation"
    )
)

count = chirps.size().getInfo()

print(
    "CHIRPS images found:",
    count
)

if count != 3:

    raise RuntimeError(
        "Expected 3 daily CHIRPS images "
        f"but found {count}."
    )


# ============================================================
# CALCULATE 3-DAY ACCUMULATED RAINFALL
# ============================================================

print()
print(
    "Calculating 3-day accumulated rainfall..."
)

rainfall_3day = (
    chirps
    .sum()
    .rename("rainfall_3day")
)

# Clip to the real 8-state NER boundary
rainfall_3day = rainfall_3day.clip(
    ner
)

print(
    "3-day rainfall image created."
)


# ============================================================
# REMOVE OLD FAILED OUTPUT
# ============================================================

if OUTPUT.exists():

    print()
    print(
        "Removing previous failed rainfall file..."
    )

    OUTPUT.unlink()


# ============================================================
# EXPORT REAL CHIRPS DATA
# ============================================================

print()
print(
    "Exporting real rainfall data..."
)

print(
    "Exporting at native CHIRPS-scale resolution..."
)

# IMPORTANT:
#
# We intentionally DO NOT specify:
#
#     crs="EPSG:6933"
#
# here.
#
# The previous export failed because the GEE/geemap
# download path could not parse EPSG:6933.
#
# Step 68I will later reproject this real rainfall
# raster to the exact 250 m EPSG:6933 model grid.
#
# No synthetic/fake rainfall is being created.

geemap.ee_export_image(
    rainfall_3day,
    filename=str(OUTPUT),
    scale=5566,
    region=ner.geometry(),
    file_per_band=False
)


# ============================================================
# VERIFY ACTUAL FILE
# ============================================================

print()
print(
    "Checking whether rainfall TIFF was created..."
)

if not OUTPUT.exists():

    raise RuntimeError(
        "Rainfall export failed.\n\n"
        "The Earth Engine export did not create:\n"
        f"{OUTPUT}\n\n"
        "Do NOT continue to Step 68I."
    )


# Check that the file is not empty
file_size = OUTPUT.stat().st_size

if file_size == 0:

    raise RuntimeError(
        "Rainfall export created an empty file:\n"
        f"{OUTPUT}"
    )


print(
    "Rainfall file successfully created!"
)

print(
    "File size:",
    round(file_size / (1024 * 1024), 2),
    "MB"
)

print(
    "Output:",
    OUTPUT
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("RAINFALL EXPORT COMPLETE")
print("=" * 70)

print()
print("Source:")
print(
    "UCSB-CHG/CHIRPS/DAILY"
)

print()
print("Period:")
print(
    START_DATE,
    "to",
    "2026-07-31"
)

print()
print("Variable:")
print(
    "3-day accumulated precipitation"
)

print()
print("Units:")
print("mm")

print()
print("Raw rainfall output:")
print(OUTPUT)

print()
print(
    "Next step: Step 68I will align this "
    "real rainfall raster to the exact "
    "250 m NER model grid."
)

print()
print("=" * 70)