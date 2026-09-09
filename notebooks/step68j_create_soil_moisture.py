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
    / "soil_moisture_3day_250m_raw.tif"
)


# ============================================================
# SETTINGS
# ============================================================

# Current SMAP L4 dataset
DATASET = "NASA/SMAP/SPL4SMGP/008"

# Last 3 complete days used for the rainfall period
START_DATE = "2026-07-29"
END_DATE = "2026-08-01"

# SMAP L4 surface soil moisture band
BAND = "sm_surface"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 68J — REAL NASA SMAP 3-DAY SOIL MOISTURE")
print("=" * 70)

print("Dataset:")
print(DATASET)

print()
print("Period:")
print(START_DATE, "to", "2026-07-31")

print()
print("Variable:")
print("Surface soil moisture")

print()
print("Source:")
print("NASA SMAP")


# ============================================================
# INITIALIZE EARTH ENGINE
# ============================================================

print()
print("Initializing Google Earth Engine...")

try:

    ee.Initialize(
        project="land-slide-research"
    )

except Exception:

    print("Starting Earth Engine authentication...")

    ee.Authenticate()

    ee.Initialize(
        project="land-slide-research"
    )

print("Earth Engine initialized successfully.")


# ============================================================
# LOAD NER
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
# LOAD REAL SMAP DATA
# ============================================================

print()
print("Loading NASA SMAP soil moisture...")

smap = (
    ee.ImageCollection(DATASET)
    .filterDate(
        START_DATE,
        END_DATE
    )
    .filterBounds(
        ner
    )
    .select(
        BAND
    )
)

count = smap.size().getInfo()

print(
    "SMAP images found:",
    count
)

if count == 0:

    raise RuntimeError(
        "No SMAP images were found for the selected period."
    )


# ============================================================
# CALCULATE 3-DAY MEAN
# ============================================================

print()
print(
    "Calculating 3-day mean soil moisture..."
)

soil_moisture_3day = (
    smap
    .mean()
    .rename("soil_moisture")
    .clip(ner)
)

print(
    "3-day soil moisture image created."
)


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if OUTPUT.exists():

    print()
    print("Removing previous output...")

    OUTPUT.unlink()


# ============================================================
# EXPORT REAL SMAP DATA
# ============================================================

print()
print("Exporting real NASA SMAP data...")

# Do NOT force EPSG:6933 here.
# Step 68K will reproject it to the exact
# 250 m model grid locally.

geemap.ee_export_image(
    soil_moisture_3day,
    filename=str(OUTPUT),
    scale=9000,
    region=ner.geometry(),
    file_per_band=False
)


# ============================================================
# VERIFY OUTPUT
# ============================================================

print()
print("Checking output file...")

if not OUTPUT.exists():

    raise RuntimeError(
        "Soil moisture export failed.\n"
        f"File was not created:\n{OUTPUT}"
    )

file_size = OUTPUT.stat().st_size

if file_size == 0:

    raise RuntimeError(
        "Soil moisture output file is empty."
    )

print(
    "Soil moisture file successfully created!"
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
print("SOIL MOISTURE EXPORT COMPLETE")
print("=" * 70)

print()
print("Source:")
print("NASA SMAP L4")

print()
print("Period:")
print(
    START_DATE,
    "to",
    "2026-07-31"
)

print()
print("Variable:")
print("3-day mean surface soil moisture")

print()
print("Output:")
print(OUTPUT)

print("=" * 70)