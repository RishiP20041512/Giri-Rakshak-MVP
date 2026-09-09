import pandas as pd
import geopandas as gpd

# ==================================================
# NORTHEAST INDIA BOUNDING BOX
# ==================================================

NER_BBOX = {
    "south": 21.5,
    "north": 29.5,
    "west": 88.0,
    "east": 97.5
}

all_points = []

print("=" * 50)
print("STEP 6 - NORTHEAST INDIA LANDSLIDE INVENTORY")
print("=" * 50)

print("\nNER bounding box:")
print(f"South: {NER_BBOX['south']}")
print(f"North: {NER_BBOX['north']}")
print(f"West : {NER_BBOX['west']}")
print(f"East : {NER_BBOX['east']}")

# ==================================================
# 1. EXISTING PAPER POINTS
# ==================================================

paper = pd.read_csv(
    "raw_data/landslides/paper_points.csv"
)

paper["region"] = "Dibang Valley (paper)"

all_points.append(
    paper[["lat", "lon", "date", "source", "region"]]
)

print("\nPaper points:", len(paper))

# ==================================================
# 2. NASA GLOBAL LANDSLIDE CATALOG
# ==================================================

nasa_path = (
    "raw_data/landslides/"
    "nasa_global_shapefile/"
    "global_landslide_catalog_NASA.shp"
)

print("\nReading NASA Global Landslide Catalog...")

nasa_gdf = gpd.read_file(nasa_path)

print("Total NASA records:", len(nasa_gdf))

nasa_gdf = nasa_gdf.to_crs("EPSG:4326")

nasa = pd.DataFrame(
    nasa_gdf.drop(columns="geometry")
)

nasa["latitude"] = nasa_gdf.geometry.y
nasa["longitude"] = nasa_gdf.geometry.x

# Filter NASA to entire NER
nasa_ner = nasa[
    (nasa["latitude"] >= NER_BBOX["south"]) &
    (nasa["latitude"] <= NER_BBOX["north"]) &
    (nasa["longitude"] >= NER_BBOX["west"]) &
    (nasa["longitude"] <= NER_BBOX["east"])
].copy()

nasa_ner["lat"] = nasa_ner["latitude"]
nasa_ner["lon"] = nasa_ner["longitude"]

if "event_date" in nasa_ner.columns:
    nasa_ner["date"] = nasa_ner["event_date"]
else:
    nasa_ner["date"] = ""

nasa_ner["source"] = "NASA Global Landslide Catalog"
nasa_ner["region"] = "NER (broad)"

all_points.append(
    nasa_ner[["lat", "lon", "date", "source", "region"]]
)

print(
    "NASA catalog events inside NER:",
    len(nasa_ner)
)

# ==================================================
# 3. GFLD
# ==================================================

try:

    gfld_path = "raw_data/landslides/gfld_ner.geojson"

    gfld = gpd.read_file(gfld_path)

    gfld = gfld.to_crs("EPSG:4326")

    gfld["lat"] = gfld.geometry.y
    gfld["lon"] = gfld.geometry.x

    gfld = gfld[
        (gfld["lat"] >= NER_BBOX["south"]) &
        (gfld["lat"] <= NER_BBOX["north"]) &
        (gfld["lon"] >= NER_BBOX["west"]) &
        (gfld["lon"] <= NER_BBOX["east"])
    ].copy()

    if "ev_date" in gfld.columns:
        gfld["date"] = gfld["ev_date"]
    else:
        gfld["date"] = ""

    gfld["source"] = (
        "Froude & Petley GFLD "
        "(fatal events only)"
    )

    gfld["region"] = "NER (broad)"

    all_points.append(
        gfld[["lat", "lon", "date", "source", "region"]]
    )

    print("GFLD events inside NER:", len(gfld))

except Exception as e:

    print("\nGFLD not loaded.")
    print("Reason:", e)

# ==================================================
# 4. COMBINE DATASETS
# ==================================================

final = pd.concat(
    all_points,
    ignore_index=True
)

final = final.dropna(
    subset=["lat", "lon"]
)

final = final.drop_duplicates(
    subset=["lat", "lon"]
).reset_index(drop=True)

# ==================================================
# 5. SAVE
# ==================================================

output = (
    "raw_data/landslides/"
    "landslides_all_ner.csv"
)

final.to_csv(
    output,
    index=False
)

print("\n" + "=" * 50)
print("NER INVENTORY COMPLETE")
print("=" * 50)

print(
    "Total combined NER points:",
    len(final)
)

print("\nPoints by source:")
print(final["source"].value_counts())

print("\nPoints by region:")
print(final["region"].value_counts())

print("\nSaved:")
print(output)