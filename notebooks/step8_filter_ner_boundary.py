import pandas as pd
import geopandas as gpd

# ==========================================================
# FILES
# ==========================================================

BOUNDARY_FILE = (
    "raw_data/boundaries/"
    "geoBoundaries-IND-ADM1.geojson"
)

LANDSLIDE_FILE = (
    "raw_data/landslides/"
    "landslides_all_ner.csv"
)

BACKGROUND_FILE = (
    "raw_data/landslides/"
    "background_all_ner.csv"
)

OUTPUT_LANDSLIDES = (
    "raw_data/landslides/"
    "landslides_ner_clean.csv"
)

OUTPUT_BACKGROUND = (
    "raw_data/landslides/"
    "background_ner_clean.csv"
)

# ==========================================================
# NER STATES
# ==========================================================

NER_STATES = [
    "Arunāchal Pradesh",
    "Assam",
    "Manipur",
    "Meghālaya",
    "Mizoram",
    "Nāgāland",
    "Sikkim",
    "Tripura"
]

# ==========================================================
# LOAD BOUNDARY
# ==========================================================

print("=" * 55)
print("STEP 8 - NER BOUNDARY FILTER")
print("=" * 55)

states = gpd.read_file(BOUNDARY_FILE)

print("\nBoundary states loaded:", len(states))

# Keep only the 8 NER states
ner_boundary = states[
    states["shapeName"].isin(NER_STATES)
].copy()

print("NER states selected:", len(ner_boundary))

print("\nSelected states:")
print(
    ner_boundary["shapeName"]
    .to_string(index=False)
)

# Combine the 8 states into one geometry
ner_geometry = ner_boundary.geometry.union_all()

# ==========================================================
# FUNCTION TO FILTER POINT DATA
# ==========================================================

def filter_points(input_file, output_file, name):

    df = pd.read_csv(input_file)

    print(f"\n{name} before filtering:", len(df))

    # Convert points to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(
            df["lon"],
            df["lat"]
        ),
        crs="EPSG:4326"
    )

    # Keep points inside actual NER boundary
    inside = gdf.geometry.within(ner_geometry)

    result = gdf[inside].copy()

    # Remove geometry before saving CSV
    result = pd.DataFrame(
        result.drop(columns="geometry")
    )

    result.to_csv(
        output_file,
        index=False
    )

    print(f"{name} inside NER:", len(result))
    print(
        f"{name} removed:",
        len(df) - len(result)
    )

    return result


# ==========================================================
# FILTER LANDSLIDES
# ==========================================================

landslides = filter_points(
    LANDSLIDE_FILE,
    OUTPUT_LANDSLIDES,
    "Landslide points"
)

# ==========================================================
# FILTER BACKGROUND
# ==========================================================

background = filter_points(
    BACKGROUND_FILE,
    OUTPUT_BACKGROUND,
    "Background points"
)

# ==========================================================
# SUMMARY
# ==========================================================

print("\n" + "=" * 55)
print("NER BOUNDARY FILTER COMPLETE")
print("=" * 55)

print("\nFinal landslides:", len(landslides))
print("Final background:", len(background))
print(
    "Total points:",
    len(landslides) + len(background)
)

print("\nSaved:")
print(OUTPUT_LANDSLIDES)
print(OUTPUT_BACKGROUND)