import pandas as pd
import random

# ==================================================
# NER BOUNDING BOX
# ==================================================

SOUTH = 21.5
NORTH = 29.5
WEST = 88.0
EAST = 97.5

# Read NER landslide inventory
landslides = pd.read_csv(
    "raw_data/landslides/landslides_all_ner.csv"
)

print("NER landslide points:", len(landslides))

# Number of background points
# Use 3 negatives for every positive
target = len(landslides) * 3

print("Target background points:", target)

# ==================================================
# Generate random background points
# ==================================================

random.seed(42)

background = []

while len(background) < target:

    lat = random.uniform(SOUTH, NORTH)
    lon = random.uniform(WEST, EAST)

    # Keep background point away from known landslides
    too_close = False

    for _, row in landslides.iterrows():

        # Approximate degree distance
        dlat = lat - row["lat"]
        dlon = lon - row["lon"]

        distance_sq = dlat**2 + dlon**2

        # Approximately 5 km exclusion zone
        if distance_sq < 0.05**2:
            too_close = True
            break

    if not too_close:

        background.append({
            "lat": lat,
            "lon": lon,
            "date": "",
            "source": "Random NER background",
            "region": "NER (broad)",
            "label": 0
        })

background = pd.DataFrame(background)

# ==================================================
# Save
# ==================================================

output = (
    "raw_data/landslides/"
    "background_all_ner.csv"
)

background.to_csv(
    output,
    index=False
)

print("\n========================================")
print("NER BACKGROUND GENERATION COMPLETE")
print("========================================")

print("Background points:", len(background))
print("Saved:", output)

print("\nFirst 5:")
print(background.head())