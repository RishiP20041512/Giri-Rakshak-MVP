import requests
import json
import os
import time

with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

bbox = config["bbox"]

# Split the large bounding box into 4 smaller boxes
mid_lat = (bbox["south"] + bbox["north"]) / 2
mid_lon = (bbox["west"] + bbox["east"]) / 2

boxes = [
    (bbox["south"], bbox["west"], mid_lat, mid_lon),
    (bbox["south"], mid_lon, mid_lat, bbox["east"]),
    (mid_lat, bbox["west"], bbox["north"], mid_lon),
    (mid_lat, mid_lon, bbox["north"], bbox["east"]),
]

url = "https://overpass.kumi.systems/api/interpreter"

headers = {
    "User-Agent": "GiriRakshak/1.0"
}

all_elements = []

for i, (south, west, north, east) in enumerate(boxes, start=1):

    print(f"\nRequesting area {i}/4...")
    print(f"BBox: {south}, {west}, {north}, {east}")

    query = f"""
    [out:json][timeout:120];
    (
      way["highway"]({south},{west},{north},{east});
      node["place"~"village|town|city"]({south},{west},{north},{east});
    );
    out geom;
    """

    try:
        response = requests.post(
            url,
            data={"data": query},
            headers=headers,
            timeout=180
        )

        print("HTTP status:", response.status_code)

        response.raise_for_status()

        data = response.json()

        elements = data.get("elements", [])

        print("Elements received:", len(elements))

        all_elements.extend(elements)

        # Don't hit the public server too quickly
        time.sleep(5)

    except requests.exceptions.RequestException as e:
        print(f"ERROR in area {i}: {e}")
        continue


# Remove duplicate OSM objects caused by box boundaries
unique = {}

for element in all_elements:
    key = (element["type"], element["id"])
    unique[key] = element

all_elements = list(unique.values())

output = {
    "version": 0.6,
    "generator": "GiriRakshak",
    "elements": all_elements
}

os.makedirs("raw_data/roads_villages", exist_ok=True)

output_file = "raw_data/roads_villages/osm_data.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f)

print("\n--------------------------------")
print("OSM DOWNLOAD COMPLETE")
print("--------------------------------")
print("Total unique elements:", len(all_elements))
print("Saved to:", output_file)