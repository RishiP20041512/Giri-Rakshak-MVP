import requests
import json
import os

# Load project configuration
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

bbox = config["bbox"]

# Put your NEW OpenTopography API key here
API_KEY = "32cc91911c632e5cdfc2c0e3ce67a96b"

url = "https://portal.opentopography.org/API/globaldem"

params = {
    "demtype": "SRTMGL1",
    "south": bbox["south"],
    "north": bbox["north"],
    "west": bbox["west"],
    "east": bbox["east"],
    "outputFormat": "GTiff",
    "API_Key": API_KEY
}

print("Downloading DEM...")
print("Bounding box:", bbox)

response = requests.get(
    url,
    params=params,
    timeout=180
)

print("HTTP status:", response.status_code)

response.raise_for_status()

# Make sure the folder exists
os.makedirs("raw_data/dem", exist_ok=True)

output_file = "raw_data/dem/dem.tif"

with open(output_file, "wb") as f:
    f.write(response.content)

print("DEM saved to:", output_file)
print("File size:", len(response.content), "bytes")