import earthaccess
import json

with open("config.json") as f:
    config = json.load(f)

bbox = config["bbox"]

earthaccess.login()

results = earthaccess.search_data(
    short_name="SPL3SMP_E",
    bounding_box=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
    temporal=("2023-06-01", "2023-06-03")   # small test range first
)

print(f"Found {len(results)} files")
earthaccess.download(results, "raw_data/soil_moisture/")