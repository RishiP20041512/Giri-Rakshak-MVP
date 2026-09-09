import earthaccess
import json

with open("config.json") as f:
    config = json.load(f)

bbox = config["bbox"]

earthaccess.login()  # prompts for your urs.earthdata.nasa.gov username/password the first time

results = earthaccess.search_data(
    short_name="GPM_3IMERGDF",
    bounding_box=(bbox["west"], bbox["south"], bbox["east"], bbox["north"]),
    temporal=("2023-06-01", "2023-06-03")  # monsoon season - adjust years if you want more history
)

print(f"Found {len(results)} files")
earthaccess.download(results, "raw_data/rainfall/")