import requests
import json
import time


# ============================================================
# CONFIG
# ============================================================

with open("config.json", encoding="utf-8") as f:
    config = json.load(f)

bbox = config["bbox"]

south = bbox["south"]
west = bbox["west"]
north = bbox["north"]
east = bbox["east"]


# ============================================================
# OVERPASS QUERY
# ============================================================

query = f"""
[out:json][timeout:180];

(
  way["highway"]({south},{west},{north},{east});
  way["waterway"]({south},{west},{north},{east});
  node["place"~"village|town|city"]({south},{west},{north},{east});
);

out geom;
"""


# ============================================================
# MULTIPLE SERVERS
# ============================================================

servers = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter"
]


# ============================================================
# DOWNLOAD
# ============================================================

output = "raw_data/roads_villages/osm_data.json"

success = False


for server in servers:

    print("\n========================================")
    print("Trying:")
    print(server)
    print("========================================")

    try:

        response = requests.post(
            server,
            data={"data": query},
            timeout=240,
            headers={
                "User-Agent":
                "GiriRakshak/1.0 research prototype"
            }
        )

        print(
            "HTTP status:",
            response.status_code
        )

        if response.status_code != 200:

            print(
                "Server returned an error."
            )

            continue


        data = response.json()

        elements = data.get(
            "elements",
            []
        )

        print(
            "Elements received:",
            len(elements)
        )


        if len(elements) == 0:

            print(
                "Server returned zero elements."
            )

            continue


        # Save actual OSM data

        with open(
            output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f
            )


        print(
            "\nOSM DATA SUCCESSFULLY SAVED"
        )

        print(
            "File:",
            output
        )

        print(
            "Elements:",
            len(elements)
        )


        success = True

        break


    except Exception as e:

        print(
            "Error:",
            repr(e)
        )

        time.sleep(2)


# ============================================================
# FINAL RESULT
# ============================================================

if not success:

    print("\n========================================")
    print("OSM DOWNLOAD FAILED")
    print("========================================")

    print(
        "All Overpass servers failed."
    )

    print(
        "Do not continue with the OSM overlay yet."
    )

    raise SystemExit(1)


print("\n========================================")
print("STEP 29A COMPLETE")
print("========================================")