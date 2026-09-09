import requests

WMS_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"

LAYER = "sisdp_phase2:SISDP_P2_LULC_10K_2016_2019_AR"

# Test point inside Arunachal Pradesh
lon = 92.63415
lat = 27.45

# Small bounding box around the point
delta = 0.01

bbox = (
    f"{lon-delta},"
    f"{lat-delta},"
    f"{lon+delta},"
    f"{lat+delta}"
)

params = {
    "SERVICE": "WMS",
    "VERSION": "1.1.1",
    "REQUEST": "GetFeatureInfo",

    "LAYERS": LAYER,

    "QUERY_LAYERS": LAYER,

    "STYLES": "",

    "SRS": "EPSG:4326",

    "BBOX": bbox,

    "WIDTH": 101,

    "HEIGHT": 101,

    "X": 50,

    "Y": 50,

    "INFO_FORMAT": "text/html",
}

print("Querying Bhuvan LULC...")

print("Layer:")
print(LAYER)

print()

print("Test point:")
print("Longitude:", lon)
print("Latitude :", lat)

print()

try:

    response = requests.get(
        WMS_URL,
        params=params,
        timeout=60
    )

    print("HTTP status:", response.status_code)

    print(
        "Content-Type:",
        response.headers.get("Content-Type")
    )

    print()

    print("----- Bhuvan Response -----")

    print(response.text[:10000])

    print("---------------------------")


except Exception as e:

    print("ERROR:")
    print(e)