import requests

WMS_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
LAYER = "geomorphology:AR_GM50K_0506"

# A test point inside Arunachal Pradesh
lon = 92.63415
lat = 27.45

# Small bounding box around the test point
delta = 0.01

bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

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

print("Querying Bhuvan...")
print("Layer:", LAYER)
print("Point:", lon, lat)

response = requests.get(
    WMS_URL,
    params=params,
    timeout=60
)

print("\nHTTP status:", response.status_code)
print("Content-Type:", response.headers.get("Content-Type"))

print("\n----- Bhuvan Response -----")
print(response.text[:5000])
print("---------------------------")