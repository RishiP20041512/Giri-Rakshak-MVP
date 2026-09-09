import requests

url = (
    "https://webgis1.nic.in/nicstreet/rest/services/"
    "admin2024/MapServer/9/query"
)

params = {
    "where": "1=1",
    "outFields": "*",
    "returnGeometry": "true",
    "f": "geojson"
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

with open(
    "raw_data/boundaries/india_states.geojson",
    "w",
    encoding="utf-8"
) as f:
    f.write(response.text)

print("State boundary downloaded.")
print("Status:", response.status_code)
print("Size:", len(response.text), "characters")