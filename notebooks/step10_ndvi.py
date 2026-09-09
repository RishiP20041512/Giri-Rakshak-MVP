



import json
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType

with open("config.json") as f:
    config_data = json.load(f)

bbox_vals = config_data["bbox"]

config = SHConfig()
config.sh_client_id = "sh-ba99c9d1-bcbe-4889-a938-a78dadf28999"
config.sh_client_secret = "5BLmw54sjWkywflSNxMsa7n1oGfKeHbr"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"

# This line is the fix — rebinds the collection to the CDSE endpoint
collection = DataCollection.SENTINEL2_L2A.define_from("s2l2a_cdse", service_url=config.sh_base_url)

bbox = BBox(bbox=[bbox_vals["west"], bbox_vals["south"], bbox_vals["east"], bbox_vals["north"]], crs=CRS.WGS84)

evalscript = """
//VERSION=3
function setup() {
  return { input: ["B04","B08"], output: { bands: 1, sampleType: "FLOAT32" } };
}
function evaluatePixel(sample) {
  return [(sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.0001)];
}
"""

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[SentinelHubRequest.input_data(
        data_collection=collection,   # <-- use the redefined collection here, not DataCollection.SENTINEL2_L2A
        time_interval=("2023-06-01", "2023-06-30"),
        mosaicking_order="leastCC"
    )],
    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
    bbox=bbox,
    size=(512, 512),
    config=config,
    data_folder="raw_data/satellite/"
)

data = request.get_data(save_data=True)
print("NDVI data saved to raw_data/satellite/")