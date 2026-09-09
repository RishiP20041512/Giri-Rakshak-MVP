import pandas as pd
import geopandas as gpd
import os

paper_points = pd.read_csv("raw_data/landslides/paper_points.csv")

coolr_path = "raw_data/landslides/coolr_points.geojson"
if os.path.exists(coolr_path) and os.path.getsize(coolr_path) > 50:
    try:
        gdf = gpd.read_file(coolr_path)
        if len(gdf) > 0:
            gdf["lat"] = gdf.geometry.y
            gdf["lon"] = gdf.geometry.x
            date_col = None
            for c in gdf.columns:
                if "date" in c.lower():
                    date_col = c
                    break
            coolr_df = pd.DataFrame({
                "lat": gdf["lat"],
                "lon": gdf["lon"],
                "date": gdf[date_col] if date_col else "",
                "source": "NASA COOLR"
            })
            print(f"Found {len(coolr_df)} points from COOLR")
        else:
            coolr_df = pd.DataFrame(columns=["lat","lon","date","source"])
            print("COOLR returned 0 points for this area")
    except Exception as e:
        print(f"Could not read COOLR file: {e}")
        coolr_df = pd.DataFrame(columns=["lat","lon","date","source"])
else:
    print("No COOLR file found or file is empty — using paper points only")
    coolr_df = pd.DataFrame(columns=["lat","lon","date","source"])

final = pd.concat([paper_points, coolr_df], ignore_index=True)
final.to_csv("raw_data/landslides/landslides.csv", index=False)
print(f"\nTotal landslide points saved: {len(final)}")
print(final)