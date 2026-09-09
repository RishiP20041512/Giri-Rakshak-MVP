@'
# Data Source Log

| File | Source | URL | Date Downloaded | Notes |
|------|--------|-----|------------------|-------|
'@ | Set-Content -Path "data_sources.md"| landslides.csv | NASA COOLR + Mihu et al. 2026 paper | https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/COOLR_Events_Points/FeatureServer | 2026-08-30 | 5 paper points + COOLR API results |
| rainfall/*.nc4 | NASA GPM (GPM_3IMERGDF) via earthaccess | https://gpm.nasa.gov/data/directory | 2026-08-30 | Monsoon season 2023, Dibang Valley bbox |
| soil_moisture/*.h5 | NASA SMAP (SPL3SMP_E) via earthaccess | https://nsidc.org/data/smap | 2026-08-30 | June 2023 test range |
| satellite/ndvi tif | Sentinel-2 via Copernicus Data Space | https://dataspace.copernicus.eu | 2026-08-30 | June 2023, least cloud cover |
| roads_villages/osm_data.json | OpenStreetMap Overpass API | https://overpass-api.de | 2026-08-30 | Dibang Valley bbox |
