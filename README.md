# Giri-Rakshak — Landslide Early Warning System (MVP)

## Run it

```bash
pip install -r requirements.txt
streamlit run Giri-Rakshak_app_FINAL_v9.py
```

**`Giri-Rakshak_app_FINAL_v9.py` is the one live app.** Earlier iterations
(`app.py`, `app_backup.py`, `app_location.py`, `sikkim.py`,
`Giri-Rakshak_app_FINAL.py` through `_v8.py`) are archived in `_archive/`
for reference only — do not run them, they are not maintained.

Optional, for the satellite disturbance check to authenticate:
```bash
earthengine authenticate
```
(Runs once per machine. Without it, the app still runs fully — the
satellite panel just reports it isn't authenticated instead of crashing.)

## What's real in this build

| Component | Status |
|---|---|
| Static susceptibility model | **Real.** 8-factor Random Forest (700 trees), trained on 991 NASA COOLR + literature landslide points across NE India, 250m grid. ROC-AUC 0.72 / PR-AUC 0.65 — see `processed/step72_susceptibility_validation_report.txt`. |
| Rainfall trigger engine | **Real.** Rolling 24h/72h rainfall windows + Antecedent Wetness Index, live from Open-Meteo, per-district calibrated thresholds (`dynamic/config.py`). See `dynamic/rainfall_trigger.py`. |
| Risk fusion | **Real.** Weighted combination of susceptibility + trigger, with satellite-disturbance escalation override. See `risk_fusion.py` and `dynamic/pipeline.py`. |
| Confidence engine | **Real, partial.** Scores spatial coverage, predictor completeness, live-data health, and satellite corroboration (when checked). Model uncertainty is explicitly NOT computed — see `confidence_engine.py` docstring for why. |
| Decision / alert engine | **Real.** SQLite-backed (`alerts.db`) dedup, escalation, acknowledgement, and full audit trail. See `alert_engine.py`, visible live in the sidebar. |
| Satellite change detection | **Real, requires auth.** Google Earth Engine Sentinel-1/2 comparison. Needs `earthengine authenticate` locally, or a service account in `.streamlit/secrets.toml` for a portable deploy. |
| GPS routing | **Partial.** Code is real (`dynamic pipeline` + `step6/7` scripts), but `pilot_route_data/` (road network + pre-rendered maps) is not included in this repo — the routing panel will show an honest "not available" message until that data is added back. |
| Database (PostGIS), API layer (FastAPI), SMS/email alerts, RBAC | **Not built.** `psycopg2`/`fastapi`/`uvicorn` are listed in `requirements.txt` for future use but are not currently wired to anything — all data is read from local files, and Streamlit runs the whole app in-process. Architecturally scoped, not implemented in this MVP. |

## Data sources

See `data_sources.md` for the full log (NASA COOLR landslide inventory,
GPM-IMERG rainfall, SMAP soil moisture, Sentinel-2 NDVI, OpenStreetMap
roads) and `processed/step69_production_8factor_rf_config.json` for the
exact feature set the production model was trained on.

## Repository layout

```
Giri-Rakshak_app_FINAL_v9.py   Live Streamlit app (entry point)
risk_fusion.py                 Susceptibility + trigger -> risk score
confidence_engine.py           Trust score for the risk result
alert_engine.py                Persisted alert dedup/escalation/audit trail
dynamic/                       Rainfall trigger, sensor fusion, satellite pipeline
notebooks/                     One-off data acquisition & model training scripts
processed/                     Trained model, susceptibility raster, validation reports
_archive/                      Superseded app versions — not maintained, kept for reference
```
