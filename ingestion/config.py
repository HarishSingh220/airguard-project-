"""
config.py
Environment variables and fixed locations shared by every ingestion script.

Required environment variables (set as GitHub Actions secrets; see
.github/workflows/ingest.yml):
    DATABASE_URL        postgres connection string, e.g.
                        postgresql://user:pass@host:5432/airguard
    FIRMS_MAP_KEY       from https://firms.modaps.eosdis.nasa.gov/api/map_key/
    CDSAPI_URL          ADS API root, e.g. https://ads.atmosphere.copernicus.eu/api
    CDSAPI_KEY          your ADS personal access token
                        (used by fetch_cams.py and cds_backfill.py's "cams-no2-o3" job.
                        cds_backfill.py's "era5-pressure-qw" job needs the SEPARATE
                        climate CDS credential instead -- register at
                        cds.climate.copernicus.eu and point a second ~/.cdsapirc, or
                        pass --url/--key, at https://cds.climate.copernicus.eu/api.
                        ADS and the climate CDS are different services with different
                        dataset catalogues even though both use the cdsapi library.)
    CPCB_API_KEY        data.gov.in API key (Dashboard -> My Account -> API key);
                        the public sample key works for light testing but is rate
                        limited and shared by everyone, so request your own.

Optional environment variables:
    CPCB_PROXY_URL      base URL of a relay that forwards to api.data.gov.in from an
                        IP that isn't a GitHub Actions runner (see
                        ingestion/cpcb_proxy_worker.js for a ready-to-deploy
                        Cloudflare Worker). If unset, fetch_cpcb.py calls
                        api.data.gov.in directly as before. Set it to your deployed
                        Worker's URL, e.g. https://cpcb-proxy.<you>.workers.dev
                        (no trailing slash) -- fetch_cpcb.py appends the same
                        /resource/<id> path and query string it always has.
"""
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "")
FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
CDSAPI_URL = os.environ.get("CDSAPI_URL", "")
CDSAPI_KEY = os.environ.get("CDSAPI_KEY", "")
# Optional: see "Optional environment variables" above. Empty string means "no
# proxy configured", which fetch_cpcb.py treats the same as not set.
CPCB_PROXY_URL = os.environ.get("CPCB_PROXY_URL", "").rstrip("/")
# No fallback default: a shared sample key is rate-limited and shared across every
# user of this script, so silently falling back to it just makes failures show up
# later and elsewhere (as rate-limit errors or skipped stations) instead of here,
# clearly, at startup. fetch_cpcb.py checks this is non-empty before making any
# request. Set your own key: data.gov.in -> My Account -> API key.
CPCB_API_KEY = os.environ.get("CPCB_API_KEY", "")

# The five stations build_features.py already knows, with the station name used
# by the CPCB dashboard/API (edit STATION_CPCB_NAME if your station list differs
# from what final_delhi_dataset.csv already uses).
STATIONS = {
    "Anand Vihar": {"lat": 28.6469, "lon": 77.3158, "cpcb_name": "Anand Vihar, Delhi - DPCC"},
    "Dwarka Sector 8": {"lat": 28.5710, "lon": 77.0710, "cpcb_name": "Dwarka-Sector 8, Delhi - DPCC"},
    "IMD Lodhi Road": {"lat": 28.5918, "lon": 77.2273, "cpcb_name": "IMD Lodhi Road, Delhi - IITM"},
    "ITO": {"lat": 28.6316, "lon": 77.2410, "cpcb_name": "ITO, Delhi - CPCB"},
    "Punjabi Bagh": {"lat": 28.6740, "lon": 77.1310, "cpcb_name": "Punjabi Bagh, Delhi - DPCC"},
}
# Station names on the CPCB dashboard change occasionally (an "-DPCC"/"-CPCB"
# suffix, a monitor being renamed). If fetch_cpcb.py reports 0 rows for a
# station, check the current name in the API response and update cpcb_name.

# Delhi centroid, used for the single-point weather and CAMS pulls.
DELHI_LAT, DELHI_LON = 28.6139, 77.2090

# Punjab-Haryana-Delhi NCR bounding box for FIRMS fire detections
# (west, south, east, north), matching the belt build_features.py expects
# fire_daily_features.csv to already summarise.
FIRE_BBOX = (73.5, 27.5, 78.5, 32.5)

PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 300]
# era5_inputs.levels6_daily() also uses spd300 (wind speed at 300 hPa), hence 300 here.