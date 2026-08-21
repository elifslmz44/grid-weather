"""
Central configuration for the "Can Britain's Electricity Grid Reveal the Weather?" project.

Everything that a future maintainer might want to change -- date ranges, the set of
cities used to build the weather proxy, API endpoints -- lives here so the rest of the
codebase never hard-codes these values inline.

Timezone note (important, read before touching any timestamp logic):
    * NESO settlement data is expressed in *local clock time* (Europe/London). Each day
      normally has 48 half-hourly settlement periods, but 46 on the spring-forward DST day
      and 50 on the autumn fall-back day. We preserve this and handle it explicitly in
      clean.py -- we do NOT assume every day has 48 periods.
    * Open-Meteo is queried with timezone="Europe/London" so both series share the same
      wall-clock convention before alignment. This is documented again in ingest_weather.py.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_WEATHER_DIR = RAW_DIR / "weather"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODEL_RESULTS_DIR = OUTPUTS_DIR / "model_results"

for _d in (RAW_DIR, RAW_WEATHER_DIR, PROCESSED_DIR, FIGURES_DIR, MODEL_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------
# Analysis window (inclusive). Change these two lines to widen/narrow the study period.
# 2015-2025 gives 11 full years with embedded wind & solar reliably reported.
# --------------------------------------------------------------------------------------
START_YEAR = 2015
END_YEAR = 2025

TIMEZONE = "Europe/London"

# --------------------------------------------------------------------------------------
# NESO (National Energy System Operator) CKAN Data Portal
# --------------------------------------------------------------------------------------
NESO_API_BASE = "https://api.neso.energy/api/3/action"
# The "Historic Demand Data" dataset (one CSV resource per year).
NESO_DEMAND_PACKAGE_ID = "historic-demand-data"

# Confirmed resource IDs (verified live during Phase 1, 2026-08). These are used ONLY as a
# fallback if dynamic discovery via package_show fails -- the pipeline prefers discovery so
# it keeps working when NESO adds a new year. Verify columns after first download; do not
# assume column names before inspecting the real CSV.
NESO_DEMAND_RESOURCE_IDS_FALLBACK = {
    2006: "949bb4a4-8374-4730-89ef-302d82428d2c",
    2019: "dd9de980-d724-415a-b344-d8ae11321432",
    2022: "bb44a1b5-75b1-4db2-8491-257f23385006",
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2026: "8a4a771c-3929-4e56-93ad-cdf13219dea5",
}
# Rolling "Demand Data Update" resource (previous month -> today). Handy for topping up the
# current partial year, which the per-year files publish ~21 days in arrears.
NESO_DEMAND_UPDATE_RESOURCE_ID = "177f6fa4-ae49-4182-81ea-0c6b35f26ca6"

# --------------------------------------------------------------------------------------
# Weather proxy: population-weighted GB temperature
# --------------------------------------------------------------------------------------
# Open-Meteo Historical Weather API (ERA5 reanalysis). Free, no key, hourly, from 1940.
OPENMETEO_ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"

# Fetch weather in UTC so there is zero DST ambiguity to resolve later. The demand series is
# also built in UTC, so both align directly with no timezone inference heuristics. (TIMEZONE
# above stays Europe/London -- it is only used to anchor settlement periods at local midnight.)
WEATHER_REQUEST_TIMEZONE = "UTC"

# Great Britain population centres (England + Scotland + Wales; NESO is a GB operator, so
# Northern Ireland is deliberately excluded). Weights are approximate built-up-area / urban
# populations used purely to weight the national temperature estimate toward where people
# (and therefore electricity load) actually are. They are normalised at runtime, so exact
# values matter less than their relative sizes. Documented in the README.
WEATHER_CITIES = [
    {"name": "London",     "lat": 51.5074, "lon": -0.1278, "population": 9_000_000},
    {"name": "Birmingham", "lat": 52.4862, "lon": -1.8904, "population": 2_600_000},
    {"name": "Manchester", "lat": 53.4808, "lon": -2.2426, "population": 2_700_000},
    {"name": "Leeds",      "lat": 53.8008, "lon": -1.5491, "population": 1_900_000},
    {"name": "Glasgow",    "lat": 55.8642, "lon": -4.2518, "population": 1_200_000},
    {"name": "Sheffield",  "lat": 53.3811, "lon": -1.4701, "population":   730_000},
    {"name": "Bristol",    "lat": 51.4545, "lon": -2.5879, "population":   700_000},
    {"name": "Newcastle",  "lat": 54.9783, "lon": -1.6178, "population":   810_000},
    {"name": "Liverpool",  "lat": 53.4084, "lon": -2.9916, "population":   900_000},
    {"name": "Edinburgh",  "lat": 55.9533, "lon": -3.1883, "population":   540_000},
]

# --------------------------------------------------------------------------------------
# Networking / politeness
# --------------------------------------------------------------------------------------
REQUEST_TIMEOUT = 60      # seconds
REQUEST_RETRIES = 4
REQUEST_BACKOFF = 2.0     # seconds, exponential
USER_AGENT = "grid-weather-portfolio/0.1 (+https://github.com/)"
