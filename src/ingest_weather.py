"""
Ingest a population-weighted Great Britain temperature series from Open-Meteo (ERA5).

Why this design:
    * We want a *national* temperature that reflects where electricity load actually is, so
      we sample several GB population centres and weight them by population (config.WEATHER_CITIES).
    * Open-Meteo's archive endpoint is ERA5 reanalysis: free, no key, hourly, spatially
      complete. It is model-blended, not raw station readings -- documented in the README as a
      deliberate, swappable substitution for Met Office station observations.
    * The weather layer is intentionally isolated: to switch providers later, only this file
      needs to change; downstream code just consumes data/raw/weather/gb_temperature_hourly.csv.

Timezone: we request timezone="UTC" (config.WEATHER_REQUEST_TIMEZONE). The demand series is
also built in UTC, so the two align directly with no DST inference required.

Run from the project root:
    python -m src.ingest_weather
    python -m src.ingest_weather --start-year 2018 --end-year 2025
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_weather")


def _get(url: str, params: dict) -> requests.Response:
    headers = {"User-Agent": config.USER_AGENT}
    last_exc: Exception | None = None
    for attempt in range(1, config.REQUEST_RETRIES + 1):
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            wait = config.REQUEST_BACKOFF ** attempt
            log.warning("GET failed (attempt %d/%d): %s -- retrying in %.1fs",
                        attempt, config.REQUEST_RETRIES, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"GET failed after {config.REQUEST_RETRIES} attempts") from last_exc


def fetch_city_hourly(city: dict, start: str, end: str, force: bool = False) -> pd.DataFrame:
    """
    Fetch hourly 2 m temperature for one city, caching the raw JSON response.
    start/end are ISO dates 'YYYY-MM-DD'.
    """
    cache = config.RAW_WEATHER_DIR / f"{city['name'].lower()}_{start}_{end}.json"
    if cache.exists() and not force:
        log.info("Cache hit: %s", cache.name)
        payload = json.loads(cache.read_text())
    else:
        params = {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "start_date": start,
            "end_date": end,
            "hourly": "temperature_2m",
            "timezone": config.WEATHER_REQUEST_TIMEZONE,  # UTC -> no DST ambiguity downstream
        }
        log.info("Fetching %s (%s..%s) ...", city["name"], start, end)
        resp = _get(config.OPENMETEO_ARCHIVE_BASE, params)
        payload = resp.json()
        cache.write_text(json.dumps(payload))
        time.sleep(1.0)  # be polite; well within the free-tier daily budget

    hourly = payload["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        city["name"]: hourly["temperature_2m"],
    })
    return df


def build_population_weighted_series(
    start_year: int = config.START_YEAR,
    end_year: int = config.END_YEAR,
    force: bool = False,
) -> pd.DataFrame:
    """
    Fetch every configured city and combine into an hourly national temperature estimate.

    Returns a DataFrame with columns:
        timestamp, <each city>, gb_temp_pop_weighted, gb_temp_simple_mean
    and also writes it to data/raw/weather/gb_temperature_hourly.csv.
    """
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"

    merged: pd.DataFrame | None = None
    for city in config.WEATHER_CITIES:
        df = fetch_city_hourly(city, start, end, force=force)
        merged = df if merged is None else merged.merge(df, on="timestamp", how="outer")

    assert merged is not None
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    city_names = [c["name"] for c in config.WEATHER_CITIES]
    weights = pd.Series({c["name"]: c["population"] for c in config.WEATHER_CITIES},
                        dtype="float64")
    weights = weights / weights.sum()  # normalise

    # Population-weighted mean (ignores a city if it is NaN for a given hour, re-normalising
    # over the available cities so a single gap does not blank the national value).
    temps = merged[city_names]
    w = weights[city_names]
    weighted_sum = temps.mul(w, axis=1).sum(axis=1, skipna=True)
    weight_present = temps.notna().mul(w, axis=1).sum(axis=1)
    merged["gb_temp_pop_weighted"] = weighted_sum / weight_present
    merged["gb_temp_simple_mean"] = temps.mean(axis=1, skipna=True)

    out = config.RAW_WEATHER_DIR / "gb_temperature_hourly.csv"
    merged.to_csv(out, index=False)
    log.info("Saved %s (%d hourly rows, %d cities).", out.name, len(merged), len(city_names))
    log.info("Population weights used: %s",
             {k: round(float(v), 3) for k, v in w.items()})
    return merged


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest population-weighted GB temperature.")
    parser.add_argument("--start-year", type=int, default=config.START_YEAR)
    parser.add_argument("--end-year", type=int, default=config.END_YEAR)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    df = build_population_weighted_series(args.start_year, args.end_year, force=args.force)
    print("\n--- Sample ---")
    print(df[["timestamp", "gb_temp_pop_weighted", "gb_temp_simple_mean"]].head().to_string(index=False))
    print("\nDate range:", df["timestamp"].min(), "->", df["timestamp"].max())
    print("Missing pop-weighted values:", int(df["gb_temp_pop_weighted"].isna().sum()))


if __name__ == "__main__":
    _cli()
