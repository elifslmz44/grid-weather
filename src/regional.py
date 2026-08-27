"""
Per-city regional sensitivity.

National demand responds to national weather, but the 10 cities behind the population-weighted
proxy don't contribute equally. For each city this measures the correlation between that city's
daily temperature and national daily demand: a mix of how cold the region runs and how much load
sits there. GB weather moves largely in step, so the correlations are similar in size; the ranking
is the point, not the spread.

Reads the per-city file written by ingest_weather (data/raw/weather/gb_temperature_hourly.csv).
Run from the project root:
    python -m src.regional
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("regional")

RAW = config.RAW_WEATHER_DIR / "gb_temperature_hourly.csv"
DAILY = config.PROCESSED_DIR / "daily.csv"


def run():
    if not RAW.exists():
        log.warning("no per-city weather file at %s; skipping regional analysis "
                    "(run ingest_weather first).", RAW)
        return None
    w = pd.read_csv(RAW, parse_dates=["timestamp"])
    cities = [c["name"] for c in config.WEATHER_CITIES if c["name"] in w.columns]
    if not cities:
        log.warning("no city columns found in %s; skipping.", RAW.name)
        return None
    w["date"] = w["timestamp"].dt.date
    city_daily = w.groupby("date")[cities].mean()

    daily = pd.read_csv(DAILY, parse_dates=["date"])
    daily["date"] = daily["date"].dt.date
    joined = daily.set_index("date")[["nd_mean"]].join(city_daily, how="inner").dropna()

    pops = {c["name"]: float(c["population"]) for c in config.WEATHER_CITIES}
    total_pop = sum(pops.get(c, 0.0) for c in cities) or 1.0

    rows = []
    for c in cities:
        corr = float(np.corrcoef(joined[c], joined["nd_mean"])[0, 1])
        rows.append({
            "city": c,
            "corr": round(corr, 3),
            "abs_corr": round(abs(corr), 3),
            "weight_pct": round(100 * pops.get(c, 0.0) / total_pop, 1),
        })
    rows.sort(key=lambda r: r["abs_corr"], reverse=True)

    payload = {
        "cities": rows,
        "n_days": int(len(joined)),
        "note": "Correlation of each city's daily temperature with national daily demand "
                "(negative: colder days, higher demand). GB weather is highly synoptic, so the "
                "cities move together and the magnitudes are close.",
    }
    web = config.OUTPUTS_DIR / "web_data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "regional.json").write_text(json.dumps(payload))
    log.info("Regional sensitivity over %d cities, %d days. Strongest: %s (%.2f).",
             len(rows), len(joined), rows[0]["city"], rows[0]["corr"])
    return payload


if __name__ == "__main__":
    run()
