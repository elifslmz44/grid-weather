"""
Optional validation of the ERA5 weather proxy against Met Office station observations.

The project uses ERA5 reanalysis (via Open-Meteo) because Met Office station data needs
credentials. This module is the ready-to-run comparison for when those are available: given a
Met Office DataHub API key it pulls daily observations for a set of stations, aligns them to the
ERA5-derived daily national temperature, and reports MAE, bias and correlation.

It is NOT part of the automated pipeline (the key isn't in CI). Provide the key and run manually:
    export METOFFICE_API_KEY=...        # DataHub key
    python -m src.validate_era5

Without a key it exits cleanly and does nothing.
"""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd
import requests

from . import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("validate_era5")

# A small set of long-running stations could be listed here (site id -> label). Left empty by
# default so the module is inert until someone with DataHub access fills it in.
STATIONS: dict[str, str] = {}


def _fetch_station_daily(site_id: str, key: str) -> pd.DataFrame:
    """Placeholder for a DataHub daily-observations pull. Returns date + mean temperature."""
    url = f"https://data.hub.api.metoffice.gov.uk/observations/v1/sites/{site_id}/daily"
    r = requests.get(url, headers={"apikey": key}, timeout=30)
    r.raise_for_status()
    obs = r.json().get("features", [])
    recs = [{"date": f["properties"]["time"][:10],
             "temp_c": f["properties"].get("screenTemperature")} for f in obs]
    return pd.DataFrame(recs).dropna()


def run():
    key = os.environ.get("METOFFICE_API_KEY")
    if not key or not STATIONS:
        log.info("Met Office validation skipped: %s. This module is optional and key-gated.",
                 "no METOFFICE_API_KEY set" if not key else "no STATIONS configured")
        return None

    proxy = pd.read_csv(config.PROCESSED_DIR / "daily.csv", parse_dates=["date"])[["date", "temp_mean_c"]]
    proxy["date"] = proxy["date"].dt.date

    frames = []
    for site_id, label in STATIONS.items():
        try:
            s = _fetch_station_daily(site_id, key)
            s["date"] = pd.to_datetime(s["date"]).dt.date
            frames.append(s.rename(columns={"temp_c": label})[["date", label]])
        except Exception as exc:
            log.warning("station %s (%s) failed: %s", site_id, label, exc)
    if not frames:
        log.warning("no station data retrieved; nothing to validate.")
        return None

    stations = frames[0]
    for f in frames[1:]:
        stations = stations.merge(f, on="date", how="outer")
    station_mean = stations.set_index("date").mean(axis=1).rename("station_c")

    joined = proxy.set_index("date").join(station_mean, how="inner").dropna()
    err = joined["temp_mean_c"] - joined["station_c"]
    payload = {
        "n_days": int(len(joined)),
        "mae_c": round(float(err.abs().mean()), 2),
        "bias_c": round(float(err.mean()), 2),
        "corr": round(float(np.corrcoef(joined["temp_mean_c"], joined["station_c"])[0, 1]), 3),
        "stations": list(STATIONS.values()),
    }
    web = config.OUTPUTS_DIR / "web_data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "era5_validation.json").write_text(json.dumps(payload))
    log.info("ERA5 vs Met Office over %d days: MAE %.2f C, bias %.2f C, r=%.3f.",
             payload["n_days"], payload["mae_c"], payload["bias_c"], payload["corr"])
    return payload


if __name__ == "__main__":
    run()
