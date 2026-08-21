"""
Ingest historic GB electricity demand from the NESO Data Portal (CKAN).

Design principles:
    * Discover the per-year CSV resources dynamically (package_show) so the pipeline keeps
      working when NESO publishes a new year. Fall back to the hard-coded IDs in config only
      if discovery fails.
    * Cache every raw download to data/raw/ and never re-download a file that already exists
      (unless force=True). This respects the API and makes the pipeline reproducible offline.
    * Do the minimum in this layer: fetch + cache + concatenate. All parsing, timezone work,
      de-duplication and validation happens in clean.py so responsibilities stay separated.

Run from the project root:
    python -m src.ingest_neso                 # download the configured window
    python -m src.ingest_neso --start 2018 --end 2025
    python -m src.ingest_neso --list          # just list available year->resource mappings
"""

from __future__ import annotations

import argparse
import io
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
log = logging.getLogger("ingest_neso")


# --------------------------------------------------------------------------------------
# Low-level HTTP with retry/backoff
# --------------------------------------------------------------------------------------
def _get(url: str, params: dict | None = None) -> requests.Response:
    """GET with retries and exponential backoff. Raises on final failure."""
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
    raise RuntimeError(f"GET failed after {config.REQUEST_RETRIES} attempts: {url}") from last_exc


# --------------------------------------------------------------------------------------
# Resource discovery
# --------------------------------------------------------------------------------------
def list_demand_resources() -> dict[int, dict]:
    """
    Return {year: {"resource_id":..., "url":..., "name":...}} for the Historic Demand Data
    package. Falls back to config.NESO_DEMAND_RESOURCE_IDS_FALLBACK if the API call fails.
    """
    url = f"{config.NESO_API_BASE}/package_show"
    try:
        resp = _get(url, params={"id": config.NESO_DEMAND_PACKAGE_ID})
        resources = resp.json()["result"]["resources"]
    except Exception as exc:  # noqa: BLE001 - we intentionally degrade gracefully
        log.error("package_show failed (%s); using hard-coded fallback resource IDs.", exc)
        return {
            yr: {"resource_id": rid, "url": None, "name": f"fallback_{yr}"}
            for yr, rid in config.NESO_DEMAND_RESOURCE_IDS_FALLBACK.items()
        }

    mapping: dict[int, dict] = {}
    for r in resources:
        # A year appears in the resource name (e.g. "Historic Demand Data 2022") and/or the
        # download filename (e.g. demanddata_2022.csv). Extract the first 20xx we find.
        blob = f"{r.get('name', '')} {r.get('url', '')}"
        year = _extract_year(blob)
        if year is None:
            continue
        # Prefer .csv resources; skip anything that isn't tabular.
        fmt = (r.get("format") or "").lower()
        if fmt and fmt != "csv":
            continue
        mapping[year] = {
            "resource_id": r.get("id"),
            "url": r.get("url"),
            "name": r.get("name"),
        }
    if not mapping:
        log.error("No year resources parsed from package_show; using fallback.")
        return {
            yr: {"resource_id": rid, "url": None, "name": f"fallback_{yr}"}
            for yr, rid in config.NESO_DEMAND_RESOURCE_IDS_FALLBACK.items()
        }
    log.info("Discovered %d year resources (%d..%d).",
             len(mapping), min(mapping), max(mapping))
    return dict(sorted(mapping.items()))


def _extract_year(text: str) -> int | None:
    import re
    m = re.search(r"(20\d{2})", text)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------------------
# Download + cache
# --------------------------------------------------------------------------------------
def _download_url(resource_id: str, direct_url: str | None) -> str:
    """
    Build a CSV download URL. Prefer the direct resource URL from package_show; otherwise
    fall back to the CKAN datastore_search dump endpoint which returns the full resource CSV.
    """
    if direct_url:
        return direct_url
    return f"{config.NESO_API_BASE}/datastore_search?resource_id={resource_id}&limit=100000"


def download_year(year: int, resources: dict[int, dict], force: bool = False) -> Path | None:
    """Download one year's demand CSV to data/raw/, using the cache when present."""
    if year not in resources:
        log.warning("Year %d not available in NESO resources -- skipping.", year)
        return None

    dest = config.RAW_DIR / f"neso_demand_{year}.csv"
    if dest.exists() and not force:
        log.info("Cache hit: %s", dest.name)
        return dest

    meta = resources[year]
    url = _download_url(meta["resource_id"], meta.get("url"))
    log.info("Downloading %d demand data ...", year)
    resp = _get(url)

    # Two possible shapes: a raw CSV body, or a CKAN JSON envelope (datastore fallback).
    text = resp.text.lstrip()
    if text.startswith("{"):
        records = resp.json()["result"]["records"]
        df = pd.DataFrame(records)
        df.to_csv(dest, index=False)
    else:
        dest.write_bytes(resp.content)

    size_kb = dest.stat().st_size / 1024
    log.info("Saved %s (%.0f KB)", dest.name, size_kb)
    return dest


def download_window(start: int, end: int, force: bool = False) -> list[Path]:
    """Download all years in [start, end], returning the cached file paths."""
    resources = list_demand_resources()
    paths: list[Path] = []
    for year in range(start, end + 1):
        p = download_year(year, resources, force=force)
        if p is not None:
            paths.append(p)
        time.sleep(0.5)  # be polite between requests
    return paths


def load_raw_years(start: int = config.START_YEAR, end: int = config.END_YEAR) -> pd.DataFrame:
    """
    Ensure the window is downloaded, then load and concatenate the raw per-year CSVs into a
    single DataFrame. Columns are returned exactly as NESO provides them -- no renaming here.
    """
    download_window(start, end)
    frames = []
    for year in range(start, end + 1):
        f = config.RAW_DIR / f"neso_demand_{year}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df["_source_year_file"] = year
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No NESO demand files found. Run download_window first.")
    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded raw demand: %d rows, %d columns.", len(combined), combined.shape[1])
    return combined


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ingest NESO historic demand data.")
    parser.add_argument("--start", type=int, default=config.START_YEAR)
    parser.add_argument("--end", type=int, default=config.END_YEAR)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--list", action="store_true", help="list resources and exit")
    args = parser.parse_args()

    if args.list:
        for yr, meta in list_demand_resources().items():
            print(f"{yr}: {meta['resource_id']}  {meta['name']}")
        return

    paths = download_window(args.start, args.end, force=args.force)
    log.info("Done. %d files cached in %s", len(paths), config.RAW_DIR)

    # Print the real schema of the most recent file so you can confirm column names.
    if paths:
        sample = pd.read_csv(paths[-1], nrows=5)
        print("\n--- Columns in", paths[-1].name, "---")
        for c in sample.columns:
            print("  ", c)
        print("\n--- First rows ---")
        print(sample.to_string(index=False))


if __name__ == "__main__":
    _cli()
