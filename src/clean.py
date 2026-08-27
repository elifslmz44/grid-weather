"""
Phase 3 -- clean, validate and align the raw electricity and weather series.

The hard part here is TIMESTAMPS. GB settlement periods are expressed in local clock time
(Europe/London), so a day is NOT always 48 half-hours:
    * spring-forward day  -> 23 hours -> 46 settlement periods
    * autumn fall-back day -> 25 hours -> 50 settlement periods

(Weather is fetched in UTC, so only the demand side needs this local->UTC anchoring.)

Building a correct instant from (SETTLEMENT_DATE, SETTLEMENT_PERIOD) is done by anchoring at
local midnight, converting that anchor to UTC, and then adding (period - 1) * 30 minutes of
*real elapsed time*. Because the periods track real elapsed time from local midnight, this
reproduces the 46/50-period days automatically without special-casing. Local midnight itself
is never ambiguous or non-existent in the UK (transitions happen at 01:00 / 02:00), so
localising it is always safe.

Everything downstream works in UTC. A local-time column is kept for human-readable plots.

Outputs (data/processed/):
    demand_halfhourly.csv   -- tidy half-hourly demand, UTC + local timestamps
    weather_hourly.csv      -- tidy hourly population-weighted temperature, UTC
    daily.csv               -- daily demand aggregates + daily temperature TARGETS
Plus a human-readable report at outputs/data_quality_report.md.

DISCIPLINE: the temperature columns in daily.csv (temp_mean_c/min/max) are TARGETS for later
modelling and must never be used as model inputs. features.py enforces that separation.

Run from the project root:
    python -m src.clean
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("clean")

# Plausibility bounds for GB National Demand (MW). Real ND sits roughly 15-55 GW; anything
# outside this is treated as impossible and reported (never silently kept or filled).
ND_MIN_PLAUSIBLE = 5_000
ND_MAX_PLAUSIBLE = 70_000

CORE_NUMERIC = [
    "nd", "tsd", "england_wales_demand",
    "embedded_wind_generation", "embedded_solar_generation",
    "embedded_wind_capacity", "embedded_solar_capacity",
    "pump_storage_pumping", "non_bm_stor",
]


# --------------------------------------------------------------------------------------
# Demand
# --------------------------------------------------------------------------------------
def _load_raw_demand() -> pd.DataFrame:
    files = sorted(config.RAW_DIR.glob("neso_demand_*.csv"))
    if not files:
        raise FileNotFoundError("No cached demand files. Run `python -m src.ingest_neso`.")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.strip().lower() for c in df.columns]
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    log.info("Loaded raw demand: %d rows from %d files.", len(combined), len(files))
    return combined


def build_demand_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add tz-aware `timestamp_utc` and `timestamp_local` from settlement date + period."""
    dates = pd.to_datetime(df["settlement_date"], errors="coerce", format="mixed")
    period = pd.to_numeric(df["settlement_period"], errors="coerce")
    local_midnight = dates.dt.tz_localize(config.TIMEZONE)   # midnight is never DST-odd
    utc_midnight = local_midnight.dt.tz_convert("UTC")
    offset = pd.to_timedelta((period - 1) * 30, unit="m")
    out = df.copy()
    out["timestamp_utc"] = utc_midnight + offset
    out["timestamp_local"] = out["timestamp_utc"].dt.tz_convert(config.TIMEZONE)
    return out


def clean_demand() -> pd.DataFrame:
    df = _load_raw_demand()
    df["settlement_period"] = pd.to_numeric(df["settlement_period"], errors="coerce")
    for col in CORE_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = build_demand_timestamps(df)

    before = len(df)
    df = df.dropna(subset=["timestamp_utc"])
    dropped_bad_ts = before - len(df)

    dup_mask = df.duplicated(subset=["settlement_date", "settlement_period"], keep="first")
    n_dupes = int(dup_mask.sum())
    df = df[~dup_mask]

    imp_mask = (df["nd"].isna()) | (df["nd"] < ND_MIN_PLAUSIBLE) | (df["nd"] > ND_MAX_PLAUSIBLE)
    n_impossible = int(imp_mask.sum())
    df = df[~imp_mask]

    if dropped_bad_ts:
        log.warning("Dropped %d rows with unparseable date/period.", dropped_bad_ts)
    if n_dupes:
        log.warning("Dropped %d duplicate (date, period) rows.", n_dupes)
    if n_impossible:
        log.warning("Dropped %d rows with implausible/missing ND.", n_impossible)

    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    df.attrs.update(dropped_bad_ts=dropped_bad_ts, n_dupes=n_dupes, n_impossible=n_impossible)

    keep = ["timestamp_utc", "timestamp_local", "settlement_date", "settlement_period"] + \
           [c for c in CORE_NUMERIC if c in df.columns]
    tidy = df[keep].copy()
    tidy.attrs.update(df.attrs)

    out = config.PROCESSED_DIR / "demand_halfhourly.csv"
    tidy.to_csv(out, index=False)
    log.info("Saved %s (%d rows, %s .. %s).", out.name, len(tidy),
             tidy["timestamp_utc"].min(), tidy["timestamp_utc"].max())
    return tidy


def detect_missing_halfhours(tidy: pd.DataFrame) -> pd.DataFrame:
    ts = pd.DatetimeIndex(pd.to_datetime(tidy["timestamp_utc"], utc=True)).sort_values()
    full = pd.date_range(ts.min(), ts.max(), freq="30min", tz="UTC")
    missing = full.difference(ts)
    if len(missing) == 0:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_periods"])
    gaps, start, prev = [], missing[0], missing[0]
    for t in missing[1:]:
        if (t - prev) > pd.Timedelta("30min"):
            gaps.append((start, prev))
            start = t
        prev = t
    gaps.append((start, prev))
    rows = [{"gap_start": s, "gap_end": e,
             "missing_periods": int((e - s) / pd.Timedelta("30min")) + 1} for s, e in gaps]
    return (pd.DataFrame(rows)
            .sort_values("missing_periods", ascending=False)
            .reset_index(drop=True))


def _expected_periods_for_date(date_val) -> int:
    # Work on NAIVE midnights so "next day" means the next calendar midnight, then localise
    # each end. On DST days the two local midnights are 23h or 25h apart in real time.
    d = pd.to_datetime(date_val).normalize()
    local0 = d.tz_localize(config.TIMEZONE)
    local1 = (d + pd.Timedelta(days=1)).tz_localize(config.TIMEZONE)
    hours = (local1.tz_convert("UTC") - local0.tz_convert("UTC")) / pd.Timedelta("1h")
    return int(round(hours * 2))


def verify_dst_period_counts(tidy: pd.DataFrame) -> pd.DataFrame:
    counts = tidy.groupby("settlement_date")["settlement_period"].count()
    expected = pd.Series({d: _expected_periods_for_date(d) for d in counts.index})
    mismatch = counts[counts != expected]
    return pd.DataFrame({"date": mismatch.index,
                         "periods_found": mismatch.values,
                         "periods_expected": expected[mismatch.index].values})


# --------------------------------------------------------------------------------------
# Weather
# --------------------------------------------------------------------------------------
def clean_weather() -> pd.DataFrame:
    src = config.RAW_WEATHER_DIR / "gb_temperature_hourly.csv"
    if not src.exists():
        raise FileNotFoundError("No weather file. Run `python -m src.ingest_weather`.")
    w = pd.read_csv(src)
    # Weather is fetched in UTC (config.WEATHER_REQUEST_TIMEZONE), so parsing is direct and
    # unambiguous -- no DST inference needed.
    w["timestamp_utc"] = pd.to_datetime(w["timestamp"], utc=True)

    tidy = (w[["timestamp_utc", "gb_temp_pop_weighted", "gb_temp_simple_mean"]]
            .dropna(subset=["timestamp_utc"])
            .sort_values("timestamp_utc")
            .drop_duplicates(subset=["timestamp_utc"], keep="first")
            .reset_index(drop=True))

    out = config.PROCESSED_DIR / "weather_hourly.csv"
    tidy.to_csv(out, index=False)
    log.info("Saved %s (%d rows, %s .. %s).", out.name, len(tidy),
             tidy["timestamp_utc"].min(), tidy["timestamp_utc"].max())
    return tidy


# --------------------------------------------------------------------------------------
# Daily aggregation
# --------------------------------------------------------------------------------------
def build_daily(demand: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    d = demand.copy()
    d["timestamp_utc"] = pd.to_datetime(d["timestamp_utc"], utc=True)
    d["date"] = d["timestamp_utc"].dt.date

    agg_spec = {
        "nd_mean": ("nd", "mean"),
        "nd_max": ("nd", "max"),
        "nd_min": ("nd", "min"),
        "nd_std": ("nd", "std"),
        "tsd_mean": ("tsd", "mean"),
        "n_periods": ("settlement_period", "count"),
    }
    if "embedded_wind_generation" in d:
        agg_spec["embedded_wind_mean"] = ("embedded_wind_generation", "mean")
    if "embedded_solar_generation" in d:
        agg_spec["embedded_solar_mean"] = ("embedded_solar_generation", "mean")

    agg = d.groupby("date").agg(**agg_spec).reset_index()
    agg["nd_range"] = agg["nd_max"] - agg["nd_min"]
    agg["nd_total_mwh"] = agg["nd_mean"] * 24

    w = weather.copy()
    w["timestamp_utc"] = pd.to_datetime(w["timestamp_utc"], utc=True)
    w["date"] = w["timestamp_utc"].dt.date
    wagg = w.groupby("date").agg(
        temp_mean_c=("gb_temp_pop_weighted", "mean"),
        temp_min_c=("gb_temp_pop_weighted", "min"),
        temp_max_c=("gb_temp_pop_weighted", "max"),
    ).reset_index()

    daily = agg.merge(wagg, on="date", how="inner").sort_values("date").reset_index(drop=True)
    out = config.PROCESSED_DIR / "daily.csv"
    daily.to_csv(out, index=False)
    log.info("Saved %s (%d days, %s .. %s).", out.name, len(daily),
             daily["date"].min(), daily["date"].max())
    return daily


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------
def write_quality_report(demand, weather, daily, gaps, dst_mismatch) -> Path:
    a = demand.attrs
    L = ["# Data-quality report\n",
         f"_Generated by `src/clean.py`. Study window {config.START_YEAR}-{config.END_YEAR}._\n",
         "## Electricity (NESO half-hourly)\n",
         f"- Rows after cleaning: **{len(demand):,}**",
         f"- Coverage: {demand['timestamp_utc'].min()} → {demand['timestamp_utc'].max()}",
         f"- Duplicate (date, period) rows dropped: {a.get('n_dupes', 0)}",
         f"- Unparseable date/period rows dropped: {a.get('dropped_bad_ts', 0)}",
         f"- Implausible/missing ND rows dropped (outside {ND_MIN_PLAUSIBLE:,}-{ND_MAX_PLAUSIBLE:,} MW): {a.get('n_impossible', 0)}",
         f"- Missing half-hour intervals grouped into **{len(gaps)}** gap(s)"]
    if len(gaps):
        L += ["\n  Largest gaps:\n", gaps.head(10).to_markdown(index=False)]
    L += ["", "## DST period-count check\n"]
    if len(dst_mismatch) == 0:
        L.append("- Every day has the expected 48 / 46 / 50 settlement periods. ✅")
    else:
        L += [f"- {len(dst_mismatch)} day(s) with an unexpected period count:\n",
              dst_mismatch.to_markdown(index=False)]
    L += ["", "## Weather (Open-Meteo ERA5, population-weighted)\n",
          f"- Hourly rows after cleaning: **{len(weather):,}**",
          f"- Coverage: {weather['timestamp_utc'].min()} → {weather['timestamp_utc'].max()}",
          f"- Missing pop-weighted values: {int(weather['gb_temp_pop_weighted'].isna().sum())}",
          "", "## Daily merged table (modelling base)\n",
          f"- Days: **{len(daily):,}** ({daily['date'].min()} → {daily['date'].max()})",
          f"- Daily mean-temperature span: {daily['temp_mean_c'].min():.1f}°C to {daily['temp_mean_c'].max():.1f}°C",
          "", "## Timezone handling\n",
          "- Demand timestamps: anchor at local midnight (Europe/London) → convert to UTC → "
          "add (period-1)×30 min. Reproduces 46/50-period DST days automatically.",
          "- Weather fetched directly in UTC. All joins performed in UTC."]
    report = config.OUTPUTS_DIR / "data_quality_report.md"
    report.write_text("\n".join(L))
    log.info("Wrote %s", report)
    return report


def write_quality_json(demand, weather, daily, gaps, dst_mismatch) -> dict:
    """Compact machine-readable version of the quality report, for the site's integrity panel."""
    a = demand.attrs
    payload = {
        "demand_rows": int(len(demand)),
        "coverage": [str(demand["timestamp_utc"].min().date()), str(demand["timestamp_utc"].max().date())],
        "dupes_dropped": int(a.get("n_dupes", 0)),
        "bad_ts_dropped": int(a.get("dropped_bad_ts", 0)),
        "impossible_dropped": int(a.get("n_impossible", 0)),
        "missing_gaps": int(len(gaps)),
        "dst_mismatches": int(len(dst_mismatch)),
        "weather_rows": int(len(weather)),
        "weather_missing": int(weather["gb_temp_pop_weighted"].isna().sum()),
        "daily_rows": int(len(daily)),
        "temp_span_c": [round(float(daily["temp_mean_c"].min()), 1), round(float(daily["temp_mean_c"].max()), 1)],
    }
    web = config.OUTPUTS_DIR / "web_data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "data_quality.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    demand = clean_demand()
    gaps = detect_missing_halfhours(demand)
    dst_mismatch = verify_dst_period_counts(demand)
    weather = clean_weather()
    daily = build_daily(demand, weather)
    write_quality_report(demand, weather, daily, gaps, dst_mismatch)
    write_quality_json(demand, weather, daily, gaps, dst_mismatch)

    print("\n=== Phase 3 summary ===")
    print(f"Demand half-hourly rows : {len(demand):,}")
    print(f"Missing-interval gaps   : {len(gaps)}")
    print(f"DST count mismatches    : {len(dst_mismatch)}")
    print(f"Weather hourly rows     : {len(weather):,}")
    print(f"Daily modelling rows    : {len(daily):,}")
    print("\nSee outputs/data_quality_report.md for the full report.")


if __name__ == "__main__":
    main()
