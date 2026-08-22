"""
Phase 5a -- weather-blind feature engineering.

The central discipline of this whole project lives in this file: TEMPERATURE IS NEVER A
FEATURE. The model's job is to infer temperature, so temperature (and any near-proxy for it)
must stay out of the input matrix X. Concretely we exclude:
    * temp_mean_c / temp_min_c / temp_max_c   -- these are the TARGETS
    * embedded_solar_generation                -- this is literally a sunshine measurement
    * embedded_wind_generation                 -- this is literally a wind measurement
Including the last two would smuggle weather in through the back door, so they are dropped.

Two feature sets are produced so we can answer the key question -- how much weather signal is
in electricity *behaviour* versus simply knowing the time of year?
    FEATURES_A : electricity-demand-derived features only (no calendar at all)
    FEATURES_B : FEATURES_A + calendar/astronomical context (month, day-of-year, weekday,
                 holiday, daylight hours). Daylight is astronomy, not measured weather.

All engineered features are causal: same-day demand is a legitimate input (we infer today's
temperature from today's grid), and lags/rollings only look backwards, so there is no leakage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import holidays

from . import config

log = logging.getLogger("features")

TARGET = "temp_mean_c"
FORBIDDEN_IN_X = ["temp_mean_c", "temp_min_c", "temp_max_c",
                  "embedded_solar_mean", "embedded_wind_mean",
                  "embedded_wind_generation", "embedded_solar_generation"]

# Representative GB latitude for the daylight calculation (roughly the population centroid).
GB_LATITUDE_DEG = 53.5


# --------------------------------------------------------------------------------------
# Astronomical daylight (NOT weather -- pure calendar/geometry)
# --------------------------------------------------------------------------------------
def daylight_hours(day_of_year: np.ndarray, lat_deg: float = GB_LATITUDE_DEG) -> np.ndarray:
    """
    Hours of daylight for a given day-of-year and latitude (CBM model, Forsythe et al. 1995).
    Deterministic astronomy -- depends only on the calendar and geography, never on weather.
    """
    lat = np.radians(lat_deg)
    doy = np.asarray(day_of_year, dtype=float)
    # solar declination angle
    theta = 0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.00860 * (doy - 186)))
    phi = np.arcsin(0.39795 * np.cos(theta))
    # daylight coefficient (0.8333deg accounts for sun's disc + refraction)
    arg = (np.sin(np.radians(0.8333)) + np.sin(lat) * np.sin(phi)) / (np.cos(lat) * np.cos(phi))
    arg = np.clip(arg, -1, 1)
    return 24 - (24 / np.pi) * np.arccos(arg)


# --------------------------------------------------------------------------------------
# Intraday shape features (from the half-hourly series)
# --------------------------------------------------------------------------------------
def _intraday_features(demand: pd.DataFrame) -> pd.DataFrame:
    """Per-day shape features that daily aggregates alone can't capture."""
    d = demand.copy()
    loc = pd.to_datetime(d["timestamp_local"], utc=True).dt.tz_convert(config.TIMEZONE)
    d["date"] = loc.dt.date
    d["hour"] = loc.dt.hour + loc.dt.minute / 60.0
    d = d.sort_values("timestamp_utc")
    d["dnd"] = d.groupby("date")["nd"].diff()

    def per_day(g: pd.DataFrame) -> pd.Series:
        evening = g.loc[(g.hour >= 16) & (g.hour < 20), "nd"]
        overnight = g.loc[(g.hour >= 0) & (g.hour < 5), "nd"]
        morning = g.loc[(g.hour >= 5) & (g.hour < 10), "dnd"]
        return pd.Series({
            "evening_peak_mw": evening.mean() if len(evening) else np.nan,
            "overnight_min_mw": overnight.min() if len(overnight) else np.nan,
            "morning_ramp_max_mw": morning.max() if len(morning) else np.nan,
        })

    out = d.groupby("date").apply(per_day, include_groups=False).reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out


# --------------------------------------------------------------------------------------
# Build the modelling table
# --------------------------------------------------------------------------------------
def build_feature_table() -> tuple[pd.DataFrame, list[str], list[str], str]:
    """
    Returns (df, FEATURES_A, FEATURES_B, TARGET).
    df is one row per day with all features + the temperature target, ready for modelling.
    """
    daily = pd.read_csv(config.PROCESSED_DIR / "daily.csv", parse_dates=["date"])
    demand = pd.read_csv(config.PROCESSED_DIR / "demand_halfhourly.csv")

    intraday = _intraday_features(demand)
    df = daily.merge(intraday, on="date", how="left").sort_values("date").reset_index(drop=True)

    # --- causal lags & rolling stats on demand (look backwards only) ---
    df["nd_mean_lag1"] = df["nd_mean"].shift(1)
    df["nd_mean_lag7"] = df["nd_mean"].shift(7)
    df["nd_mean_roll7"] = df["nd_mean"].shift(1).rolling(7, min_periods=4).mean()
    df["nd_range_roll7"] = df["nd_range"].shift(1).rolling(7, min_periods=4).mean()

    # --- calendar / astronomical features (Model B only) ---
    doy = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["daylight_hours"] = daylight_hours(doy.to_numpy())

    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    uk_hols = holidays.UnitedKingdom(years=list(years))
    df["is_holiday"] = df["date"].dt.date.map(lambda d: int(d in uk_hols))

    # drop early rows that lack lag features
    df = df.dropna(subset=["nd_mean_lag7", "nd_mean_roll7", "evening_peak_mw"]).reset_index(drop=True)

    FEATURES_A = [
        "nd_mean", "nd_max", "nd_min", "nd_std", "nd_range", "tsd_mean",
        "evening_peak_mw", "overnight_min_mw", "morning_ramp_max_mw",
        "nd_mean_lag1", "nd_mean_lag7", "nd_mean_roll7", "nd_range_roll7",
    ]
    FEATURES_B = FEATURES_A + [
        "month_sin", "month_cos", "doy_sin", "doy_cos",
        "dow", "is_weekend", "is_holiday", "daylight_hours",
    ]

    # hard guard: no temperature or weather-generation leakage
    leaks = [c for c in (FEATURES_A + FEATURES_B) if c in FORBIDDEN_IN_X]
    if leaks:
        raise RuntimeError(f"Weather leakage in features: {leaks}")

    log.info("Feature table: %d days, %d A-features, %d B-features.",
             len(df), len(FEATURES_A), len(FEATURES_B))
    return df, FEATURES_A, FEATURES_B, TARGET
