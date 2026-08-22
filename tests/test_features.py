"""
Tests for src/features.py -- the two guarantees that matter most for this project:
  1. No weather information (temperature, or solar/wind generation) leaks into the features.
  2. The astronomical daylight calculation is physically sensible for GB latitudes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features, config


def test_daylight_hours_seasonal_range():
    # Summer solstice (~doy 172) long days; winter solstice (~doy 355) short days.
    summer = features.daylight_hours(np.array([172]))[0]
    winter = features.daylight_hours(np.array([355]))[0]
    assert 15.5 < summer < 17.5      # ~16.6h at 53.5N
    assert 6.5 < winter < 8.5        # ~7.5h at 53.5N
    assert summer > winter


def _tiny_processed(tmp_path):
    # 40 days of half-hourly data so lags/rollings are defined.
    rng = pd.date_range("2020-01-01", "2020-02-09 23:30", freq="30min", tz="UTC")
    loc = rng.tz_convert(config.TIMEZONE)
    hour = loc.hour + loc.minute / 60
    nd = 30000 + 6000 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 300, len(rng))
    demand = pd.DataFrame({
        "timestamp_utc": rng, "timestamp_local": loc,
        "settlement_date": loc.date, "settlement_period": loc.hour * 2 + loc.minute // 30 + 1,
        "nd": nd.round(), "tsd": (nd + 4000).round(),
        "embedded_wind_generation": 1000.0, "embedded_solar_generation": 0.0,
    })
    d = demand.copy(); d["date"] = d["timestamp_utc"].dt.date
    daily = d.groupby("date").agg(
        nd_mean=("nd", "mean"), nd_max=("nd", "max"), nd_min=("nd", "min"),
        nd_std=("nd", "std"), tsd_mean=("tsd", "mean"),
        embedded_wind_mean=("embedded_wind_generation", "mean"),
        embedded_solar_mean=("embedded_solar_generation", "mean"),
        n_periods=("settlement_period", "count")).reset_index()
    daily["nd_range"] = daily.nd_max - daily.nd_min
    daily["nd_total_mwh"] = daily.nd_mean * 24
    daily["temp_mean_c"] = 5.0; daily["temp_min_c"] = 2.0; daily["temp_max_c"] = 8.0

    proc = tmp_path / "processed"; proc.mkdir()
    demand.to_csv(proc / "demand_halfhourly.csv", index=False)
    daily.to_csv(proc / "daily.csv", index=False)
    return proc


def test_no_weather_leakage_in_features(monkeypatch, tmp_path):
    proc = _tiny_processed(tmp_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", proc)

    df, feats_a, feats_b, target = features.build_feature_table()

    assert target == "temp_mean_c"
    # the target and every weather proxy must be absent from BOTH feature sets
    for forbidden in features.FORBIDDEN_IN_X:
        assert forbidden not in feats_a, f"{forbidden} leaked into Model A"
        assert forbidden not in feats_b, f"{forbidden} leaked into Model B"
    # Model B is a strict superset of Model A
    assert set(feats_a).issubset(set(feats_b))
    # temperature target still present as a column (as the label), just not a feature
    assert "temp_mean_c" in df.columns
