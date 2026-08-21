"""
Tests for src/clean.py -- focused on the parts most likely to hide subtle bugs:
timestamp construction across DST boundaries, de-duplication, and implausible-value handling.

Run from the project root:
    pytest -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import clean, config


def _make_day(date_str: str, n_periods: int, nd: float = 25_000.0) -> pd.DataFrame:
    return pd.DataFrame({
        "settlement_date": [date_str] * n_periods,
        "settlement_period": list(range(1, n_periods + 1)),
        "nd": [nd] * n_periods,
        "tsd": [nd + 4000] * n_periods,
    })


def test_normal_day_has_48_periods_and_correct_span():
    df = _make_day("2025-01-15", 48)
    out = clean.build_demand_timestamps(df)
    assert out["timestamp_utc"].iloc[0] == pd.Timestamp("2025-01-15 00:00", tz="UTC")
    assert out["timestamp_utc"].iloc[-1] == pd.Timestamp("2025-01-15 23:30", tz="UTC")
    assert str(out["timestamp_local"].iloc[0]).startswith("2025-01-15 00:00")


def test_spring_forward_day_maps_46_periods_to_23_hours():
    df = _make_day("2025-03-30", 46)
    out = clean.build_demand_timestamps(df)
    assert out["timestamp_utc"].iloc[0] == pd.Timestamp("2025-03-30 00:00", tz="UTC")
    assert out["timestamp_utc"].iloc[-1] == pd.Timestamp("2025-03-30 22:30", tz="UTC")
    assert str(out["timestamp_local"].iloc[-1]).startswith("2025-03-30 23:30")
    assert clean._expected_periods_for_date("2025-03-30") == 46


def test_autumn_fallback_day_maps_50_periods_to_25_hours():
    df = _make_day("2025-10-26", 50)
    out = clean.build_demand_timestamps(df)
    assert out["timestamp_utc"].iloc[0] == pd.Timestamp("2025-10-25 23:00", tz="UTC")
    assert out["timestamp_utc"].iloc[-1] == pd.Timestamp("2025-10-26 23:30", tz="UTC")
    assert clean._expected_periods_for_date("2025-10-26") == 50


def test_timestamps_are_strictly_increasing_across_a_dst_day():
    df = _make_day("2025-10-26", 50)
    out = clean.build_demand_timestamps(df)
    assert out["timestamp_utc"].is_monotonic_increasing
    assert out["timestamp_utc"].nunique() == 50


def test_clean_demand_dedups_and_drops_impossible(monkeypatch, tmp_path):
    raw = pd.concat([
        _make_day("2025-01-15", 48),
        _make_day("2025-01-15", 1),
    ], ignore_index=True)
    raw.loc[raw.index[-1], "nd"] = 999_999
    raw.loc[5, "nd"] = 0

    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "processed"
    raw_dir.mkdir(); proc_dir.mkdir()
    raw.to_csv(raw_dir / "neso_demand_2025.csv", index=False)

    monkeypatch.setattr(config, "RAW_DIR", raw_dir)
    monkeypatch.setattr(config, "PROCESSED_DIR", proc_dir)

    tidy = clean.clean_demand()
    assert len(tidy) == 47
    assert tidy["nd"].min() >= clean.ND_MIN_PLAUSIBLE
    assert tidy.attrs["n_dupes"] == 1
    assert tidy.attrs["n_impossible"] >= 1
