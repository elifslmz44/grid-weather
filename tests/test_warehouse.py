"""The SQL/DuckDB star schema must reproduce the pandas daily aggregation exactly."""
import pandas as pd
import pytest

from src import config

duckdb = pytest.importorskip("duckdb")
from src import warehouse  # noqa: E402


def _make_processed(tmp_path, monkeypatch):
    """Point the warehouse at tiny synthetic processed CSVs."""
    rng = pd.date_range("2020-01-01", "2020-01-05 23:30", freq="30min", tz="UTC")
    nd = 30000 + (rng.hour * 100)
    dem = pd.DataFrame({"timestamp_utc": rng, "nd": nd, "tsd": nd + 4000})
    wx = pd.DataFrame({"timestamp_utc": rng[::2], "gb_temp_pop_weighted": 8.0})
    d = dem.copy()
    d["date"] = d["timestamp_utc"].dt.date
    daily = d.groupby("date").agg(nd_mean=("nd", "mean")).reset_index()

    (tmp_path / "demand_halfhourly.csv").write_text(dem.to_csv(index=False))
    (tmp_path / "weather_hourly.csv").write_text(wx.to_csv(index=False))
    (tmp_path / "daily.csv").write_text(daily.to_csv(index=False))

    monkeypatch.setattr(warehouse, "DEMAND_CSV", tmp_path / "demand_halfhourly.csv")
    monkeypatch.setattr(warehouse, "WEATHER_CSV", tmp_path / "weather_hourly.csv")
    monkeypatch.setattr(warehouse, "DAILY_CSV", tmp_path / "daily.csv")


def test_sql_reconciles_with_pandas(tmp_path, monkeypatch):
    _make_processed(tmp_path, monkeypatch)
    con = warehouse.build()
    recon = warehouse.reconcile(con)
    assert recon["reconciled"] is True
    assert recon["matched_rows"] == recon["pandas_rows"] == 5
    assert recon["max_abs_diff_mw"] < 1e-6


def test_star_schema_tables_exist(tmp_path, monkeypatch):
    _make_processed(tmp_path, monkeypatch)
    con = warehouse.build()
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    assert {"dim_date", "fact_demand_daily", "fact_weather_daily"} <= tables
