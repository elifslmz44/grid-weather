"""
SQL / DuckDB transformation layer.

The main pipeline builds its daily table in pandas. This module rebuilds the same star schema
*in SQL* over DuckDB, straight off the tidy CSVs -- a small dimensional model:

    dim_date  ──<  fact_demand_daily
              ──<  fact_weather_daily

It exists to (a) show the transformations as declarative SQL rather than imperative pandas, and
(b) **reconcile**: it proves the SQL daily aggregation reproduces the pandas one to floating-point
tolerance. A transformation layer you can't reconcile against a source of truth isn't worth much.

Run from the project root:
    python -m src.warehouse
"""

from __future__ import annotations

import json
import logging

import duckdb
import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("warehouse")

SQL_DIR = config.PROJECT_ROOT / "sql"
DEMAND_CSV = config.PROCESSED_DIR / "demand_halfhourly.csv"
WEATHER_CSV = config.PROCESSED_DIR / "weather_hourly.csv"
DAILY_CSV = config.PROCESSED_DIR / "daily.csv"


def _run_sql_file(con, name, replacements=None):
    sql = (SQL_DIR / name).read_text()
    for token, value in (replacements or {}).items():
        sql = sql.replace(token, value)
    con.execute(sql)


def build(con=None):
    if con is None:
        con = duckdb.connect()  # in-memory
    _run_sql_file(con, "01_staging.sql",
                  {"__DEMAND_CSV__": str(DEMAND_CSV), "__WEATHER_CSV__": str(WEATHER_CSV)})
    _run_sql_file(con, "02_schema.sql")
    _run_sql_file(con, "03_analytics.sql")
    return con


def reconcile(con) -> dict:
    """Prove the SQL fact table matches the pandas daily.csv on mean demand."""
    pandas_daily = pd.read_csv(DAILY_CSV, parse_dates=["date"])[["date", "nd_mean"]]
    pandas_daily["date"] = pandas_daily["date"].dt.date
    sql_daily = con.execute(
        "SELECT date_key AS date, nd_mean FROM fact_demand_daily ORDER BY 1").df()
    sql_daily["date"] = pd.to_datetime(sql_daily["date"]).dt.date

    merged = pandas_daily.merge(sql_daily, on="date", suffixes=("_pandas", "_sql"))
    max_abs_diff = float((merged["nd_mean_pandas"] - merged["nd_mean_sql"]).abs().max())
    return {
        "pandas_rows": int(len(pandas_daily)),
        "sql_rows": int(len(sql_daily)),
        "matched_rows": int(len(merged)),
        "max_abs_diff_mw": round(max_abs_diff, 6),
        "reconciled": bool(len(merged) == len(pandas_daily) and max_abs_diff < 1e-3),
    }


def run() -> dict:
    con = build()
    recon = reconcile(con)
    if recon["reconciled"]:
        log.info("Reconciliation OK: SQL == pandas on %d days (max diff %.2e MW)",
                 recon["matched_rows"], recon["max_abs_diff_mw"])
    else:
        log.warning("Reconciliation MISMATCH: %s", recon)

    season = con.execute("SELECT * FROM mart_season").df().to_dict("records")
    weekday = con.execute("SELECT * FROM mart_weekday").df().to_dict("records")
    temp_resp = con.execute("SELECT * FROM mart_temp_response").df().to_dict("records")

    payload = {
        "reconciliation": recon,
        "tables": [
            {"name": "dim_date", "grain": "one row per calendar day",
             "columns": ["date_key", "year", "month", "dow", "day_of_year", "is_weekend", "season"]},
            {"name": "fact_demand_daily", "grain": "one row per day",
             "columns": ["date_key", "nd_mean", "nd_max", "nd_min", "nd_range", "nd_total_mwh", "n_periods"]},
            {"name": "fact_weather_daily", "grain": "one row per day",
             "columns": ["date_key", "temp_mean_c", "temp_min_c", "temp_max_c"]},
        ],
        "sample_query":
            "SELECT season, ROUND(AVG(nd_mean),0) AS mean_demand_mw, COUNT(*) AS days\n"
            "FROM v_daily GROUP BY season ORDER BY mean_demand_mw DESC;",
        "mart_season": season,
        "mart_weekday": weekday,
        "mart_temp_response": temp_resp[:40],
    }

    web = config.OUTPUTS_DIR / "web_data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "warehouse.json").write_text(json.dumps(payload))
    log.info("Wrote warehouse.json (season/weekday/temp marts, %d temp bins).", len(temp_resp))
    return payload


if __name__ == "__main__":
    run()
