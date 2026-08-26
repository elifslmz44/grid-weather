"""
Phase 10 -- honest short-horizon forecast of GB electricity DEMAND (not weather).

This is deliberately a transparent seasonal-trend model, not a black box:

    demand ~ long-term trend  +  annual seasonality (Fourier terms)
             +  day-of-week   +  UK public holidays

fitted with a ridge regression on daily mean National Demand. It extrapolates the calendar
structure forward, so it captures the season, the working week and the slow multi-year decline --
but NOT the effect of upcoming weather (a production forecaster would ingest a weather forecast).
That limitation is stated on the site; the point here is a well-backtested, interpretable baseline.

Error bands come from an expanding-window backtest (train up to an origin, forecast forward, measure
the real out-of-sample error), so they reflect how the model actually performs, not in-sample hope.

Run from the project root:
    python -m src.forecast
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import holidays
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("forecast")

HORIZON_DAYS = 60          # how far ahead we forecast
N_HARMONICS = 4            # annual Fourier terms
BACKTEST_ORIGINS = 8       # expanding-window backtest folds
BACKTEST_STEP_DAYS = 30    # spacing between backtest origins


def _calendar_features(dates: pd.DatetimeIndex, t0: pd.Timestamp) -> pd.DataFrame:
    frac = 2 * np.pi * dates.dayofyear.to_numpy() / 365.25
    cols = {"trend": (dates - t0).days.to_numpy().astype(float)}
    for k in range(1, N_HARMONICS + 1):
        cols[f"sin{k}"] = np.sin(k * frac)
        cols[f"cos{k}"] = np.cos(k * frac)
    dow = dates.dayofweek.to_numpy()
    for d in range(7):
        cols[f"dow{d}"] = (dow == d).astype(float)
    yrs = list(range(int(dates.year.min()), int(dates.year.max()) + 1))
    uk = holidays.UnitedKingdom(years=yrs)
    cols["holiday"] = np.array([1.0 if d.date() in uk else 0.0 for d in dates])
    return pd.DataFrame(cols, index=dates)


def _fit(train: pd.DataFrame, t0: pd.Timestamp):
    X = _calendar_features(pd.DatetimeIndex(train["date"]), t0)
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    model.fit(X.to_numpy(), train["nd_mean"].to_numpy())
    return model


def _predict(model, dates: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
    return model.predict(_calendar_features(dates, t0).to_numpy())


def run() -> dict:
    daily = pd.read_csv(config.PROCESSED_DIR / "daily.csv", parse_dates=["date"]).sort_values("date")
    daily = daily.dropna(subset=["nd_mean"]).reset_index(drop=True)
    t0 = daily["date"].iloc[0]

    # --- expanding-window backtest: real out-of-sample residuals by horizon ---
    resid = []
    end = daily["date"].iloc[-1]
    for i in range(BACKTEST_ORIGINS, 0, -1):
        origin = end - pd.Timedelta(days=i * BACKTEST_STEP_DAYS)
        tr = daily[daily["date"] <= origin]
        if len(tr) < 365 * 2:
            continue
        fut = daily[(daily["date"] > origin) &
                    (daily["date"] <= origin + pd.Timedelta(days=HORIZON_DAYS))]
        if fut.empty:
            continue
        model = _fit(tr, t0)
        pred = _predict(model, pd.DatetimeIndex(fut["date"]), t0)
        resid.extend((fut["nd_mean"].to_numpy() - pred).tolist())
    resid = np.array(resid, dtype=float)

    if resid.size:
        mae = float(np.mean(np.abs(resid)))
        mape = float(np.mean(np.abs(resid) / daily["nd_mean"].mean()) * 100)
        q90 = float(np.quantile(np.abs(resid), 0.90))
        coverage = float(np.mean(np.abs(resid) <= q90))
    else:  # not enough history to backtest
        mae = mape = q90 = coverage = 0.0

    # --- final forecast: train on everything, predict the next HORIZON_DAYS ---
    model = _fit(daily, t0)
    future_dates = pd.date_range(end + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")
    fc = _predict(model, future_dates, t0)

    tail = daily.tail(120)
    payload = {
        "history": {
            "date": tail["date"].dt.strftime("%Y-%m-%d").tolist(),
            "demand_mw": [round(float(v), 0) for v in tail["nd_mean"]],
        },
        "forecast": {
            "date": [d.strftime("%Y-%m-%d") for d in future_dates],
            "demand_mw": [round(float(v), 0) for v in fc],
            "lower_mw": [round(float(v - q90), 0) for v in fc],
            "upper_mw": [round(float(v + q90), 0) for v in fc],
        },
        "backtest": {
            "mae_mw": round(mae, 0), "mape_pct": round(mape, 1),
            "band_mw": round(q90, 0), "coverage": round(coverage, 3),
            "horizon_days": HORIZON_DAYS, "origins": BACKTEST_ORIGINS,
        },
    }

    web = config.OUTPUTS_DIR / "web_data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "forecast.json").write_text(json.dumps(payload))
    log.info("Forecast %d days ahead | backtest MAE %.0f MW (%.1f%%), 90%% band +/-%.0f MW, coverage %.0f%%",
             HORIZON_DAYS, mae, mape, q90, coverage * 100)
    return payload


if __name__ == "__main__":
    run()
