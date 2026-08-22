"""
Phase 5b -- baselines and the weather-blind inference models.

The experiment compares, on a strict chronological holdout:
    climatology  -- predict the training-set average temperature for that day-of-year.
                    This is the "you only know the season" reference. Beating it means the
                    grid carries temperature information BEYOND the calendar.
    Model A      -- electricity-demand features only, NO calendar. Linear and Random Forest.
    Model B      -- Model A + calendar/astronomical context. Linear and Random Forest.

Evaluation is chronological: earlier years train, later years test. Time order is never
shuffled. Metrics: MAE, RMSE, R2. Predictions and feature importances are exported to
outputs/ for the website and the deeper Phase 6 residual analysis.

Run from the project root:
    python -m src.modelling
    python -m src.modelling --test-start 2023
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config
from .features import build_feature_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("modelling")

DEFAULT_TEST_START_YEAR = 2023   # train 2015..2022, test 2023..end


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def chronological_split(df: pd.DataFrame, test_start_year: int):
    train = df[df["date"].dt.year < test_start_year].copy()
    test = df[df["date"].dt.year >= test_start_year].copy()
    log.info("Split: train %s..%s (%d), test %s..%s (%d).",
             train["date"].min().date(), train["date"].max().date(), len(train),
             test["date"].min().date(), test["date"].max().date(), len(test))
    return train, test


def climatology_baseline(train, test, target):
    """Predict each test day's temperature as the training mean for that day-of-year."""
    tr = train.copy()
    tr["doy"] = tr["date"].dt.dayofyear
    doy_mean = tr.groupby("doy")[target].mean()
    global_mean = tr[target].mean()
    te_doy = test["date"].dt.dayofyear
    return te_doy.map(doy_mean).fillna(global_mean).to_numpy()


def _fit_predict(model, train, test, feats, target):
    model.fit(train[feats], train[target])
    return model.predict(test[feats]), model


def run(test_start_year: int = DEFAULT_TEST_START_YEAR) -> dict:
    df, FEATURES_A, FEATURES_B, TARGET = build_feature_table()
    train, test = chronological_split(df, test_start_year)
    y_test = test[TARGET].to_numpy()

    results = {}
    predictions = {"date": test["date"].dt.strftime("%Y-%m-%d").tolist(),
                   "actual": [round(float(v), 2) for v in y_test]}
    importances = {}

    # --- climatology (season only) ---
    yhat = climatology_baseline(train, test, TARGET)
    results["climatology"] = _metrics(y_test, yhat)
    predictions["climatology"] = [round(float(v), 2) for v in yhat]

    # --- Model A / B, linear + random forest ---
    specs = [
        ("linear_A", LinearRegression(), FEATURES_A),
        ("linear_B", LinearRegression(), FEATURES_B),
        ("rf_A", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                       n_jobs=-1, random_state=42), FEATURES_A),
        ("rf_B", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                       n_jobs=-1, random_state=42), FEATURES_B),
    ]
    for name, model, feats in specs:
        yhat, fitted = _fit_predict(model, train, test, feats, TARGET)
        results[name] = _metrics(y_test, yhat)
        predictions[name] = [round(float(v), 2) for v in yhat]
        if hasattr(fitted, "feature_importances_"):
            imp = sorted(zip(feats, fitted.feature_importances_),
                         key=lambda kv: kv[1], reverse=True)
            importances[name] = [{"feature": f, "importance": round(float(i), 4)} for f, i in imp]

    # --- persist ---
    config.MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    web = config.OUTPUTS_DIR / "web_data"
    web.mkdir(parents=True, exist_ok=True)

    payload = {
        "test_start_year": test_start_year,
        "train_days": int(len(train)),
        "test_days": int(len(test)),
        "metrics": results,
        "features_A": FEATURES_A,
        "features_B": FEATURES_B,
    }
    (config.MODEL_RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))
    (web / "metrics.json").write_text(json.dumps(payload, indent=2))
    (web / "predictions.json").write_text(json.dumps(predictions))
    (web / "feature_importances.json").write_text(json.dumps(importances, indent=2))

    _print_table(results)
    _print_interpretation(results)
    log.info("Saved metrics + predictions + importances to outputs/.")
    return payload


def _print_table(results: dict) -> None:
    print("\n=== Model comparison (chronological holdout) ===")
    print(f"{'model':<14}{'MAE (C)':>10}{'RMSE (C)':>10}{'R2':>8}")
    order = ["climatology", "linear_A", "rf_A", "linear_B", "rf_B"]
    for name in order:
        if name in results:
            m = results[name]
            print(f"{name:<14}{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['r2']:>8.3f}")


def _print_interpretation(results: dict) -> None:
    clim = results["climatology"]["mae"]
    best_a = min(results[m]["mae"] for m in ("linear_A", "rf_A") if m in results)
    best_b = min(results[m]["mae"] for m in ("linear_B", "rf_B") if m in results)
    print("\n=== What this says (report back) ===")
    print(f"Climatology (season only) MAE : {clim:.2f} C")
    print(f"Best electricity-only  MAE    : {best_a:.2f} C  "
          f"({'beats' if best_a < clim else 'does NOT beat'} season-only by "
          f"{clim - best_a:+.2f} C)")
    print(f"Best electricity+calendar MAE : {best_b:.2f} C")
    if best_a < clim:
        print("-> Electricity DEMAND alone carries temperature signal beyond the calendar.")
    else:
        print("-> Electricity alone does not beat pure seasonality; calendar context matters most.")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Weather-blind temperature inference models.")
    p.add_argument("--test-start", type=int, default=DEFAULT_TEST_START_YEAR,
                   help="first year of the chronological test set")
    args = p.parse_args()
    run(args.test_start)


if __name__ == "__main__":
    _cli()
