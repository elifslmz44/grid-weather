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
import warnings

import numpy as np

# Apple's Accelerate BLAS (default on macOS system Python) emits spurious floating-point flags
# during matrix multiply -- "divide by zero / overflow / invalid encountered in matmul" -- even
# when the result is exact. They are harmless here (verified: linear metrics are stable and match
# the tree models), so we silence just these specific messages to keep the console readable.
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
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
    # np.errstate scopes the same spurious BLAS FP flags at the source, belt-and-suspenders
    # with the module-level warnings filter above.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        model.fit(train[feats], train[target])
        preds = model.predict(test[feats])
    return preds, model


def _get_importances(fitted, feats):
    """Return ranked importances for RF (impurity) or a linear model (|standardised coef|)."""
    est = fitted.steps[-1][1] if hasattr(fitted, "steps") else fitted
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        vals = np.abs(np.asarray(est.coef_, dtype=float))  # standardised inputs -> comparable
    else:
        return None
    ranked = sorted(zip(feats, vals), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in ranked]


def walk_forward_cv(df, feats, target, test_start_year, min_train_years=3):
    """Expanding-window time-series CV: for each year, train on all prior years and test on that
    year. Proves the result isn't an artefact of one split, and yields out-of-fold residuals used
    to calibrate prediction intervals. Uses the rf_B configuration (electricity + calendar)."""
    years = sorted(int(y) for y in df["date"].dt.year.unique())
    folds, resid = [], {}
    for i, y in enumerate(years):
        if i < min_train_years:
            continue
        tr = df[df["date"].dt.year < y]
        te = df[df["date"].dt.year == y]
        if len(te) < 30:
            continue
        model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                      n_jobs=-1, random_state=42)
        preds, _ = _fit_predict(model, tr, te, feats, target)
        yt = te[target].to_numpy()
        m = _metrics(yt, preds)
        folds.append({"year": y, "mae": round(m["mae"], 3), "rmse": round(m["rmse"], 3),
                      "r2": round(m["r2"], 3), "n": m["n"],
                      "climatology_mae": round(float(mean_absolute_error(
                          yt, climatology_baseline(tr, te, target))), 3)})
        resid[y] = yt - preds
    return folds, resid


def shap_explain(fitted, test, feats, target, dates):
    """Optional SHAP explainability for the rf_B model. Guarded: if shap isn't installed the
    pipeline simply skips it (the site hides the panel), so CI can never break on this."""
    try:
        import shap
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("shap not available — skipping explainability export (%s)", exc)
        return None
    Xte = test[feats]
    explainer = shap.TreeExplainer(fitted)
    sv = np.asarray(explainer.shap_values(Xte), dtype=float)   # (n_days, n_feats)
    base = float(np.mean(explainer.expected_value))
    preds = fitted.predict(Xte)
    yt = test[target].to_numpy()

    mean_abs = np.abs(sv).mean(axis=0)
    global_imp = sorted(zip(feats, mean_abs), key=lambda kv: kv[1], reverse=True)

    # three illustrative days: coldest, median, warmest observed
    order = np.argsort(yt)
    picks = {"coldest": int(order[0]), "typical": int(order[len(order) // 2]), "warmest": int(order[-1])}
    examples = []
    for label, idx in picks.items():
        contribs = sorted(zip(feats, sv[idx]), key=lambda kv: abs(kv[1]), reverse=True)[:8]
        examples.append({
            "label": label, "date": dates[idx],
            "actual": round(float(yt[idx]), 1), "predicted": round(float(preds[idx]), 1),
            "contributions": [{"feature": f, "shap": round(float(s), 3)} for f, s in contribs],
        })
    return {
        "base_value": round(base, 2),
        "global": [{"feature": f, "mean_abs_shap": round(float(v), 4)} for f, v in global_imp],
        "examples": examples,
    }


def explorer_model(test_start_year):
    """Small linear model for the site's interactive widget: temperature ~ average demand +
    month (as a cycle) + weekend. Linear coefficients run in the browser as a dot product. Not the
    headline model; its holdout MAE is exported too so the widget can state the accuracy gap."""
    d = pd.read_csv(config.PROCESSED_DIR / "daily.csv", parse_dates=["date"]) \
        .dropna(subset=["nd_mean", "temp_mean_c"])
    month = d["date"].dt.month.to_numpy()
    ang = 2 * np.pi * month / 12
    weekend = (d["date"].dt.dayofweek >= 5).astype(float).to_numpy()
    X = np.column_stack([d["nd_mean"].to_numpy(), np.sin(ang), np.cos(ang), weekend])
    y = d["temp_mean_c"].to_numpy()
    train = d["date"].dt.year.to_numpy() < test_start_year
    lr = LinearRegression().fit(X[train], y[train])
    mae = float(mean_absolute_error(y[~train], lr.predict(X[~train]))) if (~train).any() else 0.0
    c = lr.coef_
    return {
        "coef": {"intercept": float(lr.intercept_), "nd_mean": float(c[0]),
                 "sin_month": float(c[1]), "cos_month": float(c[2]), "weekend": float(c[3])},
        "demand_min_mw": round(float(np.percentile(d["nd_mean"], 2)), 0),
        "demand_max_mw": round(float(np.percentile(d["nd_mean"], 98)), 0),
        "demand_med_mw": round(float(d["nd_mean"].median()), 0),
        "mae_c": round(mae, 2),
    }


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

    # --- Model A / B: a regularised linear model and a random forest each ---
    # "linear" = StandardScaler + Ridge. Plain OLS is numerically unstable here because several
    # demand features are collinear (e.g. nd_range == nd_max - nd_min); ridge + scaling fixes it
    # and standardised coefficients double as an interpretable importance measure.
    specs = [
        ("linear_A", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), FEATURES_A),
        ("linear_B", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), FEATURES_B),
        ("rf_A", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                       n_jobs=-1, random_state=42), FEATURES_A),
        ("rf_B", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                       n_jobs=-1, random_state=42), FEATURES_B),
        ("gbm_B", HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                max_depth=None, random_state=42), FEATURES_B),
    ]
    fitted_models = {}
    for name, model, feats in specs:
        yhat, fitted = _fit_predict(model, train, test, feats, TARGET)
        results[name] = _metrics(y_test, yhat)
        predictions[name] = [round(float(v), 2) for v in yhat]
        fitted_models[name] = fitted
        imp = _get_importances(fitted, feats)
        if imp is not None:
            importances[name] = imp

    # --- walk-forward cross-validation + prediction intervals (rf_B) ---
    folds, resid = walk_forward_cv(df, FEATURES_B, TARGET, test_start_year)
    maes = [f["mae"] for f in folds]
    # calibrate a 90% interval half-width on out-of-fold residuals from *pre-holdout* years,
    # then measure the coverage actually achieved on the untouched holdout.
    pre = [resid[y] for y in resid if y < test_start_year]
    calib = np.concatenate(pre) if pre else (np.concatenate(list(resid.values())) if resid else np.array([0.0]))
    target_cov = 0.90
    q = float(np.quantile(np.abs(calib), target_cov)) if calib.size else 0.0
    rfb = np.array(predictions["rf_B"], dtype=float)
    predictions["rf_B_lower"] = [round(float(v - q), 2) for v in rfb]
    predictions["rf_B_upper"] = [round(float(v + q), 2) for v in rfb]
    holdout_cov = float(np.mean(np.abs(y_test - rfb) <= q)) if len(y_test) else 0.0
    cv_payload = {
        "folds": folds,
        "mae_mean": round(float(np.mean(maes)), 3) if maes else None,
        "mae_std": round(float(np.std(maes)), 3) if maes else None,
        "mae_min": round(float(np.min(maes)), 3) if maes else None,
        "mae_max": round(float(np.max(maes)), 3) if maes else None,
        "n_folds": len(folds),
        "target_coverage": target_cov,
        "interval_half_width_c": round(q, 2),
        "holdout_coverage": round(holdout_cov, 3),
    }

    # --- persist ---
    config.MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    web = config.OUTPUTS_DIR / "web_data"
    web.mkdir(parents=True, exist_ok=True)
    (web / "cv.json").write_text(json.dumps(cv_payload, indent=2))

    # --- SHAP explainability (optional, guarded) ---
    shap_payload = shap_explain(fitted_models.get("rf_B"), test, FEATURES_B, TARGET, predictions["date"]) \
        if fitted_models.get("rf_B") is not None else None
    if shap_payload is not None:
        (web / "shap.json").write_text(json.dumps(shap_payload, indent=2))

    # --- tiny interactive explorer model (runs in the browser) ---
    (web / "explorer.json").write_text(json.dumps(explorer_model(test_start_year), indent=2))

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
    print(f"Climatology (season only)      : {clim:.2f} C MAE")
    print(f"Best electricity-only (no cal) : {best_a:.2f} C  -- demand ALONE vs the calendar: "
          f"{'better' if best_a < clim else 'worse'} by {abs(clim - best_a):.2f} C")
    print(f"Best electricity + calendar    : {best_b:.2f} C")
    print(f"\nKey comparison -- both know the season, but B also sees demand:")
    delta = clim - best_b
    pct = 100 * delta / clim
    if best_b < clim:
        print(f"  Model B beats climatology by {delta:.2f} C ({pct:.0f}% lower error).")
        print(f"  -> Electricity demand carries temperature signal BEYOND the calendar,")
        print(f"     worth about {delta:.2f} C of accuracy. But demand alone is a weaker")
        print(f"     thermometer than simply knowing the date.")
    else:
        print(f"  Model B does not beat climatology -> little weather signal beyond season.")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Weather-blind temperature inference models.")
    p.add_argument("--test-start", type=int, default=DEFAULT_TEST_START_YEAR,
                   help="first year of the chronological test set")
    args = p.parse_args()
    run(args.test_start)


if __name__ == "__main__":
    _cli()
