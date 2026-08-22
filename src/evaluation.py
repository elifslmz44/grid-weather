"""
Phase 6 -- rigorous evaluation of the weather-blind model, plus the optional anomaly hunt.

Everything is computed on the chronological holdout (test years), using the best model
(Model B random forest) refit here so the module is self-contained. Five investigations:

  1. Segment breakdowns  -- MAE/RMSE/R2 by season, weekday/weekend, coldest/warmest days.
  2. Cold-spell detection -- treat "is this a cold day?" as classification and report a
                             confusion matrix + precision/recall. Threshold from OBSERVED temp.
  3. Heating/cooling asymmetry -- test (not assume) whether the grid reads cold better than
                             heat, both via model error by regime and the model-free
                             demand<->temperature correlation in each regime.
  4. "When the grid lies" -- the largest residual days, tagged with calendar context.
  5. Demand anomalies (Phase 7) -- days when demand itself deviated most from calendar
                             expectation, with the observed weather alongside (no invented
                             causes: unexplained days are labelled unknown).

Run from the project root:
    python -m src.evaluation
"""

from __future__ import annotations

import json
import logging
import warnings

import numpy as np

# Silence spurious Apple Accelerate BLAS matmul FP flags (see modelling.py for detail).
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
import pandas as pd
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score)

from . import config
from .features import build_feature_table
from .modelling import chronological_split, DEFAULT_TEST_START_YEAR
from .signal_analysis import _setup_mpl, _save_fig, _save_json, INK, ACCENT, ACCENT2, GRID

import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluation")


def _seg_metrics(y_true, y_pred) -> dict:
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None, "n": 0}
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3) if len(y_true) > 1 else None,
        "n": int(len(y_true)),
    }


def _fit_test(test_start_year: int):
    """Refit Model B (RF) on train, return the test frame with predictions + residuals."""
    df, FEATURES_A, FEATURES_B, TARGET = build_feature_table()
    train, test = chronological_split(df, test_start_year)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  n_jobs=-1, random_state=42)
    model.fit(train[FEATURES_B], train[TARGET])
    test = test.copy()
    test["pred"] = model.predict(test[FEATURES_B])
    test["actual"] = test[TARGET]
    test["resid"] = test["actual"] - test["pred"]          # +ve => model too cold
    test["abs_resid"] = test["resid"].abs()
    test["month"] = test["date"].dt.month
    test["season"] = test["month"].map(_season)
    test["is_weekend"] = (test["date"].dt.dayofweek >= 5)
    return train, test, TARGET


def _season(m: int) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(m, "autumn")


# --------------------------------------------------------------------------------------
# 1. Segment breakdowns
# --------------------------------------------------------------------------------------
def segment_breakdowns(test: pd.DataFrame) -> dict:
    out = {"overall": _seg_metrics(test["actual"], test["pred"])}
    for s in ["winter", "spring", "summer", "autumn"]:
        g = test[test.season == s]
        out[s] = _seg_metrics(g["actual"], g["pred"])
    out["weekday"] = _seg_metrics(test[~test.is_weekend]["actual"], test[~test.is_weekend]["pred"])
    out["weekend"] = _seg_metrics(test[test.is_weekend]["actual"], test[test.is_weekend]["pred"])
    # coldest / warmest deciles by OBSERVED temperature
    c_thr = test["actual"].quantile(0.10)
    w_thr = test["actual"].quantile(0.90)
    out["coldest_decile"] = _seg_metrics(test[test.actual <= c_thr]["actual"],
                                         test[test.actual <= c_thr]["pred"])
    out["warmest_decile"] = _seg_metrics(test[test.actual >= w_thr]["actual"],
                                         test[test.actual >= w_thr]["pred"])
    _save_json("eval_segments.json", out)
    return out


# --------------------------------------------------------------------------------------
# 2. Cold-spell detection (classification)
# --------------------------------------------------------------------------------------
def cold_spell_detection(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    # threshold = 10th percentile of TRAINING observed daily-mean temperature
    thr = float(train[target].quantile(0.10))
    y_true = (test["actual"] <= thr).astype(int)
    y_pred = (test["pred"] <= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    result = {
        "threshold_c": round(thr, 2),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 3),
        "n_actual_cold_days": int(y_true.sum()),
        "n_flagged_cold_days": int(y_pred.sum()),
    }
    _save_json("cold_spell.json", result)

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Reds")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not cold", "Cold"]); ax.set_yticklabels(["Not cold", "Cold"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Cold-spell detection (< {thr:.1f} C)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else INK, fontsize=13, weight="bold")
    _save_fig(fig, "11_cold_confusion.png")
    return result


# --------------------------------------------------------------------------------------
# 3. Heating / cooling asymmetry
# --------------------------------------------------------------------------------------
def heating_cooling_asymmetry(test: pd.DataFrame) -> dict:
    med = test["actual"].median()
    cold = test[test.actual < med]
    warm = test[test.actual >= med]
    # model-free: demand<->temperature correlation in each regime
    corr_cold = float(np.corrcoef(cold["nd_mean"], cold["actual"])[0, 1])
    corr_warm = float(np.corrcoef(warm["nd_mean"], warm["actual"])[0, 1])
    result = {
        "median_temp_c": round(float(med), 2),
        "model_mae_cold_half": _seg_metrics(cold["actual"], cold["pred"])["mae"],
        "model_mae_warm_half": _seg_metrics(warm["actual"], warm["pred"])["mae"],
        "demand_temp_corr_cold_half": round(corr_cold, 3),
        "demand_temp_corr_warm_half": round(corr_warm, 3),
        "interpretation": (
            "stronger demand-temperature coupling in the cold half"
            if abs(corr_cold) > abs(corr_warm) else
            "coupling not stronger in the cold half"),
    }
    _save_json("asymmetry.json", result)

    # scatter: demand vs temperature, coloured by regime, with regime fits
    fig, ax = plt.subplots()
    ax.scatter(test["actual"], test["nd_mean"], s=8, color=GRID, alpha=0.6)
    for g, c, lab in [(cold, ACCENT, "cold half"), (warm, ACCENT2, "warm half")]:
        b, a = np.polyfit(g["actual"], g["nd_mean"], 1)
        xs = np.linspace(g["actual"].min(), g["actual"].max(), 50)
        ax.plot(xs, a + b * xs, color=c, lw=2, label=f"{lab} (slope {b:,.0f} MW/C)")
    ax.set_title("Demand vs temperature: the heating asymmetry")
    ax.set_xlabel("Observed daily mean temperature (C)")
    ax.set_ylabel("Daily mean demand (MW)")
    ax.legend(frameon=False)
    _save_fig(fig, "12_asymmetry.png")
    return result


# --------------------------------------------------------------------------------------
# 4. "When the grid lies" -- largest residuals
# --------------------------------------------------------------------------------------
def largest_residuals(test: pd.DataFrame, n: int = 20) -> dict:
    years = range(test["date"].dt.year.min(), test["date"].dt.year.max() + 1)
    uk = holidays.UnitedKingdom(years=list(years))
    top = test.reindex(test["abs_resid"].sort_values(ascending=False).index).head(n)

    rows = []
    for _, r in top.iterrows():
        d = r["date"].date()
        rows.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "weekday": r["date"].strftime("%A"),
            "actual_c": round(float(r["actual"]), 2),
            "predicted_c": round(float(r["pred"]), 2),
            "residual_c": round(float(r["resid"]), 2),
            "context": uk.get(d) or ("weekend" if r["date"].dayofweek >= 5 else "unknown"),
        })
    result = {"largest_residuals": rows,
              "note": "residual = actual - predicted; +ve means the model predicted too cold"}
    _save_json("largest_residuals.json", result)

    # time series + residual-by-month + scatter
    _setup_mpl()
    t = test.sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t["date"], t["actual"], color=INK, lw=1.1, label="Observed")
    ax.plot(t["date"], t["pred"], color=ACCENT, lw=1.1, alpha=0.85, label="Inferred from grid")
    ax.set_title("Inferred vs observed temperature (holdout years)")
    ax.set_ylabel("Daily mean temp (C)")
    ax.legend(frameon=False)
    _save_fig(fig, "09_actual_vs_inferred.png")

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(t["actual"], t["pred"], s=9, color=ACCENT2, alpha=0.5)
    lim = [min(t.actual.min(), t.pred.min()), max(t.actual.max(), t.pred.max())]
    ax.plot(lim, lim, color=INK, lw=1, ls="--")
    ax.set_title("Predicted vs actual")
    ax.set_xlabel("Observed (C)"); ax.set_ylabel("Inferred (C)")
    _save_fig(fig, "10_pred_vs_actual.png")

    fig, ax = plt.subplots()
    by_month = [t[t.month == m]["resid"] for m in range(1, 13)]
    ax.boxplot(by_month, tick_labels=["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.axhline(0, color=ACCENT, lw=1)
    ax.set_title("Residuals by month")
    ax.set_ylabel("Actual - inferred (C)")
    _save_fig(fig, "13_residual_by_month.png")
    return result


# --------------------------------------------------------------------------------------
# 5. Demand anomalies (Phase 7) -- when Britain behaved strangely
# --------------------------------------------------------------------------------------
def demand_anomalies(test_start_year: int, n: int = 15) -> dict:
    """Expected demand from CALENDAR only; biggest deviations are behavioural anomalies."""
    df, FEATURES_A, FEATURES_B, TARGET = build_feature_table()
    cal = ["month_sin", "month_cos", "doy_sin", "doy_cos", "dow", "is_weekend",
           "is_holiday", "daylight_hours"]
    train = df[df["date"].dt.year < test_start_year]
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                  n_jobs=-1, random_state=42)
    model.fit(train[cal], train["nd_mean"])
    d = df.copy()
    d["expected_nd"] = model.predict(d[cal])
    d["deviation"] = d["nd_mean"] - d["expected_nd"]
    d["abs_dev"] = d["deviation"].abs()

    years = range(d["date"].dt.year.min(), d["date"].dt.year.max() + 1)
    uk = holidays.UnitedKingdom(years=list(years))
    top = d.reindex(d["abs_dev"].sort_values(ascending=False).index).head(n)
    rows = []
    for _, r in top.iterrows():
        dd = r["date"].date()
        rows.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "weekday": r["date"].strftime("%A"),
            "expected_mw": round(float(r["expected_nd"]), 0),
            "actual_mw": round(float(r["nd_mean"]), 0),
            "deviation_mw": round(float(r["deviation"]), 0),
            "observed_temp_c": round(float(r["temp_mean_c"]), 2),
            "context": uk.get(dd) or ("weekend" if r["date"].dayofweek >= 5 else "unknown"),
        })
    result = {"anomalies": rows,
              "note": "deviation = actual - calendar-expected demand; causes not established "
                      "are labelled unknown."}
    _save_json("demand_anomalies.json", result)
    return result


def run(test_start_year: int = DEFAULT_TEST_START_YEAR) -> dict:
    _setup_mpl()
    train, test, target = _fit_test(test_start_year)
    seg = segment_breakdowns(test)
    cold = cold_spell_detection(train, test, target)
    asym = heating_cooling_asymmetry(test)
    resid = largest_residuals(test)
    anom = demand_anomalies(test_start_year)

    print("\n=== Phase 6 evaluation (report back) ===")
    print(f"Overall (Model B, holdout): MAE {seg['overall']['mae']} C, "
          f"RMSE {seg['overall']['rmse']} C, R2 {seg['overall']['r2']}")
    print(f"  winter MAE {seg['winter']['mae']}  summer MAE {seg['summer']['mae']}")
    print(f"  coldest-decile MAE {seg['coldest_decile']['mae']}  "
          f"warmest-decile MAE {seg['warmest_decile']['mae']}")
    print(f"\nCold-spell detection (< {cold['threshold_c']} C): "
          f"precision {cold['precision']}, recall {cold['recall']}, F1 {cold['f1']}")
    print(f"  confusion: {cold['confusion_matrix']}")
    print(f"\nHeating/cooling asymmetry:")
    print(f"  demand-temp corr  cold {asym['demand_temp_corr_cold_half']}  "
          f"warm {asym['demand_temp_corr_warm_half']}")
    print(f"  model MAE  cold {asym['model_mae_cold_half']}  warm {asym['model_mae_warm_half']}")
    print(f"\nTop 3 'grid lies' days:")
    for r in resid["largest_residuals"][:3]:
        print(f"  {r['date']} ({r['weekday']}, {r['context']}): "
              f"actual {r['actual_c']} vs inferred {r['predicted_c']} "
              f"(off by {r['residual_c']:+} C)")
    print(f"\nTop 3 demand anomalies:")
    for r in anom["anomalies"][:3]:
        print(f"  {r['date']} ({r['context']}): actual {r['actual_mw']:,.0f} vs "
              f"expected {r['expected_mw']:,.0f} MW ({r['deviation_mw']:+,.0f})")
    print("\nFigures -> outputs/figures/   JSON -> outputs/web_data/")
    return {"segments": seg, "cold": cold, "asymmetry": asym}


if __name__ == "__main__":
    run()
