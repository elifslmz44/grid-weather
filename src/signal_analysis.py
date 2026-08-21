"""
Phase 4 -- exploratory + signal analysis of the electricity demand series.

Two halves, matching the project's two "understand the signal" stages:

  Stage 1 (shape):  hour-of-day profile, weekday vs weekend, hour x month heatmap,
                    monthly distribution, annual profile, long-term trend.
  Stage 2 (physics): Fourier periodogram, autocorrelation, morning-ramp (derivative).

Design choices worth knowing:
  * Behavioural views (hour-of-day, weekday, month) use LOCAL clock time -- that is what human
    routines actually track. The Fourier and autocorrelation work uses the UTC series because
    it is a strictly uniform 30-minute grid with no gaps (verified in Phase 3), which is what
    spectral methods require.
  * Every analysis emits BOTH a figure (outputs/figures/, for the repo/notebook) and a compact
    JSON aggregate (outputs/web_data/, for the eventual website to read). No raw data is shipped
    to the frontend -- only these small summaries.

Run from the project root:
    python -m src.signal_analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt

from . import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signal_analysis")

WEB_DATA_DIR = config.OUTPUTS_DIR / "web_data"
WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Restrained, editorial figure style (scientific publication feel, not dashboard).
INK = "#1a1a1a"
ACCENT = "#c1440e"      # warm brick red
ACCENT2 = "#2a6f97"     # cool blue
GRID = "#e6e6e6"


def _setup_mpl() -> None:
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    })


def _save_json(name: str, obj: dict) -> None:
    path = WEB_DATA_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=str))
    log.info("Wrote %s", path.relative_to(config.PROJECT_ROOT))


def _save_fig(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.savefig(path)
    plt.close(fig)
    log.info("Wrote %s", path.relative_to(config.PROJECT_ROOT))


# --------------------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------------------
def load_processed() -> tuple[pd.DataFrame, pd.DataFrame]:
    dpath = config.PROCESSED_DIR / "demand_halfhourly.csv"
    if not dpath.exists():
        raise FileNotFoundError("Run `python -m src.clean` first.")
    demand = pd.read_csv(dpath)
    demand["timestamp_utc"] = pd.to_datetime(demand["timestamp_utc"], utc=True)
    demand["timestamp_local"] = pd.to_datetime(demand["timestamp_local"], utc=True) \
        .dt.tz_convert(config.TIMEZONE)
    daily = pd.read_csv(config.PROCESSED_DIR / "daily.csv", parse_dates=["date"])
    log.info("Loaded %d half-hours, %d days.", len(demand), len(daily))
    return demand, daily


# --------------------------------------------------------------------------------------
# Stage 1 -- demand shape
# --------------------------------------------------------------------------------------
def hour_of_day_profile(demand: pd.DataFrame) -> dict:
    """Average demand across the day, overall and split winter/summer, at half-hour resolution."""
    d = demand.copy()
    loc = d["timestamp_local"]
    d["hh"] = loc.dt.hour + loc.dt.minute / 60.0
    d["season"] = np.where(loc.dt.month.isin([12, 1, 2]), "winter",
                    np.where(loc.dt.month.isin([6, 7, 8]), "summer", "other"))

    overall = d.groupby("hh")["nd"].mean()
    winter = d[d.season == "winter"].groupby("hh")["nd"].mean()
    summer = d[d.season == "summer"].groupby("hh")["nd"].mean()

    fig, ax = plt.subplots()
    ax.plot(overall.index, overall.values, color=INK, lw=2, label="All year")
    ax.plot(winter.index, winter.values, color=ACCENT, lw=1.8, label="Winter (DJF)")
    ax.plot(summer.index, summer.values, color=ACCENT2, lw=1.8, label="Summer (JJA)")
    ax.set_title("Britain's daily demand rhythm")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Mean National Demand (MW)")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(frameon=False)
    _save_fig(fig, "01_hour_of_day.png")

    result = {
        "hour": [round(h, 2) for h in overall.index.tolist()],
        "all_year_mw": overall.round(0).tolist(),
        "winter_mw": winter.round(0).tolist(),
        "summer_mw": summer.round(0).tolist(),
        "peak_hour_local": float(overall.idxmax()),
        "trough_hour_local": float(overall.idxmin()),
        "peak_mw": float(overall.max()),
        "trough_mw": float(overall.min()),
    }
    _save_json("hour_of_day.json", result)
    return result


def weekday_weekend_profile(demand: pd.DataFrame) -> dict:
    d = demand.copy()
    loc = d["timestamp_local"]
    d["hh"] = loc.dt.hour + loc.dt.minute / 60.0
    d["is_weekend"] = loc.dt.dayofweek >= 5
    wd = d[~d.is_weekend].groupby("hh")["nd"].mean()
    we = d[d.is_weekend].groupby("hh")["nd"].mean()

    fig, ax = plt.subplots()
    ax.plot(wd.index, wd.values, color=ACCENT, lw=2, label="Weekday")
    ax.plot(we.index, we.values, color=ACCENT2, lw=2, label="Weekend")
    ax.fill_between(wd.index, we.values, wd.values, where=(wd.values >= we.values),
                    color=ACCENT, alpha=0.08)
    ax.set_title("Weekday vs weekend demand")
    ax.set_xlabel("Hour of day (local time)")
    ax.set_ylabel("Mean National Demand (MW)")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(frameon=False)
    _save_fig(fig, "02_weekday_weekend.png")

    gap = (wd - we)
    result = {
        "hour": [round(h, 2) for h in wd.index.tolist()],
        "weekday_mw": wd.round(0).tolist(),
        "weekend_mw": we.round(0).tolist(),
        "mean_weekday_minus_weekend_mw": float(gap.mean().round(0)),
        "max_gap_mw": float(gap.max().round(0)),
        "max_gap_hour": float(gap.idxmax()),
    }
    _save_json("weekday_weekend.json", result)
    return result


def hour_month_heatmap(demand: pd.DataFrame) -> dict:
    d = demand.copy()
    loc = d["timestamp_local"]
    d["hour"] = loc.dt.hour
    d["month"] = loc.dt.month
    piv = d.pivot_table(index="hour", columns="month", values="nd", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(piv.values, aspect="auto", origin="lower", cmap="magma")
    ax.set_title("Demand by hour and month (MW)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_yticks(range(0, 24, 3))
    fig.colorbar(im, ax=ax, label="Mean ND (MW)")
    _save_fig(fig, "03_hour_month_heatmap.png")

    result = {
        "hours": piv.index.tolist(),
        "months": piv.columns.tolist(),
        "z_mw": piv.round(0).values.tolist(),
    }
    _save_json("hour_month_heatmap.json", result)
    return result


def monthly_distribution(daily: pd.DataFrame) -> dict:
    d = daily.copy()
    d["month"] = d["date"].dt.month
    q = d.groupby("month")["nd_mean"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()

    fig, ax = plt.subplots()
    ax.fill_between(q.index, q[0.1], q[0.9], color=ACCENT, alpha=0.12, label="10-90th pct")
    ax.fill_between(q.index, q[0.25], q[0.75], color=ACCENT, alpha=0.25, label="25-75th pct")
    ax.plot(q.index, q[0.5], color=ACCENT, lw=2, label="Median")
    ax.set_title("Daily-mean demand by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Daily mean ND (MW)")
    ax.set_xticks(range(1, 13))
    ax.legend(frameon=False)
    _save_fig(fig, "04_monthly_distribution.png")

    result = {"month": q.index.tolist(),
              "p10": q[0.1].round(0).tolist(), "p25": q[0.25].round(0).tolist(),
              "median": q[0.5].round(0).tolist(), "p75": q[0.75].round(0).tolist(),
              "p90": q[0.9].round(0).tolist()}
    _save_json("monthly_distribution.json", result)
    return result


def annual_and_trend(daily: pd.DataFrame) -> dict:
    d = daily.sort_values("date").copy()
    d["roll30"] = d["nd_mean"].rolling(30, min_periods=15).mean()

    # long-term trend via linear fit on ordinal days
    x = (d["date"] - d["date"].min()).dt.days.values.astype(float)
    y = d["nd_mean"].values
    slope, intercept = np.polyfit(x, y, 1)
    trend_mw_per_year = slope * 365.25

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(d["date"], d["nd_mean"], color=GRID, lw=0.6)
    ax.plot(d["date"], d["roll30"], color=INK, lw=1.6, label="30-day rolling mean")
    ax.plot(d["date"], intercept + slope * x, color=ACCENT, lw=1.5, ls="--",
            label=f"Trend: {trend_mw_per_year:,.0f} MW/yr")
    ax.set_title("Daily mean demand over the study period")
    ax.set_ylabel("Daily mean ND (MW)")
    ax.legend(frameon=False)
    _save_fig(fig, "05_longterm_trend.png")

    result = {
        "date": d["date"].dt.strftime("%Y-%m-%d").tolist(),
        "daily_mean_mw": d["nd_mean"].round(0).tolist(),
        "roll30_mw": d["roll30"].round(0).tolist(),
        "trend_mw_per_year": float(round(trend_mw_per_year, 1)),
        "trend_pct_per_year": float(round(100 * trend_mw_per_year / y.mean(), 3)),
    }
    _save_json("annual_and_trend.json", result)
    return result


# --------------------------------------------------------------------------------------
# Stage 2 -- physics / signal processing
# --------------------------------------------------------------------------------------
def fourier_spectrum(demand: pd.DataFrame) -> dict:
    """
    Periodogram of the UTC half-hourly ND series. Removes mean + linear trend first so the
    spectrum reflects periodic structure, not the slow decline. Reports the strongest peaks
    as physical periods (hours), which should land on 24h, 12h, the weekly 168h, and annual.
    """
    d = demand.sort_values("timestamp_utc")
    x = d["nd"].to_numpy(dtype=float)
    n = len(x)
    # detrend (remove mean + linear)
    t = np.arange(n)
    coef = np.polyfit(t, x, 1)
    x_detrended = x - (coef[0] * t + coef[1])
    # Hann window to reduce spectral leakage
    win = np.hanning(n)
    xw = x_detrended * win

    fs_per_hour = 2.0  # samples per hour (30-min sampling)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_per_hour)   # cycles per hour
    power = np.abs(np.fft.rfft(xw)) ** 2
    # ignore the zero-frequency bin
    freqs, power = freqs[1:], power[1:]
    periods_h = 1.0 / freqs

    # find peaks: local maxima above a threshold, then take the strongest by power
    peak_idx = []
    for i in range(1, len(power) - 1):
        if power[i] > power[i - 1] and power[i] > power[i + 1]:
            peak_idx.append(i)
    peak_idx = sorted(peak_idx, key=lambda i: power[i], reverse=True)[:12]
    peak_idx = sorted(peak_idx, key=lambda i: periods_h[i])

    def label(ph: float) -> str:
        for target, name in [(24, "24 h (daily)"), (12, "12 h (half-day)"),
                             (8, "8 h"), (6, "6 h"), (168, "168 h (weekly)"),
                             (84, "84 h"), (8766, "~1 year (annual)"), (4383, "~6 months")]:
            if abs(ph - target) / target < 0.05:
                return name
        return f"{ph:,.1f} h"

    top = [{"period_hours": round(float(periods_h[i]), 2),
            "period_label": label(periods_h[i]),
            "relative_power": round(float(power[i] / power.max()), 4)}
           for i in peak_idx]
    top = sorted(top, key=lambda r: r["relative_power"], reverse=True)[:8]

    # figure: power vs period (log-x), annotate expected lines
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.semilogx(periods_h, power / power.max(), color=INK, lw=0.9)
    for target, txt in [(24, "day"), (12, "½ day"), (168, "week"), (8766, "year")]:
        ax.axvline(target, color=ACCENT, ls=":", lw=1)
        ax.text(target, 1.02, txt, color=ACCENT, ha="center", va="bottom", fontsize=9)
    ax.set_title("Hidden frequencies in electricity demand")
    ax.set_xlabel("Period (hours, log scale)")
    ax.set_ylabel("Relative spectral power")
    ax.set_ylim(0, 1.1)
    _save_fig(fig, "06_fourier_spectrum.png")

    # downsample the spectrum for the web (log-spaced) to keep JSON small
    sel = np.unique(np.geomspace(1, len(periods_h) - 1, 1200).astype(int))
    result = {
        "period_hours": [round(float(periods_h[i]), 3) for i in sel],
        "relative_power": [round(float(power[i] / power.max()), 5) for i in sel],
        "top_peaks": top,
    }
    _save_json("fourier_spectrum.json", result)
    return result


def autocorrelation(demand: pd.DataFrame, max_lag_hours: int = 24 * 15) -> dict:
    """Autocorrelation of the UTC half-hourly series up to ~15 days, in half-hour lag steps."""
    x = demand.sort_values("timestamp_utc")["nd"].to_numpy(dtype=float)
    x = x - x.mean()
    var = np.dot(x, x)
    max_lag = int(max_lag_hours * 2)
    lags = np.arange(0, max_lag + 1)
    acf = np.empty(len(lags))
    for k in lags:
        acf[k] = np.dot(x[:len(x) - k], x[k:]) / var if k > 0 else 1.0
    lag_hours = lags / 2.0

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(lag_hours, acf, color=INK, lw=1)
    for day in range(1, 16):
        ax.axvline(day * 24, color=GRID, lw=0.8)
    ax.axvline(24, color=ACCENT, ls=":", lw=1)
    ax.axvline(168, color=ACCENT2, ls=":", lw=1)
    ax.set_title("Autocorrelation: demand remembers its rhythms")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("Autocorrelation")
    _save_fig(fig, "07_autocorrelation.png")

    def at(hours):
        idx = int(hours * 2)
        return round(float(acf[idx]), 4) if idx < len(acf) else None

    result = {
        "lag_hours": [round(float(h), 2) for h in lag_hours[::2]],  # hourly resolution for web
        "acf": [round(float(a), 5) for a in acf[::2]],
        "acf_at_24h": at(24), "acf_at_48h": at(48), "acf_at_168h": at(168),
    }
    _save_json("autocorrelation.json", result)
    return result


def morning_ramp(demand: pd.DataFrame) -> dict:
    """
    Rate-of-change analysis: the morning ramp is the steepest sustained rise each day. We
    measure the max half-hourly increase in ND per day and how it varies by season.
    """
    d = demand.sort_values("timestamp_utc").copy()
    d["dnd"] = d["nd"].diff()  # MW per 30 min
    d["date"] = d["timestamp_local"].dt.date
    d["month"] = d["timestamp_local"].dt.month
    daily_max_ramp = d.groupby("date")["dnd"].max()
    by_month = d.groupby("month")["dnd"].max()

    fig, ax = plt.subplots()
    ax.bar(by_month.index, by_month.values, color=ACCENT, alpha=0.85)
    ax.set_title("Steepest morning ramp by month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Max Δ demand (MW / 30 min)")
    ax.set_xticks(range(1, 13))
    _save_fig(fig, "08_morning_ramp.png")

    result = {
        "month": by_month.index.tolist(),
        "max_ramp_mw_per_30min": by_month.round(0).tolist(),
        "overall_max_ramp_mw_per_30min": float(daily_max_ramp.max().round(0)),
        "median_daily_max_ramp_mw_per_30min": float(daily_max_ramp.median().round(0)),
    }
    _save_json("morning_ramp.json", result)
    return result


# --------------------------------------------------------------------------------------
# Orchestrate
# --------------------------------------------------------------------------------------
def run_all() -> None:
    _setup_mpl()
    demand, daily = load_processed()

    hod = hour_of_day_profile(demand)
    ww = weekday_weekend_profile(demand)
    hour_month_heatmap(demand)
    monthly_distribution(daily)
    trend = annual_and_trend(daily)
    fft = fourier_spectrum(demand)
    acf = autocorrelation(demand)
    ramp = morning_ramp(demand)

    print("\n=== Phase 4 key findings (report these back) ===")
    print(f"Peak demand hour (local)      : {hod['peak_hour_local']:.1f}h "
          f"({hod['peak_mw']:,.0f} MW)")
    print(f"Trough demand hour (local)    : {hod['trough_hour_local']:.1f}h "
          f"({hod['trough_mw']:,.0f} MW)")
    print(f"Mean weekday-weekend gap      : {ww['mean_weekday_minus_weekend_mw']:,.0f} MW "
          f"(max {ww['max_gap_mw']:,.0f} MW at {ww['max_gap_hour']:.1f}h)")
    print(f"Long-term demand trend        : {trend['trend_mw_per_year']:,.0f} MW/yr "
          f"({trend['trend_pct_per_year']:+.2f}% /yr)")
    print(f"ACF at 24h / 168h             : {acf['acf_at_24h']} / {acf['acf_at_168h']}")
    print(f"Max morning ramp              : {ramp['overall_max_ramp_mw_per_30min']:,.0f} "
          f"MW / 30 min")
    print("\nTop Fourier peaks:")
    for p in fft["top_peaks"]:
        print(f"   {p['period_label']:<22} relative power {p['relative_power']:.3f}")
    print(f"\nFigures -> outputs/figures/   JSON -> outputs/web_data/")


if __name__ == "__main__":
    run_all()
