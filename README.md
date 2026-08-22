# Can Britain's Electricity Grid Reveal the Weather?

*An experiment in extracting environmental signals from electricity demand — without showing the model the temperature.*

> **Status:** work in progress. Results below are populated **only after** they are computed on
> real data. No performance numbers are invented; any metric shown has been produced by the
> evaluation pipeline in this repository.

## The question

Electricity demand responds to how people behave, and how people behave responds to the
weather. This project treats the Great Britain electricity grid as an **indirect sensor**: if
the observed temperature were hidden, how much could we infer about British weather from
electricity-demand patterns alone? The model is trained on grid-derived and calendar features
with **temperature deliberately withheld**, and only *after* it makes its predictions do we
join real weather observations to see how close it got — and, more interestingly, where and
why it fails.

## Why this project?

It sits at the intersection of physics (periodic signals, Fourier analysis, rate-of-change),
data engineering (real API ingestion, time-series cleaning, chronological validation) and
energy systems. The intellectual value is in the experimental design and interpretation, not
in chasing a leaderboard score.

## Data

| Layer | Source | Access | Granularity | Notes |
|---|---|---|---|---|
| Electricity | NESO **Historic Demand Data** (CKAN Data Portal) | Open API, no key | Half-hourly (48 settlement periods/day) | `ND`, `TSD`, embedded wind/solar, interconnectors. Published ~21 days in arrears; solar/demand subject to retrospective correction. |
| Weather | **Open-Meteo** Historical Weather API (ERA5 reanalysis) | Open API, no key | Hourly | Population-weighted across 10 GB cities. See substitution note below. |

**Weather substitution (documented deliberately):** the spec's first choice is Met Office
station observations. Those require credentials and give patchy long-run historical coverage,
so this build uses ERA5 reanalysis via Open-Meteo. ERA5 blends station, satellite, aircraft and
buoy observations through a numerical model — it is *not* raw station data, and is optimised for
consistency over pinpoint daily accuracy. The weather layer is isolated in `src/ingest_weather.py`
so it can be swapped for Met Office DataHub later without touching the rest of the pipeline.

**National weather proxy construction:** hourly 2 m temperature is fetched for 10 GB population
centres (London, Birmingham, Manchester, Leeds, Glasgow, Sheffield, Bristol, Newcastle,
Liverpool, Edinburgh) and combined into a single series weighted by approximate urban
population, so the national estimate leans toward where electricity load actually is. Weights
are listed in `src/config.py`.

**Study period:** 2015–2025 (configurable in `src/config.py`).

## Architecture

```
        NESO CKAN API                         Open-Meteo ERA5 API
              │                                        │
              ▼                                        ▼
   Raw electricity CSVs                    Raw per-city temperature JSON
     (data/raw/, cached)                     (data/raw/weather/, cached)
              │                                        │
              └───────────────┬────────────────────────┘
                              ▼
                Cleaning · validation · timezone/DST alignment
                              ▼
                     Feature engineering
              (weather-blind: grid + calendar only)
                              ▼
        ┌─────────────────────┴─────────────────────┐
        ▼                                            ▼
   Model A: electricity-only            Model B: electricity + calendar
        └─────────────────────┬─────────────────────┘
                              ▼
                 Predicted temperature  ──►  join REAL weather
                              ▼
        Chronological evaluation · cold-spell detection · residual analysis
                              ▼
              Frozen JSON/CSV  ──►  interactive website
```

## Methodology (planned)

- **Weather-blind features only.** Temperature is never an input. Two variants are compared:
  **Model A** (electricity/grid-derived features) and **Model B** (adds hour/day/month/daylight
  calendar context). The comparison answers: *how much weather signal is in electricity
  behaviour, versus simply knowing the time of year?*
- **Baselines first:** day-of-year climatology, then linear regression, before any tree ensemble.
- **Chronological holdout.** Earlier years train; later years test. Time-series order is never
  shuffled.
- **Metrics:** MAE, RMSE, R², broken down by season, weekday/weekend, and temperature extremes.
- **Cold-spell detection** framed as classification (precision / recall / confusion matrix), plus
  an explicit test of heating-vs-cooling asymmetry.

## Results

On a strict chronological holdout (train 2015–2022, test 2023–2025), the weather-blind model
reconstructs Britain's daily mean temperature to **±1.77 °C (RMSE 2.19, R² 0.82)** — having
never seen a thermometer.

| Model | Features | MAE (°C) | RMSE (°C) | R² |
|---|---|---|---|---|
| Climatology | day-of-year average only | 2.14 | 2.74 | 0.72 |
| Electricity only (RF) | demand behaviour, no calendar | 2.73 | 3.40 | 0.57 |
| Electricity + calendar (RF) | demand + month/daylight/holiday | **1.77** | **2.19** | **0.82** |

The honest finding is nuanced. Electricity demand *alone* (no calendar) is a **worse** guide to
temperature than simply knowing the date — the seasonal cycle dominates. But demand *added to*
the calendar beats climatology by **0.38 °C (≈18% lower error)**, so the grid genuinely carries
temperature information beyond seasonality. The strongest demand signals are the **7-day rolling
mean of demand** (importance 0.51) and the **overnight minimum** (0.20) — i.e. sustained load and
baseline heating, exactly the heating-load fingerprint.

Cold-spell detection, the heating-vs-cooling asymmetry, and the largest-error failure analysis are
computed in `src/evaluation.py` and rendered live on the website from `outputs/web_data/`.

## Limitations

*To be expanded as analysis proceeds.* Known up front: ERA5 is reanalysis not station data;
NESO recent weeks are provisional; calendar features can leak season, which is exactly why the
A/B split exists; correlation between demand and temperature does **not** imply the grid
"measures" weather.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 2 — ingestion (safe to re-run; everything is cached)
python -m src.ingest_neso            # electricity, configured year window
python -m src.ingest_weather         # population-weighted GB temperature (fetched in UTC)

# Phase 3 -- clean, validate, align (writes data/processed/ + a data-quality report)
python -m src.clean

# run the test suite (DST timestamp logic, dedup, validation)
pytest -q
```

```bash
# Phase 4 -- exploratory + signal analysis (figures + web_data JSON)
python -m src.signal_analysis
```

```bash
# Phase 5 -- weather-blind features + Model A/B (metrics + predictions to outputs/)
python -m src.modelling
```

```bash
# Phase 6 -- evaluation: segments, cold-spell detection, asymmetry, failure analysis
python -m src.evaluation
```

```bash
# Phase 8 -- bundle results into the static site, then preview
python -m src.build_site          # writes website/data.js from outputs/web_data/
# open website/index.html in a browser, or serve locally:
python -m http.server -d website 8000   # then visit http://localhost:8000
```

### The website

The site (`website/index.html`) is a single self-contained page that reads the JSON in
`outputs/web_data/`. Copy the results in and preview locally:

```bash
cp outputs/web_data/*.json website/data/
cd website && python -m http.server 8000   # open http://localhost:8000
```

**Deploy** (two easy options):
- *Netlify drop* — run the copy step above, then drag the `website/` folder onto https://app.netlify.com/drop for an instant URL.
- *GitHub Pages* — run `./deploy_docs.sh`, commit the generated `docs/`, then set Settings > Pages > Source to `main /docs`.

# List available NESO year resources without downloading:
python -m src.ingest_neso --list
```

No API keys are required for the default pipeline.

## Live site

*Deploy with `./deploy_docs.sh` (GitHub Pages) or Netlify drop — see The website above.*
