# Data directory

Nothing here is committed to git except this file and `.gitkeep` placeholders. Everything
else is **regenerated** by the ingestion pipeline so the repository stays small and the
provenance stays transparent.

```
data/
├── raw/                     # exactly what the APIs returned, cached (git-ignored)
│   ├── neso_demand_YYYY.csv
│   └── weather/
│       ├── <city>_<start>_<end>.json
│       └── gb_temperature_hourly.csv
└── processed/               # cleaned, aligned, validated (git-ignored)
```

Regenerate with:

```bash
python -m src.ingest_neso
python -m src.ingest_weather
```
