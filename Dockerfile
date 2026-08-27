# Reproducible one-command run of the whole pipeline.
#   docker build -t grid-weather .
#   docker run --rm -v "$PWD/docs:/app/docs" grid-weather
# The mounted volume lets the rebuilt docs/data.js land back on your host.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ingest → clean → analyse → model → evaluate → forecast → warehouse → bundle site
CMD ["bash", "-lc", "python -m src.ingest_neso && python -m src.ingest_weather && python -m src.clean && python -m src.signal_analysis && python -m src.modelling && python -m src.evaluation && python -m src.forecast && python -m src.warehouse && python -m src.build_site"]
