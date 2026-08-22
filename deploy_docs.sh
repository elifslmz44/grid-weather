#!/usr/bin/env bash
# Assemble docs/ for GitHub Pages from the website + the latest computed results.
set -e
cd "$(dirname "$0")"
rm -rf docs && mkdir -p docs/data
cp website/index.html docs/
cp website/.nojekyll docs/ 2>/dev/null || true
cp outputs/web_data/*.json docs/data/ 2>/dev/null || { echo "No web_data JSON — run the pipeline first."; exit 1; }
echo "docs/ ready. Commit it, then enable GitHub Pages: Settings > Pages > Source: main / docs"
