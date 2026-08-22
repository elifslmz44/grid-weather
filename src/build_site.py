"""
Phase 8 helper -- bundle the analysis outputs into the website.

Reads every JSON in outputs/web_data/ and writes website/data.js as a single global object:

    window.GRID_DATA = { "metrics": {...}, "hour_of_day": {...}, ... };

Bundling (rather than fetching at runtime) means the site is pure static files: it works when
opened directly and deploys to GitHub Pages with no server or CORS setup. Re-run this whenever
the analysis is refreshed so the site always reflects the real, current results.

Run from the project root:
    python -m src.build_site
"""

from __future__ import annotations

import json
import logging

from . import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("build_site")

WEB_DATA = config.OUTPUTS_DIR / "web_data"
SITE_DIR = config.PROJECT_ROOT / "website"


def build() -> None:
    if not WEB_DATA.exists():
        raise FileNotFoundError("No outputs/web_data. Run the analysis phases first "
                                "(signal_analysis, modelling, evaluation).")
    bundle = {}
    for f in sorted(WEB_DATA.glob("*.json")):
        try:
            bundle[f.stem] = json.loads(f.read_text())
            log.info("bundled %s", f.name)
        except json.JSONDecodeError as exc:
            log.warning("skipping %s (%s)", f.name, exc)

    SITE_DIR.mkdir(exist_ok=True)
    out = SITE_DIR / "data.js"
    out.write_text("window.GRID_DATA = " + json.dumps(bundle, indent=2) + ";\n")
    kb = out.stat().st_size / 1024
    log.info("wrote %s (%d datasets, %.0f KB)", out.relative_to(config.PROJECT_ROOT),
             len(bundle), kb)
    missing = [k for k in ("hour_of_day", "fourier_spectrum", "metrics", "cold_spell")
               if k not in bundle]
    if missing:
        log.warning("site will hide sections lacking data: %s", missing)


if __name__ == "__main__":
    build()
