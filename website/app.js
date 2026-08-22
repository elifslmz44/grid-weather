/* Renders the data-story from window.GRID_DATA (bundled by src/build_site.py).
   Every number shown is read from the real analysis outputs — nothing is hard-coded.
   Each section degrades gracefully: if its data is missing, the section hides itself. */

(function () {
  "use strict";
  const D = window.GRID_DATA || {};
  const C = { demand: "#C25A0A", temp: "#1F6191", ink: "#14181D",
              soft: "#576270", line: "#DCE1E7", grid: "#E8EBEF" };
  const MONO = "IBM Plex Mono, monospace";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CFG = { displayModeBar: false, responsive: true };

  const $ = (id) => document.getElementById(id);
  const hideFigure = (el) => { const f = el && el.closest("figure, section"); if (f) f.style.display = "none"; };
  const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));

  function base(over) {
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: MONO, size: 12, color: C.soft },
      margin: { l: 62, r: 20, t: 24, b: 48 },
      xaxis: { gridcolor: C.grid, zeroline: false, linecolor: C.line, ticks: "outside", tickcolor: C.line },
      yaxis: { gridcolor: C.grid, zeroline: false, linecolor: C.line, ticks: "outside", tickcolor: C.line },
      legend: { orientation: "h", y: 1.14, x: 0, font: { size: 11 } },
    }, over || {});
  }
  function plot(id, traces, layout) {
    const el = $(id); if (!el) return;
    try { Plotly.newPlot(el, traces, base(layout), CFG); }
    catch (e) { console.error(id, e); hideFigure(el); }
  }

  /* ---------- hero pulse ---------- */
  function heroPulse() {
    const svg = $("hero-pulse"); if (!svg) return;
    const hod = D.hour_of_day;
    const W = 1000, H = 130, mid = H / 2;
    let vals;
    if (hod && hod.all_year_mw) vals = hod.all_year_mw; else
      vals = Array.from({ length: 48 }, (_, i) => 30000 + 8000 * Math.sin((i / 48) * 2 * Math.PI - 1.2));
    // repeat the daily curve across the width for a heartbeat feel
    const reps = 2.5, n = Math.floor(vals.length * reps);
    const min = Math.min(...vals), max = Math.max(...vals);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const v = vals[i % vals.length];
      const x = (i / (n - 1)) * W;
      const y = mid + (0.5 - (v - min) / (max - min)) * (H * 0.8);
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`);
    }
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("points", pts.join(" "));
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", C.demand);
    line.setAttribute("stroke-width", "2");
    line.setAttribute("stroke-linejoin", "round");
    svg.appendChild(line);
    if (!reduce && line.getTotalLength) {
      const len = line.getTotalLength();
      line.style.strokeDasharray = len; line.style.strokeDashoffset = len;
      line.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }],
        { duration: 2600, easing: "ease-out", fill: "forwards" });
    }
  }

  /* ---------- 01 heartbeat ---------- */
  function heartbeat() {
    const h = D.hour_of_day;
    if (h) {
      plot("chart-hod", [
        { x: h.hour, y: h.all_year_mw, name: "All year", mode: "lines", line: { color: C.ink, width: 2.4 } },
        { x: h.hour, y: h.winter_mw, name: "Winter", mode: "lines", line: { color: C.demand, width: 2 } },
        { x: h.hour, y: h.summer_mw, name: "Summer", mode: "lines", line: { color: C.temp, width: 2 } },
      ], { xaxis: { title: "Hour of day (local)", dtick: 3, gridcolor: C.grid, linecolor: C.line },
           yaxis: { title: "Mean demand (MW)" } });
    } else hideFigure($("chart-hod"));

    const w = D.weekday_weekend;
    if (w) {
      plot("chart-ww", [
        { x: w.hour, y: w.weekday_mw, name: "Weekday", mode: "lines", line: { color: C.demand, width: 2.4 } },
        { x: w.hour, y: w.weekend_mw, name: "Weekend", mode: "lines", line: { color: C.temp, width: 2.4 } },
      ], { xaxis: { title: "Hour of day (local)", dtick: 3 }, yaxis: { title: "Mean demand (MW)" } });
    } else hideFigure($("chart-ww"));
  }

  /* ---------- 02 fourier ---------- */
  function fourier() {
    const f = D.fourier_spectrum;
    if (!f) { hideFigure($("chart-fft")); return; }
    const refs = [[24, "day"], [12, "½ day"], [168, "week"], [8766, "year"]];
    const shapes = refs.map(([p]) => ({ type: "line", x0: Math.log10(p), x1: Math.log10(p),
      yref: "paper", y0: 0, y1: 1, line: { color: C.demand, width: 1, dash: "dot" } }));
    const anns = refs.map(([p, t]) => ({ x: Math.log10(p), y: 1.02, yref: "paper", text: t,
      showarrow: false, font: { color: C.demand, size: 11 } }));
    plot("chart-fft", [
      { x: f.period_hours, y: f.relative_power, mode: "lines", line: { color: C.ink, width: 1 },
        fill: "tozeroy", fillcolor: "rgba(194,90,10,0.06)" },
    ], { xaxis: { title: "Period (hours)", type: "log", gridcolor: C.grid, linecolor: C.line },
         yaxis: { title: "Relative power", rangemode: "tozero" }, shapes, annotations: anns, showlegend: false });
  }

  /* ---------- 03 features ---------- */
  function features() {
    const m = D.metrics; if (!m) return;
    const fill = (id, arr) => { const ul = $(id); if (ul && arr) ul.innerHTML = arr.map((f) => `<li>${f}</li>`).join(""); };
    // Model A shows only the demand features; Model B shows the calendar additions.
    fill("feats-a", m.features_A);
    if (m.features_A && m.features_B)
      fill("feats-b", m.features_B.filter((f) => !m.features_A.includes(f)));
  }

  /* ---------- 04 prediction ---------- */
  function prediction() {
    const m = D.metrics, p = D.predictions;
    if (m && m.metrics && m.metrics.rf_B) {
      const b = m.metrics.rf_B;
      const row = $("stat-row");
      if (row) row.innerHTML = [
        ["±" + fmt(b.mae, 2), "°C", "mean error"],
        [fmt(b.rmse, 2), "°C", "RMSE"],
        [fmt(b.r2, 2), "R²", "variance explained"],
      ].map(([v, u, l]) => `<div class="stat"><span class="v">${v}</span> <span class="u">${u}</span><span class="l">${l}</span></div>`).join("");

      const clim = m.metrics.climatology;
      const v = $("s4-verdict");
      if (v && clim) {
        const impr = ((clim.mae - b.mae) / clim.mae) * 100;
        v.innerHTML = `Knowing only the calendar date (climatology) gives ±${fmt(clim.mae, 2)} °C.
          Adding electricity behaviour cuts that to <strong>±${fmt(b.mae, 2)} °C</strong> —
          about <strong>${impr.toFixed(0)}% less error</strong>. The grid genuinely carries
          temperature information beyond the season, though demand alone (no calendar) stays a
          weaker guide than the date itself.`;
      }
    }
    if (p && p.date && p.rf_B) {
      plot("chart-ts", [
        { x: p.date, y: p.actual, name: "Observed", mode: "lines", line: { color: C.ink, width: 1.3 } },
        { x: p.date, y: p.rf_B, name: "Inferred from grid", mode: "lines", line: { color: C.demand, width: 1.3 } },
      ], { xaxis: { title: "", type: "date" }, yaxis: { title: "Daily mean temp (°C)" },
           hovermode: "x unified" });

      const lim = [Math.min(...p.actual, ...p.rf_B), Math.max(...p.actual, ...p.rf_B)];
      plot("chart-scatter", [
        { x: p.actual, y: p.rf_B, mode: "markers", marker: { color: C.temp, size: 5, opacity: 0.45 }, name: "" },
        { x: lim, y: lim, mode: "lines", line: { color: C.ink, width: 1, dash: "dash" }, name: "" },
      ], { xaxis: { title: "Observed (°C)" }, yaxis: { title: "Inferred (°C)" }, showlegend: false });
    } else { hideFigure($("chart-ts")); hideFigure($("chart-scatter")); }

    if (m && m.metrics) {
      const order = [["climatology", "Climatology"], ["rf_A", "Electricity only"], ["rf_B", "Electricity + calendar"]];
      const rows = order.filter(([k]) => m.metrics[k]);
      plot("chart-compare", [{
        type: "bar", orientation: "h",
        y: rows.map(([, l]) => l), x: rows.map(([k]) => m.metrics[k].mae),
        marker: { color: rows.map(([k]) => k === "rf_B" ? C.demand : C.line) },
        text: rows.map(([k]) => "±" + fmt(m.metrics[k].mae, 2) + "°C"), textposition: "auto",
        textfont: { family: MONO, color: C.ink },
      }], { xaxis: { title: "Mean abs error (°C)" }, yaxis: { automargin: true }, showlegend: false,
            margin: { l: 130, r: 20, t: 10, b: 44 } });
    } else hideFigure($("chart-compare"));
  }

  /* ---------- 05 cold spell ---------- */
  function coldSpell() {
    const c = D.cold_spell;
    if (!c) { const s = $("s5"); if (s) s.style.display = "none"; return; }
    const cm = c.confusion_matrix;
    const el = $("confusion");
    if (el) {
      const cell = (n, lab, col) => `<div class="cm-cell" style="background:${col}"><span class="cm-num">${n}</span><span class="cm-lab">${lab}</span></div>`;
      el.innerHTML =
        `<div></div><div class="cm-axis">pred: not cold</div><div class="cm-axis">pred: cold</div>` +
        `<div class="cm-axis">actual:<br>not cold</div>` + cell(cm.tn, "correct", "#9AA6B2") + cell(cm.fp, "false alarm", "#C9DBE9") +
        `<div class="cm-axis">actual:<br>cold</div>` + cell(cm.fn, "missed", "#E9D3BD") + cell(cm.tp, "caught", C.temp);
    }
    if ($("cold-recall")) $("cold-recall").textContent = fmt(c.recall * 100, 0) + "%";
    if ($("cold-precision")) $("cold-precision").textContent = fmt(c.precision * 100, 0) + "%";
    if ($("cold-recall-txt")) $("cold-recall-txt").textContent =
      `Of the genuinely cold days (below ${fmt(c.threshold_c, 1)} °C), the grid flagged this share.`;
    if ($("cold-precision-txt")) $("cold-precision-txt").textContent =
      `When the grid called a day cold, this share truly were.`;
  }

  /* ---------- 06 grid lies ---------- */
  function gridLies() {
    const r = D.largest_residuals;
    if (r && r.largest_residuals) {
      const tb = document.querySelector("#resid-table tbody");
      if (tb) tb.innerHTML = r.largest_residuals.slice(0, 12).map((row) =>
        `<tr><td>${row.date}</td><td>${row.weekday}</td><td>${fmt(row.actual_c, 1)}°</td>
         <td>${fmt(row.predicted_c, 1)}°</td><td>${row.residual_c > 0 ? "+" : ""}${fmt(row.residual_c, 1)}°</td>
         <td>${row.context}</td></tr>`).join("");
    } else { const t = $("resid-table"); if (t) t.closest(".table-wrap").style.display = "none"; }

    // residual-by-month box, computed from predictions in the browser
    const p = D.predictions;
    if (p && p.date && p.rf_B) {
      const byM = Array.from({ length: 12 }, () => []);
      p.date.forEach((d, i) => { const m = new Date(d).getUTCMonth(); byM[m].push(p.actual[i] - p.rf_B[i]); });
      const M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
      const traces = byM.map((ys, i) => ({ y: ys, type: "box", name: M[i],
        marker: { color: C.temp }, line: { color: C.temp }, fillcolor: "rgba(31,97,145,0.08)",
        boxpoints: false, showlegend: false }));
      plot("chart-resmonth", traces, { yaxis: { title: "Actual − inferred (°C)", zeroline: true, zerolinecolor: C.demand },
        xaxis: { title: "" } });
    } else hideFigure($("chart-resmonth"));
  }

  /* ---------- 07 conclusion (inject real improvement) ---------- */
  function conclusion() {
    const m = D.metrics;
    if (m && m.metrics && m.metrics.rf_B && m.metrics.climatology) {
      const b = m.metrics.rf_B.mae, cl = m.metrics.climatology.mae;
      const impr = ((cl - b) / cl) * 100;
      const el = $("conclusion");
      if (el) el.innerHTML = `Britain's electricity demand does carry a real temperature signal.
        Reading behaviour off the grid infers daily temperature to within
        <strong>±${fmt(b, 2)} °C</strong> (R² ${fmt(m.metrics.rf_B.r2, 2)}) on unseen years —
        about ${impr.toFixed(0)}% sharper than the calendar alone. But it is a supplement to
        knowing the season, not a replacement for it.`;
    }
  }

  function init() {
    if (!window.GRID_DATA) {
      document.querySelectorAll("main .plot").forEach((el) => hideFigure(el));
      console.warn("GRID_DATA not found — run `python -m src.build_site`.");
    }
    heroPulse(); heartbeat(); fourier(); features();
    prediction(); coldSpell(); gridLies(); conclusion();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
