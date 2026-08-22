/* Renders the readout from window.GRID_DATA (bundled by src/build_site.py).
   Every number is read from the real analysis outputs. Amber = demand, cyan = temperature.

   Motion: charts draw themselves as they scroll into view (lines rise from the baseline, bars
   grow from zero); panels and cards fade/slide up, staggered; charts show an oscilloscope
   crosshair on hover. All motion is skipped under prefers-reduced-motion. */

(function () {
  "use strict";
  const D = window.GRID_DATA || {};
  const C = { demand: "#ffb000", temp: "#52d6c6", ink: "#f2e3bf", soft: "#a4906a",
              grid: "#241a0d", line: "#2c2114", red: "#ff5a4d" };
  const MONO = "IBM Plex Mono, monospace";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CFG = { displayModeBar: false, responsive: true };

  const $ = (id) => document.getElementById(id);
  const hide = (el) => { const f = el && el.closest(".panel, .telemetry, section"); if (f) f.style.display = "none"; };
  const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));

  const AX = { gridcolor: C.grid, zerolinecolor: C.line, linecolor: C.line,
               tickfont: { family: MONO, size: 11, color: C.soft },
               titlefont: { family: MONO, size: 12, color: C.soft }, ticks: "outside", tickcolor: C.line };
  // axis variant with an oscilloscope crosshair that follows the cursor
  const AXS = Object.assign({}, AX, { showspikes: true, spikecolor: C.demand, spikethickness: 1,
               spikedash: "dot", spikemode: "across", spikesnap: "cursor" });

  function base(over) {
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: MONO, size: 12, color: C.soft },
      margin: { l: 62, r: 20, t: 16, b: 46 }, showlegend: false,
      xaxis: Object.assign({}, AX), yaxis: Object.assign({}, AX),
      hoverlabel: { font: { family: MONO }, bgcolor: "#100c08", bordercolor: C.line },
    }, over || {});
  }
  // normal plot (scatter, box, confusion handled elsewhere)
  function plot(id, traces, layout) {
    const el = $(id); if (!el) return;
    try { Plotly.newPlot(el, traces, base(layout), CFG); } catch (e) { console.error(id, e); hide(el); }
  }
  // line chart that DRAWS IN: plot flat at the baseline, then animate up to the real shape
  function plotLine(id, traces, layout) {
    const el = $(id); if (!el) return;
    try {
      if (reduce) { Plotly.newPlot(el, traces, base(layout), CFG); return; }
      const reals = traces.map((t) => (t.line && Array.isArray(t.y)) ? t.y.slice() : null);
      let mn = Infinity; reals.forEach((a) => { if (a) a.forEach((v) => { if (v < mn) mn = v; }); });
      if (!isFinite(mn)) mn = 0;
      traces.forEach((t, i) => { if (reals[i]) t.y = reals[i].map(() => mn); });
      Plotly.newPlot(el, traces, base(layout), CFG).then(() => {
        requestAnimationFrame(() => Plotly.animate(el,
          { data: traces.map((t, i) => reals[i] ? { y: reals[i] } : {}) },
          { transition: { duration: 1100, easing: "cubic-in-out" }, frame: { duration: 1100 } }));
      });
    } catch (e) { console.error(id, e); try { Plotly.newPlot(el, traces, base(layout), CFG); } catch (_) { hide(el); } }
  }
  // bar chart that GROWS IN from zero (horizontal: animate x)
  function plotBars(id, traces, layout) {
    const el = $(id); if (!el) return;
    try {
      if (reduce) { Plotly.newPlot(el, traces, base(layout), CFG); return; }
      const realX = traces.map((t) => Array.isArray(t.x) ? t.x.slice() : null);
      traces.forEach((t, i) => { if (realX[i]) t.x = realX[i].map(() => 0); });
      Plotly.newPlot(el, traces, base(layout), CFG).then(() => {
        requestAnimationFrame(() => Plotly.animate(el,
          { data: traces.map((t, i) => realX[i] ? { x: realX[i] } : {}) },
          { transition: { duration: 900, easing: "cubic-out" }, frame: { duration: 900 } }));
      });
    } catch (e) { console.error(id, e); try { Plotly.newPlot(el, traces, base(layout), CFG); } catch (_) {} }
  }
  function glow(x, y, color, width) {
    return { x, y, mode: "lines", line: { color, width: (width || 2) * 5 }, opacity: 0.12,
             hoverinfo: "skip", showlegend: false };
  }

  /* ---------- oscilloscope hero (SVG stroke draw) ---------- */
  function heroScope() {
    const svg = $("hero-pulse"); if (!svg || svg.dataset.done) return; svg.dataset.done = "1";
    const W = 1000, H = 230, mid = H / 2, ns = "http://www.w3.org/2000/svg";
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const defs = document.createElementNS(ns, "defs");
    defs.innerHTML = '<filter id="g"><feGaussianBlur stdDeviation="3"/></filter>';
    svg.appendChild(defs);
    for (let gx = 0; gx <= W; gx += 50) addLine(gx, 0, gx, H, "#1c1408", 1);
    for (let gy = 0; gy <= H; gy += 46) addLine(0, gy, W, gy, "#1c1408", 1);
    addLine(0, mid, W, mid, "#2a2010", 1);
    const hod = D.hour_of_day;
    let vals = (hod && hod.all_year_mw) ? hod.all_year_mw
      : Array.from({ length: 48 }, (_, i) => 30000 + 8000 * Math.sin((i / 48) * 2 * Math.PI - 1.2));
    const reps = 2, n = vals.length * reps, min = Math.min(...vals), max = Math.max(...vals);
    const pts = [];
    for (let i = 0; i < n; i++) {
      const v = vals[i % vals.length];
      pts.push([(i / (n - 1)) * W, mid + (0.5 - (v - min) / (max - min)) * (H * 0.72)]);
    }
    const d = "M " + pts.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" L ");
    addPath(d, C.demand, 6, 0.18);
    const trace = addPath(d, C.demand, 2, 1);
    const beam = document.createElementNS(ns, "circle");
    beam.setAttribute("r", "4"); beam.setAttribute("fill", "#fff5dd"); beam.setAttribute("filter", "url(#g)");
    svg.appendChild(beam);
    if (!reduce && trace.getTotalLength) {
      const len = trace.getTotalLength();
      trace.style.strokeDasharray = len; trace.style.strokeDashoffset = len;
      trace.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }],
        { duration: 3000, easing: "linear", fill: "forwards" });
      let t0 = null;
      const step = (ts) => { if (!t0) t0 = ts; const p = Math.min((ts - t0) / 3000, 1);
        const pt = trace.getPointAtLength(len * p); beam.setAttribute("cx", pt.x); beam.setAttribute("cy", pt.y);
        if (p < 1) requestAnimationFrame(step); else beam.style.opacity = 0.6; };
      requestAnimationFrame(step);
    } else { const pt = pts[pts.length - 1]; beam.setAttribute("cx", pt[0]); beam.setAttribute("cy", pt[1]); }
    function addLine(x1, y1, x2, y2, col, w) { const l = document.createElementNS(ns, "line");
      l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2);
      l.setAttribute("stroke", col); l.setAttribute("stroke-width", w); svg.appendChild(l); }
    function addPath(dd, col, w, op) { const p = document.createElementNS(ns, "path");
      p.setAttribute("d", dd); p.setAttribute("fill", "none"); p.setAttribute("stroke", col);
      p.setAttribute("stroke-width", w); p.setAttribute("opacity", op); p.setAttribute("stroke-linejoin", "round");
      svg.appendChild(p); return p; }
  }

  /* ---------- 01 ---------- */
  function heartbeat() {
    const h = D.hour_of_day;
    if (h) plotLine("chart-hod", [
      glow(h.hour, h.all_year_mw, C.demand, 2.4),
      { x: h.hour, y: h.all_year_mw, name: "All year", mode: "lines", line: { color: C.ink, width: 2.4 } },
      { x: h.hour, y: h.winter_mw, name: "Winter", mode: "lines", line: { color: C.demand, width: 2 } },
      { x: h.hour, y: h.summer_mw, name: "Summer", mode: "lines", line: { color: C.temp, width: 2 } },
    ], { xaxis: Object.assign({ title: "hour of day (local)", dtick: 3 }, AXS),
         yaxis: Object.assign({ title: "mean demand (MW)" }, AX) });
    else hide($("chart-hod"));
    const w = D.weekday_weekend;
    if (w) plotLine("chart-ww", [
      { x: w.hour, y: w.weekday_mw, name: "Weekday", mode: "lines", line: { color: C.demand, width: 2.4 } },
      { x: w.hour, y: w.weekend_mw, name: "Weekend", mode: "lines", line: { color: C.temp, width: 2.4 } },
    ], { xaxis: Object.assign({ title: "hour of day (local)", dtick: 3 }, AXS),
         yaxis: Object.assign({ title: "mean demand (MW)" }, AX) });
    else hide($("chart-ww"));
  }

  /* ---------- 02 ---------- */
  function fourier() {
    const f = D.fourier_spectrum; if (!f) { hide($("chart-fft")); return; }
    const refs = [[24, "day"], [12, "½ day"], [168, "week"], [8766, "year"]];
    const shapes = refs.map(([p]) => ({ type: "line", x0: Math.log10(p), x1: Math.log10(p),
      yref: "paper", y0: 0, y1: 1, line: { color: C.temp, width: 1, dash: "dot" } }));
    const anns = refs.map(([p, t]) => ({ x: Math.log10(p), y: 1.04, yref: "paper", text: t,
      showarrow: false, font: { color: C.temp, size: 11, family: MONO } }));
    plotLine("chart-fft", [
      { x: f.period_hours, y: f.relative_power, mode: "lines", line: { color: C.demand, width: 1.4 },
        fill: "tozeroy", fillcolor: "rgba(255,176,0,0.07)" },
    ], { xaxis: Object.assign({ title: "period (hours)", type: "log" }, AXS),
         yaxis: Object.assign({ title: "relative power", rangemode: "tozero" }, AX),
         shapes, annotations: anns });
  }

  /* ---------- 03 ---------- */
  function features() {
    const m = D.metrics; if (!m) return;
    const fill = (id, arr) => { const ul = $(id); if (ul && arr) ul.innerHTML = arr.map((f) => `<li>▸ ${f}</li>`).join(""); };
    fill("feats-a", m.features_A);
    if (m.features_A && m.features_B) fill("feats-b", m.features_B.filter((f) => !m.features_A.includes(f)));
  }

  /* ---------- 04 ---------- */
  function prediction() {
    const m = D.metrics, p = D.predictions;
    if (m && m.metrics && m.metrics.rf_B) {
      const b = m.metrics.rf_B, row = $("stat-row");
      if (row) row.innerHTML = [["±" + fmt(b.mae, 2), "°C", "mean error"],
        [fmt(b.rmse, 2), "°C", "RMSE"], [fmt(b.r2, 2), "R²", "variance explained"]]
        .map(([v, u, l]) => `<div class="stat reveal"><span class="v">${v}</span> <span class="u">${u}</span><span class="l">${l}</span></div>`).join("");
      revealNew(row);
      const clim = m.metrics.climatology, v = $("s4-verdict");
      if (v && clim) { const impr = ((clim.mae - b.mae) / clim.mae) * 100;
        v.innerHTML = `Knowing only the calendar date gives ±${fmt(clim.mae, 2)} °C. Adding electricity
          behaviour cuts that to <strong>±${fmt(b.mae, 2)} °C</strong> — about
          <strong>${impr.toFixed(0)}% less error</strong>. The grid genuinely carries temperature
          information beyond the season, though demand alone (no calendar) stays a weaker guide than
          the date itself.`; }
    }
    if (p && p.date && p.rf_B) {
      plotLine("chart-ts", [
        { x: p.date, y: p.actual, name: "Observed", mode: "lines", line: { color: C.temp, width: 1.3 } },
        { x: p.date, y: p.rf_B, name: "Inferred", mode: "lines", line: { color: C.demand, width: 1.3 } },
      ], { xaxis: Object.assign({ type: "date" }, AXS),
           yaxis: Object.assign({ title: "daily mean temp (°C)" }, AXS), hovermode: "x unified" });
      const lim = [Math.min(...p.actual, ...p.rf_B), Math.max(...p.actual, ...p.rf_B)];
      plot("chart-scatter", [
        { x: p.actual, y: p.rf_B, mode: "markers", marker: { color: C.temp, size: 5, opacity: 0.4 } },
        { x: lim, y: lim, mode: "lines", line: { color: C.soft, width: 1, dash: "dash" } },
      ], { xaxis: Object.assign({ title: "observed (°C)" }, AX),
           yaxis: Object.assign({ title: "inferred (°C)" }, AX) });
    } else { hide($("chart-ts")); hide($("chart-scatter")); }
    if (m && m.metrics) {
      const order = [["climatology", "climatology"], ["rf_A", "electricity only"], ["rf_B", "elec + calendar"]];
      const rows = order.filter(([k]) => m.metrics[k]);
      plotBars("chart-compare", [{
        type: "bar", orientation: "h", y: rows.map(([, l]) => l), x: rows.map(([k]) => m.metrics[k].mae),
        marker: { color: rows.map(([k]) => k === "rf_B" ? C.demand : C.line),
                  line: { color: C.demand, width: rows.map(([k]) => k === "rf_B" ? 0 : 1) } },
        text: rows.map(([k]) => "±" + fmt(m.metrics[k].mae, 2)), textposition: "auto",
        textfont: { family: MONO, color: C.ink },
      }], { xaxis: Object.assign({ title: "mean abs error (°C)" }, AX),
            yaxis: Object.assign({ automargin: true }, AX), margin: { l: 120, r: 16, t: 10, b: 44 } });
    } else hide($("chart-compare"));
  }

  /* ---------- 05 ---------- */
  function coldSpell() {
    const c = D.cold_spell; if (!c) { const s = $("s5"); if (s) s.style.display = "none"; return; }
    const cm = c.confusion_matrix, el = $("confusion");
    if (el) { const cell = (n, lab, col, dark) => `<div class="cm-cell" style="background:${col};color:${dark ? "#0a0806" : "#f2e3bf"}"><span class="cm-num">${n}</span><span class="cm-lab">${lab}</span></div>`;
      el.innerHTML = `<div></div><div class="cm-axis">pred:<br>not cold</div><div class="cm-axis">pred:<br>cold</div>` +
        `<div class="cm-axis">actual:<br>not cold</div>` + cell(cm.tn, "correct", "#2c2114") + cell(cm.fp, "false alarm", "#2f8f84", true) +
        `<div class="cm-axis">actual:<br>cold</div>` + cell(cm.fn, "missed", "#5a3a10") + cell(cm.tp, "caught", C.temp, true); }
    if ($("cold-recall")) $("cold-recall").textContent = fmt(c.recall * 100, 0) + "%";
    if ($("cold-precision")) $("cold-precision").textContent = fmt(c.precision * 100, 0) + "%";
    if ($("cold-recall-txt")) $("cold-recall-txt").textContent =
      `Of the genuinely cold days (below ${fmt(c.threshold_c, 1)} °C), the grid flagged this share.`;
    if ($("cold-precision-txt")) $("cold-precision-txt").textContent =
      `When the grid called a day cold, this share truly were.`;
  }

  /* ---------- 06 ---------- */
  function gridLies() {
    const r = D.largest_residuals;
    if (r && r.largest_residuals) { const tb = document.querySelector("#resid-table tbody");
      if (tb) tb.innerHTML = r.largest_residuals.slice(0, 12).map((row) =>
        `<tr><td>${row.date}</td><td>${row.weekday}</td><td>${fmt(row.actual_c, 1)}°</td>
         <td>${fmt(row.predicted_c, 1)}°</td><td style="color:${Math.abs(row.residual_c) > 4 ? C.red : C.ink}">${row.residual_c > 0 ? "+" : ""}${fmt(row.residual_c, 1)}°</td>
         <td>${row.context}</td></tr>`).join(""); }
    else { const t = $("resid-table"); if (t) t.closest(".table-wrap").style.display = "none"; }
    const p = D.predictions;
    if (p && p.date && p.rf_B) {
      const byM = Array.from({ length: 12 }, () => []);
      p.date.forEach((d, i) => byM[new Date(d).getUTCMonth()].push(p.actual[i] - p.rf_B[i]));
      const M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
      const traces = byM.map((ys, i) => ({ y: ys, type: "box", name: M[i], marker: { color: C.temp },
        line: { color: C.temp }, fillcolor: "rgba(82,214,198,0.08)", boxpoints: false }));
      plot("chart-resmonth", traces, { yaxis: Object.assign({ title: "actual − inferred (°C)",
        zeroline: true, zerolinecolor: C.demand }, AX), xaxis: Object.assign({}, AX) });
    } else hide($("chart-resmonth"));
  }

  /* ---------- 07 ---------- */
  function conclusion() {
    const m = D.metrics;
    if (m && m.metrics && m.metrics.rf_B && m.metrics.climatology) {
      const b = m.metrics.rf_B.mae, cl = m.metrics.climatology.mae, impr = ((cl - b) / cl) * 100, el = $("conclusion");
      if (el) el.innerHTML = `Britain's electricity demand does carry a real temperature signal.
        Reading behaviour off the grid infers daily temperature to within
        <strong>±${fmt(b, 2)} °C</strong> (R² ${fmt(m.metrics.rf_B.r2, 2)}) on unseen years — about
        ${impr.toFixed(0)}% sharper than the calendar alone. But it is a supplement to knowing the
        season, not a replacement for it.`;
    }
  }

  /* ---------- scroll reveals + lazy per-module render ---------- */
  const SECTION_RENDER = { s1: heartbeat, s2: fourier, s3: features, s4: prediction,
                           s5: coldSpell, s6: gridLies, s7: conclusion };
  let revealObserver = null;

  function markReveals() {
    document.querySelectorAll(
      ".panel, .telemetry .tcard, .notes .wrap, .stat-row, .two-charts > *, .cold-grid, " +
      ".table-wrap, .limits, .section > .wrap"
    ).forEach((el) => el.classList.add("reveal"));
  }
  function revealNew(container) { // observe elements added after initial setup (e.g. stat cards)
    if (!revealObserver) return;
    container.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));
  }

  function init() {
    if (!window.GRID_DATA) { document.querySelectorAll("main .plot").forEach(hide);
      console.warn("GRID_DATA not found — run `python -m src.build_site`."); }
    heroScope();
    markReveals();

    if (!("IntersectionObserver" in window)) {
      // no observer: render everything and reveal immediately
      document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in-view"));
      Object.values(SECTION_RENDER).forEach((fn) => { try { fn(); } catch (e) { console.error(fn.name, e); } });
      return;
    }
    revealObserver = new IntersectionObserver((ents, obs) => {
      ents.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in-view"); obs.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

    const sectionObserver = new IntersectionObserver((ents, obs) => {
      ents.forEach((e) => { if (e.isIntersecting) { const fn = SECTION_RENDER[e.target.id];
        if (fn) { try { fn(); } catch (err) { console.error(e.target.id, err); } }
        obs.unobserve(e.target); } });
    }, { threshold: 0.04, rootMargin: "0px 0px -8% 0px" });
    Object.keys(SECTION_RENDER).forEach((id) => { const s = $(id); if (s) sectionObserver.observe(s); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
