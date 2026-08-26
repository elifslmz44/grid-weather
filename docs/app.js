/* Renders the readout from window.GRID_DATA (bundled by src/build_site.py).
   Colours read live from the active theme (light/dark toggle re-themes charts).
   Line charts use a "scrub reveal": a faint full line fixes the axis; a bright line draws in
   when scrolled into view, then follows the cursor as you hover. Reduced-motion → no draw-in. */

(function () {
  "use strict";
  const D = window.GRID_DATA || {};
  const MONO = "IBM Plex Mono, monospace";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const CFG = { displayModeBar: false, responsive: true };
  let instant = false, started = false;

  const PAL = {
    dark:  { panel: "#100c08", amber: "#ffb000", temp: "#52d6c6", ink: "#f2e3bf",
             soft: "#a4906a", grid: "#241a0d", line: "#2c2114", red: "#ff5a4d", beam: "#fff5dd" },
    light: { panel: "#efece1", amber: "#b25400", temp: "#1f6f8b", ink: "#1c1712",
             soft: "#6d6555", grid: "#e6e0d1", line: "#d5cebd", red: "#c0392b", beam: "#b25400" },
  };
  const themeName = () => (document.documentElement.dataset.theme === "light" ? "light" : "dark");
  const P = () => PAL[themeName()];
  const $ = (id) => document.getElementById(id);
  const hide = (el) => { const f = el && el.closest(".panel, .telemetry, section"); if (f) f.style.display = "none"; };
  const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));

  function AX(extra) { const p = P(); return Object.assign({
    gridcolor: p.grid, zerolinecolor: p.line, linecolor: p.line, ticks: "outside", tickcolor: p.line,
    tickfont: { family: MONO, size: 11, color: p.soft }, titlefont: { family: MONO, size: 12, color: p.soft },
  }, extra || {}); }
  function AXS(extra) { const p = P(); return AX(Object.assign({ showspikes: true, spikecolor: p.amber,
    spikethickness: 1, spikedash: "dot", spikemode: "across", spikesnap: "cursor" }, extra || {})); }
  function layoutBase(over) { const p = P(); return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: MONO, size: 12, color: p.soft }, margin: { l: 62, r: 20, t: 16, b: 46 }, showlegend: false,
    xaxis: AX(), yaxis: AX(),
    hoverlabel: { font: { family: MONO, color: p.ink, size: 12.5 }, bgcolor: p.panel, bordercolor: p.amber },
  }, over || {}); }

  function plot(id, traces, layout) { const el = $(id); if (!el) return;
    try { Plotly.newPlot(el, traces, layoutBase(layout), CFG); } catch (e) { console.error(id, e); hide(el); } }

  function plotBars(id, traces, layout) { const el = $(id); if (!el) return;
    try {
      if (reduce || instant) { Plotly.newPlot(el, traces, layoutBase(layout), CFG); return; }
      const realX = traces.map((t) => Array.isArray(t.x) ? t.x.slice() : null);
      traces.forEach((t, i) => { if (realX[i]) t.x = realX[i].map(() => 0); });
      Plotly.newPlot(el, traces, layoutBase(layout), CFG).then(() => requestAnimationFrame(() =>
        Plotly.animate(el, { data: traces.map((t, i) => realX[i] ? { x: realX[i] } : {}) },
          { transition: { duration: 900, easing: "cubic-out" }, frame: { duration: 900 } })));
    } catch (e) { console.error(id, e); try { Plotly.newPlot(el, traces, layoutBase(layout), CFG); } catch (_) {} } }

  /* Scrub-reveal line chart.
     series: [{x, y, color, width, name, hovertemplate, fill, fillcolor}]
     A faint full copy of each series fixes the axis + carries hover; a bright copy is revealed
     from the left — animated once on load, then driven by the cursor on hover. */
  function plotScrub(id, series, layout) {
    const el = $(id); if (!el || !series.length) return;
    try {
      const p = P();
      const baseTr = series.map((s) => ({ x: s.x, y: s.y, mode: "lines",
        line: { color: s.color, width: (s.width || 2) * 0.5 }, opacity: 0.22,
        name: s.name, hovertemplate: s.hovertemplate, showlegend: false }));
      const hiTr = series.map((s) => ({ x: [], y: [], mode: "lines",
        line: { color: s.color, width: s.width || 2 }, fill: s.fill, fillcolor: s.fillcolor,
        hoverinfo: "skip", showlegend: false }));
      const traces = baseTr.concat(hiTr);
      const hiIdx = series.map((_, i) => series.length + i);
      const N = series[0].x.length;
      // build the legend from the real series, so labels + colours always match the curves
      const panel = el.closest(".panel-screen");
      const leg = panel && panel.querySelector(".chlegend");
      if (leg) leg.innerHTML = series.map((s) => `<span style="color:${s.color}">▮ ${s.name}</span>`).join(" ");
      const lay = layoutBase(Object.assign({ hovermode: "x unified" }, layout || {}));
      const setK = (k) => { try { Plotly.restyle(el,
        { x: series.map((s) => s.x.slice(0, k)), y: series.map((s) => s.y.slice(0, k)) }, hiIdx); } catch (e) {} };
      let animId = 0;
      const animateTo = (target, dur) => { const my = ++animId; let t0 = null;
        const step = (ts) => { if (my !== animId) return; if (t0 == null) t0 = ts; const pr = Math.min((ts - t0) / dur, 1);
          setK(Math.max(1, Math.round(pr * target))); if (pr < 1) requestAnimationFrame(step); };
        requestAnimationFrame(step); };
      Plotly.newPlot(el, traces, lay, CFG).then(() => {
        if (reduce || instant) { setK(N); return; }
        animateTo(N, 1100); // draw in once on load
        // follow the cursor: reveal exactly up to the hovered point, cancelling any running animation
        if (el.on) el.on("plotly_hover", (ev) => { const pt = ev.points && ev.points[0]; if (!pt) return;
          const idx = pt.pointIndex != null ? pt.pointIndex : pt.pointNumber; if (idx == null) return;
          ++animId; setK(idx + 1); });
        // restore the full line only when the cursor truly leaves the chart (not between points)
        el.addEventListener("mouseleave", () => animateTo(N, 450));
      });
    } catch (e) { console.error(id, e);
      plot(id, series.map((s) => ({ x: s.x, y: s.y, mode: "lines", line: { color: s.color, width: s.width } })), layout); }
  }

  /* ---------- hero scope (two channels) ---------- */
  function heroScope() {
    const svg = $("hero-pulse"); if (!svg) return;
    svg.innerHTML = "";
    const p = P(), W = 1000, H = 230, mid = H / 2, ns = "http://www.w3.org/2000/svg";
    const hod = D.hour_of_day || {};
    const hasTemp = Array.isArray(hod.temp_by_hour_c) && hod.temp_by_hour_c.length > 1;
    const tag = document.querySelector(".scope-tag");
    if (tag) tag.innerHTML = hasTemp
      ? '<span class="ch1">CH1 ▮ DEMAND</span> &nbsp; <span class="ch2">CH2 ▮ TEMPERATURE</span>'
      : '<span class="ch1">CH1 ▮ DEMAND</span>';
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const defs = document.createElementNS(ns, "defs");
    defs.innerHTML = '<filter id="g"><feGaussianBlur stdDeviation="3"/></filter>'; svg.appendChild(defs);
    for (let gx = 0; gx <= W; gx += 50) addLine(gx, 0, gx, H, p.grid, 1);
    for (let gy = 0; gy <= H; gy += 46) addLine(0, gy, W, gy, p.grid, 1);
    addLine(0, mid, W, mid, p.line, 1);
    const demandVals = hod.all_year_mw ||
      Array.from({ length: 48 }, (_, i) => 30000 + 8000 * Math.sin((i / 48) * 2 * Math.PI - 1.2));
    const toPts = (arr, reps, band) => { const n = arr.length * reps, mn = Math.min(...arr), mx = Math.max(...arr),
      span = (mx - mn) || 1, pts = [];
      for (let i = 0; i < n; i++) { const v = arr[i % arr.length];
        pts.push([(i / (n - 1)) * W, mid + (0.5 - (v - mn) / span) * (H * band)]); } return pts; };
    let tempTrace = null;
    if (hasTemp) { const tp = toPts(hod.temp_by_hour_c, 2, 0.58);
      const td = "M " + tp.map((q) => `${q[0].toFixed(1)},${q[1].toFixed(1)}`).join(" L ");
      addPath(td, p.temp, 5, 0.14); tempTrace = addPath(td, p.temp, 1.6, 0.9); }
    const dp = toPts(demandVals, 2, 0.72);
    const dd = "M " + dp.map((q) => `${q[0].toFixed(1)},${q[1].toFixed(1)}`).join(" L ");
    addPath(dd, p.amber, 6, 0.18); const trace = addPath(dd, p.amber, 2, 1);
    const beam = document.createElementNS(ns, "circle"); beam.setAttribute("r", "4");
    beam.setAttribute("fill", p.beam); beam.setAttribute("filter", "url(#g)"); svg.appendChild(beam);
    const drawIn = (path, dur) => { if (!path || !path.getTotalLength) return;
      const len = path.getTotalLength(); path.style.strokeDasharray = len; path.style.strokeDashoffset = len;
      path.animate([{ strokeDashoffset: len }, { strokeDashoffset: 0 }], { duration: dur, easing: "linear", fill: "forwards" }); };
    if (!reduce && !instant && trace.getTotalLength) {
      drawIn(trace, 3000); drawIn(tempTrace, 3000); // both channels sweep in together
      const len = trace.getTotalLength();
      let t0 = null; const step = (ts) => { if (!t0) t0 = ts; const pr = Math.min((ts - t0) / 3000, 1);
        const pt = trace.getPointAtLength(len * pr); beam.setAttribute("cx", pt.x); beam.setAttribute("cy", pt.y);
        if (pr < 1) requestAnimationFrame(step); else beam.style.opacity = 0.6; }; requestAnimationFrame(step);
    } else { const pt = dp[dp.length - 1]; beam.setAttribute("cx", pt[0]); beam.setAttribute("cy", pt[1]); }
    function addLine(x1, y1, x2, y2, col, w) { const l = document.createElementNS(ns, "line");
      l.setAttribute("x1", x1); l.setAttribute("y1", y1); l.setAttribute("x2", x2); l.setAttribute("y2", y2);
      l.setAttribute("stroke", col); l.setAttribute("stroke-width", w); svg.appendChild(l); }
    function addPath(dpath, col, w, op) { const q = document.createElementNS(ns, "path"); q.setAttribute("d", dpath);
      q.setAttribute("fill", "none"); q.setAttribute("stroke", col); q.setAttribute("stroke-width", w);
      q.setAttribute("opacity", op); q.setAttribute("stroke-linejoin", "round"); svg.appendChild(q); return q; }
  }

  /* ---------- 01 ---------- */
  function heartbeat() { const p = P(), h = D.hour_of_day;
    if (h) plotScrub("chart-hod", [
      { x: h.hour, y: h.winter_mw, color: p.amber, width: 2, name: "winter",
        hovertemplate: "winter · %{x}:00 · %{y:,.0f} MW<extra></extra>" },
      { x: h.hour, y: h.all_year_mw, color: p.ink, width: 2.4, name: "all year",
        hovertemplate: "all year · %{x}:00 · %{y:,.0f} MW<extra></extra>" },
      { x: h.hour, y: h.summer_mw, color: p.temp, width: 2, name: "summer",
        hovertemplate: "summer · %{x}:00 · %{y:,.0f} MW<extra></extra>" },
    ], { xaxis: AXS({ title: "hour of day (local)", dtick: 3 }), yaxis: AX({ title: "mean demand (MW)" }) });
    else hide($("chart-hod"));
    const w = D.weekday_weekend;
    if (w) plotScrub("chart-ww", [
      { x: w.hour, y: w.weekday_mw, color: p.amber, width: 2.4, name: "weekday",
        hovertemplate: "weekday · %{x}:00 · %{y:,.0f} MW<extra></extra>" },
      { x: w.hour, y: w.weekend_mw, color: p.temp, width: 2.4, name: "weekend",
        hovertemplate: "weekend · %{x}:00 · %{y:,.0f} MW<extra></extra>" },
    ], { xaxis: AXS({ title: "hour of day (local)", dtick: 3 }), yaxis: AX({ title: "mean demand (MW)" }) });
    else hide($("chart-ww"));
    renderHeatmap();
  }

  function renderHeatmap() { const p = P(), h = D.hour_month_heatmap; if (!h) { hide($("chart-heatmap")); return; }
    const M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const months = h.months.map((mo) => M[(mo - 1) % 12] || mo);
    plot("chart-heatmap", [{ type: "heatmap", z: h.z_mw, x: months, y: h.hours,
      colorscale: [[0, p.panel], [1, p.amber]],
      colorbar: { tickfont: { family: MONO, color: p.soft, size: 10 }, outlinecolor: p.line, thickness: 10 },
      hovertemplate: "%{x} · %{y}:00 · %{z:,.0f} MW<extra></extra>" }],
      { xaxis: AX({}), yaxis: AX({ title: "hour of day", dtick: 3 }), margin: { l: 56, r: 20, t: 16, b: 40 } });
  }

  function responseCurve() { const p = P(), r = D.temperature_response; if (!r) { hide($("chart-response")); return; }
    plot("chart-response", [
      { x: r.scatter_temp_c, y: r.scatter_demand_mw, mode: "markers",
        marker: { color: p.soft, size: 4, opacity: 0.35 }, name: "days",
        hovertemplate: "%{x:.1f} °C · %{y:,.0f} MW<extra></extra>" },
      { x: r.fit_temp_c, y: r.fit_demand_mw, mode: "lines", line: { color: p.amber, width: 3 }, name: "fit",
        hovertemplate: "%{x:.1f} °C · %{y:,.0f} MW<extra>fitted</extra>" },
    ], { xaxis: AX({ title: "daily mean temperature (°C)" }), yaxis: AX({ title: "daily mean demand (MW)" }),
         shapes: [{ type: "line", x0: r.knot_temp_c, x1: r.knot_temp_c, yref: "paper", y0: 0, y1: 1,
                    line: { color: p.temp, width: 1, dash: "dot" } }],
         annotations: [{ x: r.knot_temp_c, y: 1.03, yref: "paper", text: "breakpoint " + r.knot_temp_c + "°C",
                    showarrow: false, font: { color: p.temp, size: 11, family: MONO } }] });
    const row = $("response-stats");
    if (row) { const mwCold = Math.abs(r.mw_per_degree_colder);
      row.innerHTML = [["", r.knot_temp_c, 0, "°C", "balance point"],
        ["", mwCold, 0, "MW", "per °C colder"], ["", r.asymmetry_ratio, 1, "×", "heating vs cooling"]]
        .map(([pre, val, dec, u, l]) => `<div class="stat reveal"><span class="v" data-t="${val}" data-d="${dec}" data-p="${pre}">${pre}0</span> <span class="u">${u}</span><span class="l">${l}</span></div>`).join("");
      row.querySelectorAll(".v").forEach((el) => countUp(el, +el.dataset.t, +el.dataset.d, el.dataset.p, ""));
      revealNew(row); }
    const v = $("response-verdict");
    if (v) { const mwCold = Math.abs(r.mw_per_degree_colder);
      v.innerHTML = `Below about <strong>${r.knot_temp_c} °C</strong>, each degree colder adds roughly
        <strong>${mwCold.toLocaleString()} MW</strong> of demand — around <strong>${r.asymmetry_ratio}×</strong>
        the grid's response to the same rise in heat. That asymmetry is exactly why the reconstruction
        reads cold snaps far more sharply than warm spells.`; }
  }

  /* ---------- 02 ---------- */
  function fourier() { const p = P(), f = D.fourier_spectrum; if (!f) { hide($("chart-fft")); return; }
    const refs = [[24, "day"], [12, "½ day"], [168, "week"], [8766, "year"]];
    const shapes = refs.map(([q]) => ({ type: "line", x0: Math.log10(q), x1: Math.log10(q), yref: "paper",
      y0: 0, y1: 1, line: { color: p.temp, width: 1, dash: "dot" } }));
    const anns = refs.map(([q, t]) => ({ x: Math.log10(q), y: 1.04, yref: "paper", text: t, showarrow: false,
      font: { color: p.temp, size: 11, family: MONO } }));
    plotScrub("chart-fft", [{ x: f.period_hours, y: f.relative_power, color: p.amber, width: 1.4,
      fill: "tozeroy", fillcolor: "rgba(255,176,0,0.07)", name: "power",
      hovertemplate: "period %{x:.0f} h · power %{y:.3f}<extra></extra>" }],
      { xaxis: AXS({ title: "period (hours)", type: "log" }), yaxis: AX({ title: "relative power", rangemode: "tozero" }),
        shapes, annotations: anns });
  }

  /* ---------- 03 ---------- */
  function features() { const m = D.metrics; if (!m) return;
    const fill = (id, arr) => { const ul = $(id); if (ul && arr) ul.innerHTML = arr.map((f) => `<li>▸ ${f}</li>`).join(""); };
    fill("feats-a", m.features_A);
    if (m.features_A && m.features_B) fill("feats-b", m.features_B.filter((f) => !m.features_A.includes(f)));
  }

  /* ---------- 04 ---------- */
  function prediction() { const p = P(), m = D.metrics, pr = D.predictions;
    if (m && m.metrics && m.metrics.rf_B) {
      const row = $("stat-row"), v = $("s4-verdict"), clim = m.metrics.climatology;
      const verdicts = {
        climatology: (b) => `Using only the calendar date — the average temperature for that day of the year —
          the error is <strong>±${fmt(b.mae, 2)} °C</strong>. This is the baseline every model must beat.`,
        rf_A: (b) => `From electricity behaviour <em>alone</em>, with no calendar at all, the grid infers
          temperature to <strong>±${fmt(b.mae, 2)} °C</strong>${clim ? ` — actually worse than the ±${fmt(clim.mae, 2)} °C
          you get from the date, because the seasonal cycle dominates` : ""}.`,
        rf_B: (b) => `Electricity behaviour <em>plus</em> the calendar reaches <strong>±${fmt(b.mae, 2)} °C</strong>${clim ?
          ` — about <strong>${(((clim.mae - b.mae) / clim.mae) * 100).toFixed(0)}% less error</strong> than the date alone` : ""}.
          So the grid carries real temperature information beyond the season.`,
      };
      const showModel = (key) => { const b = m.metrics[key]; if (!b || !row) return;
        row.innerHTML = [["±", b.mae, 2, "°C", "mean error"], ["", b.rmse, 2, "°C", "RMSE"],
          ["", b.r2, 2, "R²", "variance explained"]]
          .map(([pre, val, dec, u, l]) => `<div class="stat reveal"><span class="v" data-t="${val}" data-d="${dec}" data-p="${pre}">${pre}0</span> <span class="u">${u}</span><span class="l">${l}</span></div>`).join("");
        row.querySelectorAll(".v").forEach((el) => countUp(el, +el.dataset.t, +el.dataset.d, el.dataset.p, ""));
        revealNew(row);
        if (v && verdicts[key]) v.innerHTML = verdicts[key](b);
        const tog = $("model-toggle");
        if (tog) tog.querySelectorAll("button").forEach((btn) => btn.classList.toggle("on", btn.dataset.key === key));
      };
      // build the toggle (only for models that exist in the data)
      const opts = [["climatology", "calendar only"], ["rf_A", "electricity only"], ["rf_B", "electricity + calendar"]]
        .filter(([k]) => m.metrics[k]);
      const tog = $("model-toggle");
      if (tog && opts.length > 1) { tog.innerHTML = opts.map(([k, lbl]) =>
        `<button type="button" data-key="${k}">${lbl}</button>`).join("");
        tog.querySelectorAll("button").forEach((btn) => btn.addEventListener("click", () => showModel(btn.dataset.key))); }
      showModel("rf_B");
    }
    if (pr && pr.date && pr.rf_B) {
      plotScrub("chart-ts", [
        { x: pr.date, y: pr.actual, color: p.temp, width: 1.4, name: "observed",
          hovertemplate: "observed · %{x|%d %b %Y} · %{y:.1f} °C<extra></extra>" },
        { x: pr.date, y: pr.rf_B, color: p.amber, width: 1.4, name: "inferred",
          hovertemplate: "inferred · %{x|%d %b %Y} · %{y:.1f} °C<extra></extra>" },
      ], { xaxis: AXS({ type: "date", rangeselector: {
             buttons: [{ count: 3, label: "3M", step: "month", stepmode: "backward" },
                       { count: 6, label: "6M", step: "month", stepmode: "backward" },
                       { count: 1, label: "1Y", step: "year", stepmode: "backward" },
                       { step: "all", label: "All" }],
             bgcolor: p.panel, activecolor: p.amber, bordercolor: p.line, borderwidth: 1,
             font: { family: MONO, color: p.ink, size: 11 }, x: 0, y: 1.12 } }),
           yaxis: AXS({ title: "daily mean temp (°C)" }) });
      const lim = [Math.min(...pr.actual, ...pr.rf_B), Math.max(...pr.actual, ...pr.rf_B)];
      plot("chart-scatter", [
        { x: pr.actual, y: pr.rf_B, mode: "markers", marker: { color: p.temp, size: 5, opacity: 0.4 },
          hovertemplate: "obs %{x:.1f} → inf %{y:.1f} °C<extra></extra>" },
        { x: lim, y: lim, mode: "lines", line: { color: p.soft, width: 1, dash: "dash" }, hoverinfo: "skip" },
      ], { xaxis: AX({ title: "observed (°C)" }), yaxis: AX({ title: "inferred (°C)" }) });
    } else { hide($("chart-ts")); hide($("chart-scatter")); }
    if (m && m.metrics) {
      const order = [["climatology", "climatology"], ["rf_A", "electricity only"], ["rf_B", "elec + calendar"]];
      const rows = order.filter(([k]) => m.metrics[k]);
      const maxMae = Math.max.apply(null, rows.map(([k]) => m.metrics[k].mae));
      plotBars("chart-compare", [{ type: "bar", orientation: "h", y: rows.map(([, l]) => l),
        x: rows.map(([k]) => m.metrics[k].mae),
        marker: { color: rows.map(([k]) => k === "rf_B" ? p.amber : p.line),
                  line: { color: p.amber, width: rows.map(([k]) => k === "rf_B" ? 0 : 1) } },
        text: rows.map(([k]) => "±" + fmt(m.metrics[k].mae, 2)), textposition: "auto",
        textfont: { family: MONO, color: p.ink },
        hovertemplate: "%{y}<br>±%{x:.2f} °C mean error<extra></extra>" }],
        { xaxis: AX({ title: "mean abs error (°C)", range: [0, maxMae * 1.18], zeroline: false }),
          yaxis: AX({ automargin: true }), margin: { l: 120, r: 16, t: 10, b: 44 } });
    } else hide($("chart-compare"));
  }

  /* ---------- 05 ---------- */
  function coldSpell() { const p = P(), c = D.cold_spell; if (!c) { const s = $("s5"); if (s) s.style.display = "none"; return; }
    const cm = c.confusion_matrix, el = $("confusion");
    if (el) { const cell = (n, lab, col, dark) => `<div class="cm-cell" style="background:${col};color:${dark ? "#0a0806" : p.ink}"><span class="cm-num">${n}</span><span class="cm-lab">${lab}</span></div>`;
      el.innerHTML = `<div></div><div class="cm-axis">pred:<br>not cold</div><div class="cm-axis">pred:<br>cold</div>` +
        `<div class="cm-axis">actual:<br>not cold</div>` + cell(cm.tn, "correct", p.line) + cell(cm.fp, "false alarm", "#2f8f84", true) +
        `<div class="cm-axis">actual:<br>cold</div>` + cell(cm.fn, "missed", "#5a3a10") + cell(cm.tp, "caught", p.temp, true); }
    if ($("cold-recall")) countUp($("cold-recall"), c.recall * 100, 0, "", "%");
    if ($("cold-precision")) countUp($("cold-precision"), c.precision * 100, 0, "", "%");
    if ($("cold-recall-txt")) $("cold-recall-txt").textContent = `Of the genuinely cold days (below ${fmt(c.threshold_c, 1)} °C), the grid flagged this share.`;
    if ($("cold-precision-txt")) $("cold-precision-txt").textContent = `When the grid called a day cold, this share truly were.`;
  }

  /* ---------- 06 ---------- */
  function gridLies() { const p = P(), r = D.largest_residuals;
    if (r && r.largest_residuals) { const tb = document.querySelector("#resid-table tbody");
      if (tb) tb.innerHTML = r.largest_residuals.slice(0, 12).map((row) =>
        `<tr><td>${row.date}</td><td>${row.weekday}</td><td>${fmt(row.actual_c, 1)}°</td>
         <td>${fmt(row.predicted_c, 1)}°</td><td style="color:${Math.abs(row.residual_c) > 4 ? p.red : p.ink}">${row.residual_c > 0 ? "+" : ""}${fmt(row.residual_c, 1)}°</td>
         <td>${row.context}</td></tr>`).join(""); }
    else { const t = $("resid-table"); if (t) t.closest(".table-wrap").style.display = "none"; }
    const pr = D.predictions;
    if (pr && pr.date && pr.rf_B) {
      const byM = Array.from({ length: 12 }, () => []);
      pr.date.forEach((d, i) => byM[new Date(d).getUTCMonth()].push(pr.actual[i] - pr.rf_B[i]));
      const M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
      const traces = byM.map((ys, i) => ({ y: ys, type: "box", name: M[i], marker: { color: p.temp },
        line: { color: p.temp }, fillcolor: "rgba(82,214,198,0.08)", boxpoints: false, hoverinfo: "y" }));
      plot("chart-resmonth", traces, { yaxis: AX({ title: "actual − inferred (°C)", zeroline: true, zerolinecolor: p.amber }), xaxis: AX() });
    } else hide($("chart-resmonth"));
  }

  /* ---------- 07 ---------- */
  function conclusion() { const m = D.metrics;
    if (m && m.metrics && m.metrics.rf_B && m.metrics.climatology) {
      const b = m.metrics.rf_B.mae, cl = m.metrics.climatology.mae, impr = ((cl - b) / cl) * 100, el = $("conclusion");
      if (el) el.innerHTML = `Britain's electricity demand does carry a real temperature signal.
        Reading behaviour off the grid infers daily temperature to within
        <strong>±${fmt(b, 2)} °C</strong> (R² ${fmt(m.metrics.rf_B.r2, 2)}) on unseen years — about
        ${impr.toFixed(0)}% sharper than the calendar alone. But it is a supplement to knowing the
        season, not a replacement for it.`;
    }
  }

  /* ---------- reveals, lazy render, theme ---------- */
  const SECTION_RENDER = { s1: heartbeat, s2: fourier, s2b: responseCurve, s3: features, s4: prediction, s5: coldSpell, s6: gridLies, s7: conclusion };
  const rendered = new Set();
  let revealObserver = null;

  function markReveals() { document.querySelectorAll(
    ".panel, .telemetry .tcard, .notes .wrap, .stat-row, .two-charts > *, .cold-grid, .table-wrap, .limits, .section > .wrap"
  ).forEach((el) => el.classList.add("reveal")); }
  function revealNew(container) { if (!revealObserver) return;
    container.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el)); }
  function renderSection(id) { const fn = SECTION_RENDER[id]; if (!fn) return;
    try { fn(); rendered.add(id); } catch (e) { console.error(id, e); } }

  function applyTheme(t) { document.documentElement.dataset.theme = t;
    const btn = $("theme-toggle"); if (btn) btn.textContent = t === "light" ? "☾ DARK" : "☀ LIGHT";
    try { localStorage.setItem("gw-theme", t); } catch (e) {} }
  function retheme() { instant = true; heroScope();
    rendered.forEach((id) => { try { SECTION_RENDER[id](); } catch (e) { console.error(id, e); } }); instant = false; }

  /* ---------- interactive extras ---------- */
  // animate a number counting up to its target (respects reduced-motion)
  function countUp(el, target, decimals, prefix, suffix) {
    prefix = prefix || ""; suffix = suffix || "";
    const done = () => { el.textContent = prefix + Number(target).toFixed(decimals) + suffix; };
    if (reduce || instant || !isFinite(target)) return done();
    const dur = 900, t0 = (window.performance && performance.now) ? performance.now() : Date.now();
    const tick = (now) => { const p = Math.min(((now || Date.now()) - t0) / dur, 1);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = prefix + (target * e).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(tick); else done(); };
    requestAnimationFrame(tick);
  }
  // status-bar grid frequency, gently wandering around 50 Hz like the real thing
  function liveFrequency() {
    const el = document.querySelector(".statusbar .live"); if (!el) return;
    if (reduce) { el.textContent = "50.00 Hz"; return; }
    let f = 50.0;
    setInterval(() => { f += (Math.random() - 0.5) * 0.03 + (50 - f) * 0.1;
      f = Math.max(49.95, Math.min(50.05, f)); el.textContent = f.toFixed(2) + " Hz"; }, 1500);
  }
  // thin progress bar tracking scroll position
  function scrollProgress() {
    const bar = $("scroll-progress"); if (!bar) return;
    const upd = () => { const h = document.documentElement, max = h.scrollHeight - h.clientHeight;
      bar.style.width = (max > 0 ? (h.scrollTop || document.body.scrollTop) / max * 100 : 0) + "%"; };
    window.addEventListener("scroll", upd, { passive: true }); upd();
  }
  // right-edge module navigator: click to jump, highlights the section you're in
  function buildNav() {
    const ids = ["s1", "s2", "s2b", "s3", "s4", "s5", "s6", "s7"];
    const labels = ["Heartbeat", "Frequencies", "Response", "Experiment", "Inference", "Cold snap", "Failures", "Conclusion"];
    if (!ids.some((id) => $(id))) return;
    const nav = document.createElement("nav"); nav.id = "modnav";
    ids.forEach((id, i) => { const s = $(id); if (!s) return;
      const a = document.createElement("a"); a.href = "#" + id; a.className = "dot"; a.dataset.id = id;
      a.setAttribute("aria-label", labels[i]);
      a.innerHTML = `<span class="dot-label">${String(i + 1).padStart(2, "0")} · ${labels[i]}</span>`;
      a.addEventListener("click", (e) => { e.preventDefault();
        s.scrollIntoView({ behavior: reduce ? "auto" : "smooth" }); });
      nav.appendChild(a); });
    document.body.appendChild(nav);
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((ents) => { ents.forEach((e) => { if (e.isIntersecting)
        nav.querySelectorAll(".dot").forEach((d) => d.classList.toggle("active", d.dataset.id === e.target.id)); }); },
        { rootMargin: "-45% 0px -45% 0px" });
      ids.forEach((id) => { const s = $(id); if (s) io.observe(s); });
    }
  }

  function trendCard() { const t = D.annual_and_trend, card = $("trend-card"), el = $("trend-pct");
    if (!card || !el || !t || t.trend_pct_per_year == null) return;
    el.textContent = Math.abs(t.trend_pct_per_year).toFixed(1) + "%/yr";
    card.style.display = "";
  }

  function init() {
    if (started) return; started = true;
    let t; try { t = localStorage.getItem("gw-theme"); } catch (e) {}
    if (t !== "light" && t !== "dark") t = "dark"; // dark by default; a saved choice still wins
    applyTheme(t);
    const btn = $("theme-toggle");
    if (btn) btn.addEventListener("click", () => { applyTheme(themeName() === "light" ? "dark" : "light"); retheme(); });

    if (!window.GRID_DATA) { document.querySelectorAll("main .plot").forEach(hide);
      console.warn("GRID_DATA not found — run `python -m src.build_site`."); }
    heroScope();
    markReveals();
    liveFrequency();
    scrollProgress();
    buildNav();
    trendCard();

    if (!("IntersectionObserver" in window)) {
      document.querySelectorAll(".reveal").forEach((el) => el.classList.add("in-view"));
      Object.keys(SECTION_RENDER).forEach(renderSection); return;
    }
    revealObserver = new IntersectionObserver((ents, obs) => { ents.forEach((e) => {
      if (e.isIntersecting) { e.target.classList.add("in-view"); obs.unobserve(e.target); } });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    document.querySelectorAll(".reveal").forEach((el) => revealObserver.observe(el));

    const sectionObserver = new IntersectionObserver((ents, obs) => { ents.forEach((e) => {
      if (e.isIntersecting) { renderSection(e.target.id); obs.unobserve(e.target); } });
    }, { threshold: 0.04, rootMargin: "0px 0px -8% 0px" });
    Object.keys(SECTION_RENDER).forEach((id) => { const s = $(id); if (s) sectionObserver.observe(s); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
