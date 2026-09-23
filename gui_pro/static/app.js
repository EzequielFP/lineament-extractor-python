"use strict";
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const MODEL_COLORS = { consenso: "#f43f5e", valles: "#60a5fa", crestas: "#4ade80", escarpes: "#fb923c",
  bordes: "#c084fc", pci_line: "#22d3ee", pci_multi: "#eab308" };
const PALETTE = ["#f43f5e", "#22c55e", "#3b82f6", "#f59e0b", "#a855f7", "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#14b8a6"];
const CONF_COLORS = { Alta: "#ef4444", Media: "#f59e0b", Baja: "#60a5fa" };
const TYPE_COLORS = { valle: "#60a5fa", cresta: "#4ade80", escarpe: "#fb923c", borde: "#c084fc", borde_sombreado: "#22d3ee" };

let DEF = null, logNext = 0, polling = null, result = null;

// ---------------------------------------------------------------- utilidades
function toast(msg, ok = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast" + (ok ? " ok" : ""); t.hidden = false;
  clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), ok ? 3500 : 7000);
}
async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch (e) { /* sin cuerpo */ }
    throw new Error(m);
  }
  return r.json();
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v) => typeof v === "number" ? (Number.isInteger(v) ? v.toLocaleString("es") : v.toLocaleString("es", { maximumFractionDigits: 3 })) : esc(v ?? "");

function table(rows, cols) {
  if (!rows || !rows.length) return "<p class='hint'>Sin datos.</p>";
  cols = cols || Object.keys(rows[0]);
  return `<table class="t"><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${cols.map((c) => `<td>${fmt(r[c])}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
}

// ---------------------------------------------------------------- formulario
const NUM_FIELDS = ["sensitivity", "min_length_m", "work_res_m", "line_length_m", "long_factor", "link_gap_m",
  "link_angle_deg", "link_lateral_m", "fit_tol_m", "angle_tol_deg", "n_orient", "min_support", "max_families",
  "drainage_area_m2", "density_cell_m", "density_radius_m", "RADI", "GTHR", "LTHR", "FTHR", "ATHR", "DTHR",
  "pci_azimuth", "pci_altitude", "val_buffer_m", "val_angle_deg"];
const TXT_FIELDS = ["dem_path", "hillshade_path", "reference_path", "field_path", "zona", "output_dir"];
const BOOL_FIELDS = ["drainage", "save_shapefiles", "save_evidence_rasters"];

function setForm(p, keepFiles = false) {
  if (!keepFiles) TXT_FIELDS.forEach((k) => { $("#" + k).value = p[k] || ""; });
  NUM_FIELDS.forEach((k) => { if (p[k] !== undefined) $("#" + k).value = p[k]; });
  BOOL_FIELDS.forEach((k) => { $("#" + k).checked = !!p[k]; });
  $("#scales_m").value = (p.scales_m || []).join(", ");
  $$("[data-w]").forEach((el) => { el.value = (p.consensus_weights || {})[el.dataset.w] ?? 1; });
  $$("#models input").forEach((el) => { el.checked = (p.models || []).includes(el.value); });
  $("#sensitivity_v").textContent = Number($("#sensitivity").value).toFixed(2);
  window._base = p;
}
function getForm() {
  const p = Object.assign({}, window._base || DEF.params);
  TXT_FIELDS.forEach((k) => { p[k] = $("#" + k).value.trim().replace(/^"|"$/g, ""); });
  NUM_FIELDS.forEach((k) => { const v = parseFloat($("#" + k).value); if (!Number.isNaN(v)) p[k] = v; });
  ["n_orient", "max_families", "RADI", "LTHR"].forEach((k) => { p[k] = Math.round(p[k]); });
  BOOL_FIELDS.forEach((k) => { p[k] = $("#" + k).checked; });
  p.scales_m = $("#scales_m").value.split(/[,; ]+/).map(parseFloat).filter((x) => x > 0);
  p.consensus_weights = {};
  $$("[data-w]").forEach((el) => { p.consensus_weights[el.dataset.w] = parseFloat(el.value) || 0; });
  p.models = $$("#models input:checked").map((el) => el.value);
  return p;
}

async function pick(kind, save = false) {
  const w = window.pywebview && window.pywebview.api;
  if (w) return kind === "folder" ? w.pick_folder() : w.pick_file(kind, save);
  return prompt(kind === "folder" ? "Ruta de la carpeta:" : "Ruta del archivo:") || "";
}

async function init() {
  DEF = await api("/api/defaults");
  $("#version").textContent = "v" + DEF.version + " · motor multi-evidencia";
  const ps = $("#preset");
  ps.innerHTML = Object.entries(DEF.presets).map(([k, v]) => `<option value="${k}">${esc(v.label.split(":")[0])}</option>`).join("");
  $("#models").innerHTML = Object.entries(DEF.models).map(([k, v]) =>
    `<label><input type="checkbox" value="${k}"><span class="dot" style="background:${MODEL_COLORS[k]}"></span>` +
    `<span>${esc(k)}<small>${esc(v)}</small></span></label>`).join("");
  ps.value = "recomendado";
  setForm(DEF.presets.recomendado.params);
  $("#preset_hint").textContent = DEF.presets.recomendado.label;
  ps.onchange = () => {
    const pr = DEF.presets[ps.value];
    setForm(pr.params, true);
    $("#preset_hint").textContent = pr.label;
  };
  $("#sensitivity").oninput = (e) => { $("#sensitivity_v").textContent = Number(e.target.value).toFixed(2); };
  $$("[data-pick]").forEach((b) => b.onclick = async () => {
    const v = await pick(b.dataset.kind);
    if (v) $("#" + b.dataset.pick).value = v;
  });
  $$(".tab").forEach((t) => t.onclick = () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
    $$(".pane").forEach((p) => p.classList.toggle("active", p.id === "pane-" + t.dataset.tab));
    if (t.dataset.tab === "mapa") viewer.resize();
  });
  $("#run").onclick = start;
  $("#cancel").onclick = async () => { await api("/api/cancel", {}); $("#stage").textContent = "Cancelando…"; };
  $("#open_report").onclick = () => api("/api/open", { what: "report" });
  $("#open_folder").onclick = () => api("/api/open", { what: "folder" });
  $("#open_excel").onclick = () => api("/api/open", { what: "excel" }).catch((e) => toast(e.message));
  $("#open_prev").onclick = async () => {
    const f = await pick("folder");
    if (!f) return;
    try { const r = await api("/api/load_result", { folder: f }); showResult(r.result); toast("Corrida cargada.", true); }
    catch (e) { toast(e.message); }
  };
  $("#cfg_save").onclick = async () => {
    const f = await pick("json", true); if (!f) return;
    await api("/api/config/save", { path: f, params: getForm() }); toast("Configuración guardada.", true);
  };
  $("#cfg_load").onclick = async () => {
    const f = await pick("json"); if (!f) return;
    try { const r = await api("/api/config/load", { path: f }); setForm(r.params); toast("Configuración cargada.", true); }
    catch (e) { toast(e.message); }
  };
  viewer.init();
}

// ---------------------------------------------------------------- ejecución
async function start() {
  const p = getForm();
  if (!p.dem_path) return toast("Seleccione un DEM.");
  if (!p.models.length) return toast("Seleccione al menos un modelo.");
  try { await api("/api/run", p); } catch (e) { return toast(e.message); }
  $("#log").textContent = ""; logNext = 0;
  $("#run").disabled = true; $("#cancel").disabled = false;
  $("#run").textContent = "Procesando…";
  polling = setInterval(poll, 500);
}
async function poll() {
  let s;
  try { s = await api(`/api/status?since=${logNext}`); } catch (e) { return; }
  if (s.logs.length) {
    const L = $("#log"); L.textContent += s.logs.join("\n") + "\n"; L.scrollTop = L.scrollHeight;
  }
  logNext = s.next;
  $("#bar").style.width = (100 * s.progress).toFixed(1) + "%";
  $("#stage").textContent = s.stage || "…";
  $("#elapsed").textContent = s.elapsed ? `${Math.round(s.elapsed)} s` : "";
  if (!s.running) {
    clearInterval(polling); polling = null;
    $("#run").disabled = false; $("#cancel").disabled = true; $("#run").textContent = "Ejecutar";
    if (s.error) { toast(s.error); $("#stage").textContent = "Error: " + s.error; }
    else if (s.result) { showResult(s.result); toast("Proceso terminado.", true); }
  }
}

// ---------------------------------------------------------------- resultados
async function showResult(r) {
  result = r;
  $("#open_report").disabled = false; $("#open_folder").disabled = false; $("#open_excel").disabled = false;
  $("#bar").style.width = "100%";
  $("#stage").textContent = `Resultados: ${r.output_dir}`;
  let h = "<h2>Resumen por modelo</h2>" + table(r.summary);
  if (r.stats && r.stats.length) h += "<h2>Estadística estructural</h2>" + table(r.stats);
  if (r.campo && r.campo.tabla && r.campo.tabla.length) h += `<h2>Campo vs DEM (correlación de rosetas r = ${fmt(r.campo.r_subverticales)})</h2>` + table(r.campo.tabla);
  if (r.validation && r.validation.length) h += "<h2>Comparación (precisión / recall por longitud)</h2>" + table(r.validation);
  for (const [m, f] of Object.entries(r.families || {})) h += `<h2>Familias — ${esc(m)}</h2>` + table(f);
  $("#resumen").innerHTML = h;
  const figs = r.figures && r.figures.length ? r.figures
    : [["rosetas.png", "Rosetas"], ["longitudes.png", "Longitudes"]];
  const t = Date.now();
  $("#figuras").innerHTML = figs.map(([f, c]) =>
    `<figure><img src="/api/result/figuras/${f}?t=${t}" loading="lazy" onerror="this.parentElement.remove()" alt="${esc(c)}"><figcaption>${esc(c)} · <a href="#" data-open="${esc(f)}">abrir en tamaño completo</a></figcaption></figure>`).join("");
  $$("[data-open]").forEach((a) => a.onclick = (e) => { e.preventDefault(); api("/api/open", { what: "figure", name: a.dataset.open }); });
  $$("#figuras img").forEach((im) => im.onclick = () => {
    const lb = document.createElement("div"); lb.className = "lightbox";
    lb.innerHTML = `<img src="${im.src}">`; lb.onclick = () => lb.remove(); document.body.appendChild(lb);
  });
  await viewer.load(t);
}

// ---------------------------------------------------------------- visor
const viewer = {
  cv: null, ctx: null, img: null, data: null, s: 1, tx: 0, ty: 0, vis: {}, sel: null,
  init() {
    this.cv = $("#map"); this.ctx = this.cv.getContext("2d");
    new ResizeObserver(() => this.resize()).observe(this.cv.parentElement);
    let drag = null;
    this.cv.addEventListener("mousedown", (e) => { drag = { x: e.clientX, y: e.clientY, tx: this.tx, ty: this.ty, moved: false }; this.cv.classList.add("drag"); });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      this.tx = drag.tx + dx; this.ty = drag.ty + dy; this.draw();
    });
    window.addEventListener("mouseup", (e) => {
      if (drag && !drag.moved && e.target === this.cv) this.pickAt(e);
      drag = null; this.cv.classList.remove("drag");
    });
    this.cv.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = this.cv.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const k = Math.exp(-e.deltaY * 0.0015);
      this.tx = mx - (mx - this.tx) * k; this.ty = my - (my - this.ty) * k; this.s *= k; this.draw();
    }, { passive: false });
    ["color_by", "conf_min", "hide_drain", "bg_alpha"].forEach((id) => $("#" + id).addEventListener("input", () => {
      $("#conf_v").textContent = Number($("#conf_min").value).toFixed(2); this.draw();
    }));
    $("#fit").onclick = () => this.fit();
  },
  resize() {
    const p = this.cv.parentElement, d = window.devicePixelRatio || 1;
    this.cv.width = p.clientWidth * d; this.cv.height = p.clientHeight * d;
    this.cv.style.width = p.clientWidth + "px"; this.cv.style.height = p.clientHeight + "px";
    this.ctx.setTransform(d, 0, 0, d, 0, 0); this.draw();
  },
  async load(t) {
    const [img, data] = await Promise.all([
      new Promise((res) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => res(null); im.src = `/api/result/preview/fondo.png?t=${t}`; }),
      fetch(`/api/result/preview/lineas.json?t=${t}`).then((r) => r.ok ? r.json() : null)]);
    this.img = img; this.data = data; this.sel = null; $("#info").hidden = true;
    $("#map_empty").hidden = !!data;
    if (!data) return;
    const names = Object.keys(data.layers);
    const main = result && result.main;
    this.vis = {}; names.forEach((n) => { this.vis[n] = n === main || names.length === 1; });
    $("#layer_list").innerHTML = names.map((n) =>
      `<label class="chk"><input type="checkbox" data-layer="${n}" ${this.vis[n] ? "checked" : ""}>` +
      `<span class="dot" style="width:10px;height:10px;border-radius:50%;background:${MODEL_COLORS[n] || "#999"}"></span>${esc(n)}` +
      `<span class="n">${data.layers[n].features.length}</span></label>`).join("");
    $$("[data-layer]").forEach((c) => c.onchange = () => { this.vis[c.dataset.layer] = c.checked; this.draw(); });
    this.fit();
  },
  fit() {
    if (!this.data) return;
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    this.s = Math.min(W / this.data.width, H / this.data.height) * 0.96;
    this.tx = (W - this.data.width * this.s) / 2; this.ty = (H - this.data.height * this.s) / 2; this.draw();
  },
  color(layer, p, fi) {
    const by = $("#color_by").value;
    if (by === "modelo") return MODEL_COLORS[layer] || "#ddd";
    if (by === "confianza") return CONF_COLORS[p.clase_conf] || "#ddd";
    if (by === "tipo") return TYPE_COLORS[p.tipo] || "#ddd";
    const n = parseInt(String(p.familia || "F1").slice(1), 10) - 1;
    return PALETTE[(n >= 0 ? n : 0) % PALETTE.length];
  },
  visible(p) {
    const cmin = parseFloat($("#conf_min").value);
    if (p.confianza != null && p.confianza < cmin) return false;
    if ($("#hide_drain").checked && p.drenaje != null && p.drenaje >= 0.6) return false;
    return true;
  },
  draw() {
    const c = this.ctx; if (!c) return;
    const W = this.cv.clientWidth, H = this.cv.clientHeight;
    c.fillStyle = "#0b1119"; c.fillRect(0, 0, W, H);
    if (!this.data) return;
    c.save(); c.translate(this.tx, this.ty); c.scale(this.s, this.s);
    if (this.img) { c.globalAlpha = parseFloat($("#bg_alpha").value); c.imageSmoothingEnabled = true; c.drawImage(this.img, 0, 0); c.globalAlpha = 1; }
    const lw = Math.max(1.6 / this.s, 0.3);
    c.lineCap = "round";
    const legend = new Map();
    for (const [name, L] of Object.entries(this.data.layers)) {
      if (!this.vis[name]) continue;
      for (const f of L.features) {
        if (!this.visible(f.p)) continue;
        const col = this.color(name, f.p);
        c.strokeStyle = col; c.lineWidth = lw * (0.7 + (f.p.confianza ?? 0.6));
        c.beginPath(); f.c.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y))); c.stroke();
        const by = $("#color_by").value;
        const key = by === "modelo" ? name : by === "confianza" ? f.p.clase_conf : by === "tipo" ? f.p.tipo : `${name}:${f.p.familia}`;
        if (key && !legend.has(key)) legend.set(key, col);
      }
    }
    if (this.sel) {
      c.strokeStyle = "#ffffff"; c.lineWidth = lw * 3.2; c.globalAlpha = 0.9;
      c.beginPath(); this.sel.c.forEach(([x, y], i) => (i ? c.lineTo(x, y) : c.moveTo(x, y))); c.stroke(); c.globalAlpha = 1;
    }
    c.restore();
    this.legend(legend);
  },
  legend(m) {
    const by = $("#color_by").value, el = $("#legend");
    const fams = {};
    if (by === "familia" && this.data) for (const [n, L] of Object.entries(this.data.layers)) (L.familias || []).forEach((f) => { fams[`${n}:${f.familia}`] = f; });
    const items = [...m.entries()].sort((a, b) => a[0].localeCompare(b[0])).slice(0, 14);
    el.innerHTML = items.map(([k, col]) => {
      let label = k;
      if (fams[k]) { const f = fams[k]; label = `${k.split(":")[0]} · ${f.familia} ${f.rumbo} (${f.pct_longitud.toFixed(0)}%)`; }
      return `<div><i style="background:${col}"></i>${esc(label)}</div>`;
    }).join("");
  },
  pickAt(e) {
    if (!this.data) return;
    const r = this.cv.getBoundingClientRect();
    const x = (e.clientX - r.left - this.tx) / this.s, y = (e.clientY - r.top - this.ty) / this.s;
    const tol = 7 / this.s; let best = null, bd = tol;
    for (const [name, L] of Object.entries(this.data.layers)) {
      if (!this.vis[name]) continue;
      for (const f of L.features) {
        if (!this.visible(f.p)) continue;
        for (let i = 1; i < f.c.length; i++) {
          const d = segDist(x, y, f.c[i - 1], f.c[i]);
          if (d < bd) { bd = d; best = { f, name }; }
        }
      }
    }
    this.sel = best ? best.f : null;
    const info = $("#info");
    if (best) {
      const p = best.f.p;
      const rows = [["Modelo", best.name], ["ID", p.id], ["Longitud", `${fmt(p.longitud_m)} m`], ["Rumbo", p.rumbo], ["Familia", p.familia],
        ["Tipo", p.tipo], ["Evidencias", p.n_evid], ["Confianza", p.confianza != null ? `${fmt(p.confianza)} (${p.clase_conf || ""})` : "—"],
        ["Sobre drenaje", p.drenaje != null ? `${Math.round(100 * p.drenaje)} %` : "—"]];
      info.innerHTML = `<table>${rows.map(([a, b]) => `<tr><td>${a}</td><td>${esc(b ?? "—")}</td></tr>`).join("")}</table>`;
      info.hidden = false;
    } else info.hidden = true;
    this.draw();
  },
};
function segDist(px, py, a, b) {
  const dx = b[0] - a[0], dy = b[1] - a[1], L = dx * dx + dy * dy;
  let t = L ? ((px - a[0]) * dx + (py - a[1]) * dy) / L : 0; t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - a[0] - t * dx, py - a[1] - t * dy);
}

window.addEventListener("DOMContentLoaded", () => init().catch((e) => toast("Error iniciando: " + e.message)));
