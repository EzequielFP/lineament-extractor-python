"""Productos derivados: densidades, intersecciones y reporte HTML (las figuras están en figures.py)."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import cv2
import numpy as np
from rasterio.transform import Affine

# ---------------------------------------------------------------------------
# Densidades
# ---------------------------------------------------------------------------
def _kernel_density(acc: np.ndarray, radius_m: float, cell_m: float) -> np.ndarray:
    """Densidad por km² con núcleo cuártico (Silverman), como Kernel Density de ArcGIS:
    suave y con integral unitaria, así que conserva las unidades de `acc`."""
    rad = max(1, int(round(radius_m / cell_m)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    r2 = (xx ** 2 + yy ** 2) / rad ** 2
    ker = np.where(r2 <= 1, (1 - r2) ** 2, 0).astype(np.float32)
    ker /= ker.sum()
    d = cv2.filter2D(acc.astype(np.float32), -1, ker, borderType=cv2.BORDER_CONSTANT)
    d = np.where(d > 1e-3 * max(float(d.max()), 1e-12), d, 0)   # ruido numérico de la FFT
    return d / (cell_m ** 2 / 1e6)                               # por km²


def _coarse_grid(grid, cell_m):
    f = max(1, int(round(cell_m / grid.res)))
    H, W = int(np.ceil(grid.shape[0] / f)), int(np.ceil(grid.shape[1] / f))
    T = grid.transform * Affine.scale(f, f)
    return f, H, W, T


def length_density(segs_geo: np.ndarray, grid, cell_m: float, radius_m: float):
    """Densidad de lineamientos (km/km²), núcleo cuártico de radio R."""
    f, H, W, T = _coarse_grid(grid, cell_m)
    acc = np.zeros((H, W), np.float64)
    if len(segs_geo):
        inv = ~T
        step = cell_m / 4.0
        for x0, y0, x1, y1 in segs_geo:
            L = np.hypot(x1 - x0, y1 - y0)
            k = max(int(np.ceil(L / step)), 1)
            t = (np.arange(k) + 0.5) / k
            xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            c = np.floor(inv.a * xs + inv.b * ys + inv.c).astype(int)
            r = np.floor(inv.d * xs + inv.e * ys + inv.f).astype(int)
            ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
            np.add.at(acc, (r[ok], c[ok]), L / k)
    dens = _kernel_density(acc, radius_m, grid.res * f)
    return (dens / 1000.0).astype(np.float32), T      # m/km² -> km/km²


def intersections(segs_geo: np.ndarray):
    """Puntos de cruce entre lineamientos (índice espacial STRtree)."""
    from shapely import STRtree
    from shapely.geometry import LineString
    if len(segs_geo) < 2:
        return np.zeros((0, 2))
    lines = [LineString([(a, b), (c, d)]) for a, b, c, d in segs_geo]
    tree = STRtree(lines)
    i, j = tree.query(lines, predicate="intersects")
    m = i < j
    pts = []
    for a, b in zip(i[m], j[m]):
        p = lines[a].intersection(lines[b])
        if p.geom_type == "Point":
            pts.append((p.x, p.y))
    return np.array(pts).reshape(-1, 2)


def point_density(pts: np.ndarray, grid, cell_m: float, radius_m: float):
    f, H, W, T = _coarse_grid(grid, cell_m)
    acc = np.zeros((H, W), np.float32)
    if len(pts):
        inv = ~T
        c = np.floor(inv.a * pts[:, 0] + inv.b * pts[:, 1] + inv.c).astype(int)
        r = np.floor(inv.d * pts[:, 0] + inv.e * pts[:, 1] + inv.f).astype(int)
        ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
        np.add.at(acc, (r[ok], c[ok]), 1)
    return _kernel_density(acc, radius_m, grid.res * f).astype(np.float32), T


# ---------------------------------------------------------------------------
# Reporte HTML autocontenido
# ---------------------------------------------------------------------------
def _img(data: bytes, alt: str) -> str:
    return f'<img alt="{html.escape(alt)}" src="data:image/jpeg;base64,{base64.b64encode(data).decode()}">'


def _table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "<p><em>Sin datos.</em></p>"
    cols = cols or list(rows[0])
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in cols)
    body = ""
    for r in rows:
        tds = ""
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                v = f"{v:,.3f}" if abs(v) < 10 else f"{v:,.1f}"
            tds += f"<td>{html.escape(str(v))}</td>"
        body += f"<tr>{tds}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def html_report(path, meta: dict, summary: list[dict], families: dict, figures: dict,
                validation: list[dict], files: list[str], params: dict):
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f5f7;color:#1d232b}
    header{background:#1f2d3d;color:#fff;padding:22px 32px}
    header h1{margin:0;font-size:22px} header p{margin:4px 0 0;color:#b9c4d0;font-size:13px}
    main{max-width:1180px;margin:0 auto;padding:20px 24px 60px}
    section{background:#fff;border-radius:10px;padding:18px 22px;margin:18px 0;
            box-shadow:0 1px 3px rgba(0,0,0,.08)}
    h2{font-size:17px;margin:0 0 12px;color:#1f2d3d}
    table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
    th,td{border-bottom:1px solid #e3e6ea;padding:6px 8px;text-align:left}
    th{background:#f0f2f5;font-weight:600}
    img{max-width:100%;border:1px solid #e3e6ea;border-radius:6px;margin:6px 0}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}
    .note{font-size:12.5px;color:#5b6570} code{background:#eef1f4;padding:1px 4px;border-radius:3px}
    """
    parts = [f"<!doctype html><html lang='es'><head><meta charset='utf-8'>"
             f"<title>Reporte de lineamientos</title><style>{css}</style></head><body>",
             f"<header><h1>Reporte de extracción de lineamientos</h1>"
             f"<p>{html.escape(meta.get('dem', ''))} · {html.escape(meta.get('fecha', ''))} · "
             f"resolución de trabajo {meta.get('res', 0):.2f} m · {meta.get('shape', '')} px · "
             f"tiempo {meta.get('tiempo_s', 0):.0f} s</p></header><main>"]
    parts.append("<section><h2>Resumen por modelo</h2>" + _table(summary) +
                 "<p class='note'>Densidad = longitud total / área válida. Confianza Alta ≥ 0.60, "
                 "Media ≥ 0.42. En el banco sintético la clase Alta tuvo ~86 % de precisión.</p></section>")
    if validation:
        parts.append("<section><h2>Comparación contra referencia / entre modelos</h2>" +
                     _table(validation) +
                     "<p class='note'>Precisión y recall medidos por LONGITUD: una línea coincide si hay "
                     "otra a menos del buffer y con rumbo dentro de la tolerancia. Si la referencia es "
                     "la salida de otro software (p. ej. Catalyst), esto mide acuerdo, no exactitud.</p></section>")
    for k, fams in families.items():
        parts.append(f"<section><h2>Familias direccionales — {html.escape(k)}</h2>" +
                     _table(fams) + "</section>")
    if figures:
        parts.append("<section><h2>Figuras</h2><div class='grid'>")
        for name, data in figures.items():
            parts.append(f"<div><strong>{html.escape(name)}</strong>{_img(data, name)}</div>")
        parts.append("</div></section>")
    parts.append("<section><h2>Archivos generados</h2><ul>" +
                 "".join(f"<li><code>{html.escape(f)}</code></li>" for f in files) + "</ul></section>")
    parts.append("<section><h2>Parámetros</h2><pre style='font-size:12px;white-space:pre-wrap'>" +
                 html.escape("\n".join(f"{k}: {v}" for k, v in params.items())) + "</pre></section>")
    parts.append("</main></body></html>")
    Path(path).write_text("".join(parts), encoding="utf-8")
