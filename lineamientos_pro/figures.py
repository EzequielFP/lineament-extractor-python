"""
Láminas de publicación (300 dpi) generadas en cada corrida.

  01_mapa_lineamientos      mapa por familias con norte, escala y roseta
  02_analisis_direccional   rosetas (longitud y frecuencia), histograma axial, tabla de familias
  03_longitudes             histograma, ley de potencia N(>L), rumbo vs longitud, cajas por familia
  04_densidad               densidad de lineamientos y de cruces
  05_confianza_tipo         mapa por confianza, longitud por clase, tipo y drenaje por familia
  06_comparacion_modelos    rosetas por modelo, totales y acuerdo entre modelos
  07_campo_vs_dem           estereograma de campo y comparación de rumbos campo / DEM

También se puede ejecutar sobre una corrida existente:
  python -m lineamientos_pro.figures <carpeta_corrida> [--campo datos.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from . import stats as st  # noqa: E402

# ---------------------------------------------------------------- estilo
C_DARK, C_MED, C_LIGHT = "#1A237E", "#3949AB", "#C5CAE9"
C_RED, C_GREY, C_BG = "#C62828", "#455A64", "#FAFAFA"
FAM_COLORS = ["#C62828", "#1565C0", "#2E7D32", "#EF6C00", "#6A1B9A", "#00838F",
              "#AD1457", "#5D4037", "#827717", "#37474F"]
MODEL_COLORS = {"consenso": "#C62828", "valles": "#1565C0", "crestas": "#2E7D32",
                "escarpes": "#EF6C00", "bordes": "#6A1B9A", "pci_line": "#00838F",
                "pci_multi": "#827717", "referencia": "#37474F"}
CONF_COLORS = {"Alta": "#C62828", "Media": "#F9A825", "Baja": "#1E88E5"}
TYPE_COLORS = {"valle": "#1565C0", "cresta": "#2E7D32", "escarpe": "#EF6C00", "borde": "#6A1B9A",
               "borde_sombreado": "#00838F"}
CMAP_DENS = LinearSegmentedColormap.from_list(
    "dens", ["#FFFFFF", "#E8EAF6", "#9FA8DA", "#3949AB", "#1A237E"], N=256)
DPI = 300


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5,
        "axes.titleweight": "bold", "axes.titlecolor": C_DARK, "axes.labelcolor": C_GREY,
        "axes.edgecolor": "#90A4AE", "xtick.color": C_GREY, "ytick.color": C_GREY,
        "axes.grid": False, "savefig.facecolor": "white", "figure.facecolor": "white",
        "legend.fontsize": 8, "legend.framealpha": 0.92, "legend.edgecolor": C_LIGHT,
    })


def fam_color(fam: str) -> str:
    try:
        return FAM_COLORS[(int(str(fam)[1:]) - 1) % len(FAM_COLORS)]
    except ValueError:
        return C_GREY


def _header(fig, title, subtitle="", y=0.99):
    """Título y subtítulo con separación fija en pulgadas (no depende del alto)."""
    h = fig.get_figheight()
    fig.text(0.5, y, title.upper(), ha="center", va="top", fontsize=13.5,
             fontweight="bold", color=C_DARK)
    if subtitle:
        fig.text(0.5, y - 0.33 / h, subtitle, ha="center", va="top", fontsize=9.5, color=C_GREY)


def _top(fig, inches=1.05):
    """Fracción superior disponible bajo el encabezado."""
    return 1 - inches / fig.get_figheight()


def _footer(fig, text, y=0.012):
    if not text:
        return
    fig.add_artist(Line2D([0.05, 0.95], [y + 0.036, y + 0.036], transform=fig.transFigure,
                          color=C_LIGHT, lw=1.0))
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=8.5, color=C_GREY, style="italic",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8EAF6", edgecolor=C_LIGHT))


def _fam_footer(fams, label="Familias"):
    if not fams:
        return ""
    return f"{label}:  " + "  ·  ".join(
        f"{f['familia']} {f['rumbo']} ({f['pct_longitud']:.0f} %)" for f in fams[:8])


def _save(fig, path):
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return Path(path)


def _panel_label(ax, letter, title):
    ax.set_title(f"{letter}.  {title}", loc="center", pad=10)


# ---------------------------------------------------------------- piezas
def rose(ax, az, w=None, bin_deg=10, color=None, fams=None, title="", n_label=None,
         highlight=0.6, alpha=0.9, outline_only=False, label=None):
    """Roseta bidireccional estilo publicación (ejes ya polares)."""
    h, bins = st.rose_counts(az, w, bin_deg)
    th = np.deg2rad(bins[:-1] + bin_deg / 2)
    width = np.deg2rad(bin_deg)
    mx = h.max() if h.max() > 0 else 1
    hn = h / mx
    if outline_only:
        # contorno escalonado (sigue los bordes de cada intervalo)
        ang = np.repeat(np.concatenate([np.deg2rad(bins[:-1]), np.deg2rad(bins[:-1]) + np.pi]), 2)
        ang = np.append(ang[1:], ang[0] + 2 * np.pi)
        rad = np.repeat(np.concatenate([hn, hn]), 2)
        ax.plot(ang, rad, color=color or C_RED, lw=1.8, label=label)
        ax.fill(ang, rad, color=color or C_RED, alpha=0.12)
    else:
        cols = CMAP_DENS(0.25 + 0.75 * hn) if color is None else [color] * len(h)
        for off in (0, np.pi):
            bars = ax.bar(th + off, hn, width=width, color=cols, edgecolor=C_DARK, linewidth=0.4,
                          alpha=alpha, align="center", label=label if off == 0 else None)
            for b, v in zip(bars, hn):
                if highlight and v >= highlight:
                    b.set_edgecolor(C_RED)
                    b.set_linewidth(1.4)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=8.5, color=C_GREY)
    ax.set_yticks([1 / 3, 2 / 3, 1])
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.12)
    ax.spines["polar"].set_color("#90A4AE")
    ax.grid(color="#90A4AE", linewidth=0.35, alpha=0.5)
    ax.set_facecolor(C_BG)
    if fams:
        for f in fams[:6]:
            a = np.deg2rad(f["azimut_medio"])
            c = fam_color(f["familia"])
            for off in (0, np.pi):
                ax.plot([a + off, a + off], [0, 1.1], color=c, lw=1.1, ls="--")
    if title:
        ax.set_title(title, pad=18)
    return h


def fam_legend(ax, fams, y=-0.10, ncol=3):
    if not fams:
        return
    handles = [Line2D([], [], color=fam_color(f["familia"]), lw=1.4, ls="--",
                      label=f"{f['familia']} {f['rumbo']} ({f['pct_longitud']:.0f} %)") for f in fams[:6]]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol,
              fontsize=7.5, frameon=False, handlelength=1.8, columnspacing=1.0)


def _nice_length(span_m):
    for v in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 50000):
        if v >= span_m / 5:
            return v
    return 100000


def north_arrow(ax, x=0.94, y=0.90, size=0.07):
    ax.annotate("", xy=(x, y + size), xytext=(x, y), xycoords="axes fraction",
                arrowprops=dict(facecolor=C_DARK, edgecolor="white", width=5, headwidth=13,
                                headlength=11, linewidth=0.8))
    ax.text(x, y + size + 0.012, "N", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.7))


def scale_bar(ax, extent, loc=(0.05, 0.05)):
    x0, x1, y0, y1 = extent
    L = _nice_length(x1 - x0)
    bx = x0 + loc[0] * (x1 - x0)
    by = y0 + loc[1] * (y1 - y0)
    h = 0.012 * (y1 - y0)
    for i in range(4):
        ax.add_patch(plt.Rectangle((bx + i * L / 4, by), L / 4, h, facecolor=C_DARK if i % 2 == 0 else "white",
                                   edgecolor=C_DARK, lw=0.6, zorder=10))
    lab = f"{L / 1000:g} km" if L >= 1000 else f"{L:g} m"
    for xx, t in ((bx, "0"), (bx + L, lab)):
        ax.text(xx, by + 1.8 * h, t, ha="center", va="bottom", fontsize=7.5, color=C_DARK, zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", edgecolor="none", alpha=0.7))


def map_axes(ax, bg, extent, gdf, color_by="familia", lw_by_conf=True, title="", legend=True,
             fams=None, bg_alpha=1.0, cmap_bg="gray"):
    if bg is not None:
        ax.imshow(bg, cmap=cmap_bg, extent=extent, vmin=0.15, vmax=1.0, alpha=bg_alpha,
                  interpolation="bilinear", zorder=0)
    if len(gdf):
        segs, cols, lws = [], [], []
        for _, r in gdf.iterrows():
            c = np.asarray(r.geometry.coords)[:, :2]
            segs.append(c)
            if color_by == "familia":
                cols.append(fam_color(r.get("familia", "F1")))
            elif color_by == "confianza":
                cols.append(CONF_COLORS.get(r.get("clase_conf"), C_RED))
            elif color_by == "tipo":
                cols.append(TYPE_COLORS.get(r.get("tipo"), C_GREY))
            else:
                cols.append(color_by)
            conf = r.get("confianza")
            lws.append(0.6 + 1.5 * (conf if (lw_by_conf and conf is not None and np.isfinite(conf)) else 0.4))
        ax.add_collection(LineCollection(segs, colors=cols, linewidths=lws, capstyle="round", zorder=3))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=6.5)
    for lab in ax.get_yticklabels():
        lab.set_rotation(90)
        lab.set_va("center")
    north_arrow(ax)
    scale_bar(ax, extent)
    if legend and len(gdf):
        handles = []
        if color_by == "familia":
            fl = {f["familia"]: f for f in (fams or [])}
            for fam in sorted(gdf["familia"].unique(), key=lambda s: int(str(s)[1:] or 0)):
                f = fl.get(fam)
                lab = f"{fam}  {f['rumbo']}  ({f['pct_longitud']:.0f} %)" if f else fam
                handles.append(Line2D([], [], color=fam_color(fam), lw=2.2, label=lab))
        elif color_by == "confianza":
            handles = [Line2D([], [], color=v, lw=2.2, label=f"Confianza {k}") for k, v in CONF_COLORS.items()]
        elif color_by == "tipo":
            handles = [Line2D([], [], color=TYPE_COLORS.get(t, C_GREY), lw=2.2, label=t)
                       for t in sorted(gdf["tipo"].dropna().unique())]
        if handles:
            ax.legend(handles=handles, loc="lower right", fontsize=7.5)
    if title:
        ax.set_title(title, pad=8)


def _table(ax, rows, cols, headers=None, col_widths=None, fontsize=8):
    ax.axis("off")
    if not rows:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", color=C_GREY)
        return
    cell = [[_fmt(r.get(c, "")) for c in cols] for r in rows]
    t = ax.table(cellText=cell, colLabels=headers or cols, loc="upper center", cellLoc="center",
                 colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.35)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(C_LIGHT)
        if i == 0:
            c.set_facecolor("#E8EAF6")
            c.set_text_props(color=C_DARK, fontweight="bold")
        elif cols[j] == "familia":
            c.set_text_props(color=fam_color(rows[i - 1]["familia"]), fontweight="bold")


def _fmt(v):
    if isinstance(v, float):
        if not np.isfinite(v):
            return "—"
        return f"{v:.3g}" if abs(v) < 1 else (f"{v:.1f}" if abs(v) < 1000 else f"{v:,.0f}")
    return str(v)


# ---------------------------------------------------------------- láminas
def plate_map(out, bg, extent, gdf, fams, model, meta):
    fig = plt.figure(figsize=(11, 11.8))
    _header(fig, f"Lineamientos estructurales — {meta.get('zona', '')}".rstrip(" —"),
            f"Modelo {model}  ·  n = {len(gdf)}  ·  {gdf['longitud_m'].sum() / 1000:.1f} km  ·  "
            f"{meta.get('fuente', '')}")
    ax = fig.add_axes([0.06, 0.07, 0.88, 0.83])
    map_axes(ax, bg, extent, gdf, "familia", fams=fams)
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0.008, 0.752), 0.235, 0.24, boxstyle="round,pad=0.004",
                                transform=ax.transAxes, facecolor="white", alpha=0.88,
                                edgecolor=C_LIGHT, zorder=8))
    ins = ax.inset_axes([0.028, 0.775, 0.195, 0.195], projection="polar", zorder=9)
    rose(ins, gdf["azimut"], gdf["longitud_m"], bin_deg=10, highlight=0)
    ins.set_xticklabels(["N", "", "E", "", "S", "", "W", ""], fontsize=6.5)
    ins.patch.set_alpha(0.85)
    _footer(fig, _fam_footer(fams))
    return _save(fig, out)


def plate_directional(out, gdf, fams, model, meta):
    nrow = max(len(fams), 1)
    fig = plt.figure(figsize=(16, 7.6 + 0.28 * nrow))
    _header(fig, "Análisis direccional de lineamientos",
            f"Modelo {model}  ·  n = {len(gdf)}  ·  rosetas bidireccionales, intervalos de 10°")
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[5.2, 0.9 + 0.3 * nrow], hspace=0.38,
                           wspace=0.28, left=0.05, right=0.97, top=_top(fig, 1.25), bottom=0.07)
    az, L = gdf["azimut"].values, gdf["longitud_m"].values
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    rose(ax, az, L, fams=fams)
    fam_legend(ax, fams)
    _panel_label(ax, "A", "Roseta ponderada por longitud")
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    rose(ax, az, None, fams=fams)
    fam_legend(ax, fams)
    _panel_label(ax, "B", "Roseta de frecuencia")
    # C: histograma axial + densidad von Mises
    ax = fig.add_subplot(gs[0, 2])
    h, bins = st.rose_counts(az, L / 1000, 5)
    ax.bar(bins[:-1] + 2.5, h, width=5, color=C_LIGHT, edgecolor=C_MED, lw=0.5)
    grid = np.arange(0, 180.5, 0.5)
    d = np.deg2rad(2 * (grid[:, None] - az[None, :]))
    kde = (np.exp(25 * (np.cos(d) - 1)) * L[None, :]).sum(1)
    ax.plot(grid, kde / kde.max() * h.max(), color=C_DARK, lw=1.6, label="densidad (von Mises)")
    for f in fams[:6]:
        c = fam_color(f["familia"])
        ax.axvline(f["azimut_medio"], color=c, ls="--", lw=1.1)
        ax.text(f["azimut_medio"], h.max() * 1.04, f["familia"], color=c, ha="center",
                fontsize=8, fontweight="bold")
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xticklabels(["N", "N30E", "N60E", "E", "N60W", "N30W", "N"])
    ax.set_ylabel("Longitud (km) por intervalo de 5°")
    ax.set_ylim(0, h.max() * 1.15 if h.max() else 1)
    ax.legend(loc="upper right")
    _panel_label(ax, "C", "Distribución axial de rumbos")
    # D: tabla de familias
    ax = fig.add_subplot(gs[1, :])
    rows = st.family_stats(gdf)
    cols = ["familia", "rumbo", "azimut_medio", "desv_circular", "n", "long_km", "pct_long",
            "long_media_m", "long_max_m"]
    hdr = ["Familia", "Rumbo", "Azimut medio (°)", "Desv. circular (°)", "n", "Longitud (km)",
           "% longitud", "Long. media (m)", "Long. máx. (m)"]
    if rows and "tipo_dominante" in rows[0]:
        cols += ["tipo_dominante"]
        hdr += ["Tipo dominante"]
    if rows and "confianza_media" in rows[0]:
        cols += ["confianza_media"]
        hdr += ["Confianza media"]
    _table(ax, rows, cols, hdr)
    ax.set_title("D.  Estadística por familia direccional", pad=4)
    ax_all = st.axial_stats(az, L)
    sig = "orientación preferente significativa" if ax_all["rayleigh_p"] < 0.05 else "sin orientación preferente única"
    _footer(fig, f"Media axial ponderada {ax_all['azimut_medio']:.0f}° ({st._rumbo(ax_all['azimut_medio'])})  ·  "
                 f"R = {ax_all['R']:.2f}  ·  Rayleigh p = {ax_all['rayleigh_p']:.2g} ({sig})")
    return _save(fig, out)


def plate_lengths(out, gdf, fams, model, meta):
    fig = plt.figure(figsize=(15, 10.5))
    _header(fig, "Estadística de longitudes", f"Modelo {model}  ·  n = {len(gdf)}")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.22, left=0.06, right=0.97,
                           top=_top(fig, 1.0), bottom=0.1)
    L = gdf["longitud_m"].values
    ls = st.length_stats(L)
    ax = fig.add_subplot(gs[0, 0])
    b = np.logspace(np.log10(max(L.min(), 1)), np.log10(L.max() * 1.01), 28)
    ax.hist(L, bins=b, color=C_LIGHT, edgecolor=C_MED, lw=0.6)
    ax.set_xscale("log")
    ax.axvline(ls["mediana_m"], color=C_RED, ls="--", lw=1.3, label=f"mediana {ls['mediana_m']:.0f} m")
    ax.axvline(ls["media_m"], color=C_DARK, ls=":", lw=1.3, label=f"media {ls['media_m']:.0f} m")
    ax.set_xlabel("Longitud (m)")
    ax.set_ylabel("N.º de lineamientos")
    ax.legend()
    _panel_label(ax, "A", "Histograma de longitudes")
    # B: ley de potencia
    ax = fig.add_subplot(gs[0, 1])
    x = np.sort(L)
    ccdf = 1 - np.arange(len(x)) / len(x)
    ax.loglog(x, ccdf * len(x), "o", ms=3, color=C_MED, alpha=0.6, label="observado")
    pl = st.power_law_fit(L)
    if np.isfinite(pl["exponente"]):
        xx = x[x >= pl["xmin"]]
        # el ajuste es sobre la fracción de la cola (L >= xmin): se escala al tamaño de la cola
        ax.loglog(xx, 10 ** (pl["intercepto"]) * xx ** (-pl["exponente"]) * len(xx), color=C_RED, lw=1.8,
                  label=f"N(>L) ∝ L^-{pl['exponente']:.2f}  (R² = {pl['r2']:.2f})")
        ax.axvline(pl["xmin"], color=C_GREY, ls=":", lw=1)
    ax.set_xlabel("Longitud L (m)")
    ax.set_ylabel("N.º de lineamientos con longitud > L")
    ax.legend()
    ax.grid(True, which="both", color="#CFD8DC", lw=0.4)
    _panel_label(ax, "B", "Distribución acumulada (ley de potencia)")
    # C: rumbo vs longitud
    ax = fig.add_subplot(gs[1, 0])
    c = [fam_color(f) for f in gdf["familia"]]
    s = 8 + 40 * gdf["confianza"].fillna(0.5).values if "confianza" in gdf else 14
    ax.scatter(gdf["azimut"], L, c=c, s=s, alpha=0.75, edgecolor="white", lw=0.3)
    ax.set_yscale("log")
    ax.set_xlim(0, 180)
    ax.set_xticks(range(0, 181, 30))
    ax.set_xticklabels(["N", "N30E", "N60E", "E", "N60W", "N30W", "N"])
    ax.set_ylabel("Longitud (m)")
    ax.set_xlabel("Rumbo")
    for f in fams[:6]:
        ax.axvline(f["azimut_medio"], color=fam_color(f["familia"]), ls="--", lw=0.9)
    _panel_label(ax, "C", "Rumbo vs longitud (tamaño = confianza)")
    # D: cajas por familia
    ax = fig.add_subplot(gs[1, 1])
    fl = sorted(gdf["familia"].unique(), key=lambda s: int(str(s)[1:] or 0))
    data = [gdf.loc[gdf["familia"] == f, "longitud_m"].values for f in fl]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                    flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for p, f in zip(bp["boxes"], fl):
        p.set_facecolor(fam_color(f))
        p.set_alpha(0.55)
        p.set_edgecolor(C_DARK)
    for m in bp["medians"]:
        m.set_color(C_DARK)
    fr = {f["familia"]: f["rumbo"] for f in fams}
    ax.set_xticks(range(1, len(fl) + 1))
    ax.set_xticklabels([f"{f}\n{fr.get(f, '')}" for f in fl])
    ax.set_yscale("log")
    ax.set_ylabel("Longitud (m)")
    _panel_label(ax, "D", "Longitud por familia")
    _footer(fig, f"Total {ls['total_km']:.2f} km  ·  media {ls['media_m']:.0f} m  ·  mediana {ls['mediana_m']:.0f} m  ·  "
                 f"P10–P90 {ls['p10_m']:.0f}–{ls['p90_m']:.0f} m  ·  máx. {ls['max_m']:.0f} m  ·  "
                 f"exponente {pl['exponente']:.2f}")
    return _save(fig, out)


def plate_density(out, bg, extent, gdf, rasters, pts, model, meta, radius_m):
    """rasters: lista de (arr, extent, etiqueta, título)."""
    n = len(rasters)
    fig = plt.figure(figsize=(8.2 * n, 9.6))
    _header(fig, "Densidad de lineamientos",
            f"Modelo {model}  ·  núcleo cuártico, radio {radius_m:g} m")
    gs = gridspec.GridSpec(1, n, figure=fig, wspace=0.14, left=0.04, right=0.97,
                           top=_top(fig, 1.15), bottom=0.1)
    for k, (arr, ext, lab, ttl) in enumerate(rasters):
        ax = fig.add_subplot(gs[0, k])
        if bg is not None:
            ax.imshow(bg, cmap="gray", extent=extent, vmin=0.15, vmax=1.0, zorder=0)
        a = np.where(arr > 1e-3 * max(float(arr.max()), 1e-12), arr, 0)
        im = ax.imshow(np.ma.masked_equal(a, 0), cmap="magma_r", extent=ext, alpha=0.72, zorder=1,
                       interpolation="bilinear", vmin=0)
        if a.max() > 0:
            H, W = a.shape
            dx, dy = (ext[1] - ext[0]) / W, (ext[3] - ext[2]) / H
            xs = np.linspace(ext[0] + dx / 2, ext[1] - dx / 2, W)
            ys = np.linspace(ext[3] - dy / 2, ext[2] + dy / 2, H)      # fila 0 = norte
            ax.contour(xs, ys, a, levels=np.linspace(0, a.max(), 6)[1:-1], colors="white",
                       linewidths=0.6, zorder=2)
        if k == 0 and len(gdf):
            segs = [np.asarray(g.coords)[:, :2] for g in gdf.geometry]
            ax.add_collection(LineCollection(segs, colors=C_DARK, linewidths=0.6, alpha=0.55, zorder=3))
        if k == n - 1 and len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=14, marker="x", c=C_DARK, zorder=4, lw=1,
                       label=f"cruces (n={len(pts)})")
            ax.legend(loc="lower right")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect("equal")
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=6.5)
        north_arrow(ax)
        scale_bar(ax, extent)
        cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cb.set_label(lab)
        ax.set_title(f"{chr(65 + k)}.  {ttl}", pad=8)
    _footer(fig, f"Densidad media {meta.get('densidad', 0):.2f} km/km²  ·  máxima {rasters[0][0].max():.2f} km/km²  ·  "
                 f"frecuencia {meta.get('frecuencia', 0):.1f} lin./km²  ·  {len(pts)} cruces "
                 f"({meta.get('cruces_por_lin', 0):.2f} por lineamiento)")
    return _save(fig, out)


def plate_confidence(out, bg, extent, gdf, fams, model, meta):
    fig = plt.figure(figsize=(16, 10))
    _header(fig, "Confianza, tipo de evidencia y drenaje", f"Modelo {model}  ·  n = {len(gdf)}")
    gs = gridspec.GridSpec(3, 2, figure=fig, width_ratios=[1.25, 1], hspace=0.55, wspace=0.18,
                           left=0.04, right=0.97, top=_top(fig, 1.0), bottom=0.09)
    ax = fig.add_subplot(gs[:, 0])
    map_axes(ax, bg, extent, gdf, "confianza")
    ax.set_title("A.  Lineamientos por clase de confianza", pad=8)
    fl = sorted(gdf["familia"].unique(), key=lambda s: int(str(s)[1:] or 0))
    fr = {f["familia"]: f["rumbo"] for f in fams}
    xl = [f"{f} {fr.get(f, '')}" for f in fl]
    # B: longitud por clase y familia
    ax = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(fl))
    for c, col in CONF_COLORS.items():
        v = np.array([gdf.loc[(gdf["familia"] == f) & (gdf["clase_conf"] == c), "longitud_m"].sum() / 1000
                      for f in fl])
        ax.bar(xl, v, bottom=bottom, color=col, label=c, edgecolor="white", lw=0.5)
        bottom += v
    ax.set_ylabel("km")
    ax.legend(ncol=3, loc="upper right")
    ax.set_title("B.  Longitud por familia y confianza")
    # C: tipo de evidencia
    ax = fig.add_subplot(gs[1, 1])
    bottom = np.zeros(len(fl))
    for t in sorted(gdf["tipo"].dropna().unique()):
        v = np.array([gdf.loc[(gdf["familia"] == f) & (gdf["tipo"] == t), "longitud_m"].sum() / 1000 for f in fl])
        ax.bar(xl, v, bottom=bottom, color=TYPE_COLORS.get(t, C_GREY), label=t, edgecolor="white", lw=0.5)
        bottom += v
    ax.set_ylabel("km")
    ax.legend(ncol=4, loc="upper right")
    ax.set_title("C.  Evidencia dominante por familia")
    # D: drenaje
    ax = fig.add_subplot(gs[2, 1])
    if "drenaje" in gdf and gdf["drenaje"].notna().any():
        v = [100 * (gdf.loc[gdf["familia"] == f, "sigue_dren"].fillna(False).astype(bool)
                    * gdf.loc[gdf["familia"] == f, "longitud_m"]).sum()
             / max(gdf.loc[gdf["familia"] == f, "longitud_m"].sum(), 1e-9) for f in fl]
        ax.bar(xl, v, color=[fam_color(f) for f in fl], alpha=0.75, edgecolor=C_DARK, lw=0.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% longitud")
        for i, val in enumerate(v):
            ax.text(i, val + 2, f"{val:.0f} %", ha="center", fontsize=7.5, color=C_GREY)
    else:
        ax.text(0.5, 0.5, "Análisis de drenaje no disponible", ha="center", va="center",
                transform=ax.transAxes, color=C_GREY)
    ax.set_title("D.  Lineamientos que siguen la red de drenaje")
    _footer(fig, "Confianza: Alta ≥ 0.60 · Media ≥ 0.42 · Baja < 0.42  ·  Un lineamiento sobre drenaje "
                 "no se descarta: los tramos rectos de cauce suelen estar controlados por fracturas")
    return _save(fig, out)


def plate_models(out, results, agreement, area_km2):
    items = [(k, r) for k, r in results.items() if len(r["gdf"])]
    n = len(items)
    cols = min(n, 4)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(4.2 * cols + 1, 4.1 * rows + 5.0))
    _header(fig, "Comparación de modelos", f"{n} modelos  ·  rosetas ponderadas por longitud")
    gs = gridspec.GridSpec(rows + 1, cols, figure=fig, height_ratios=[1] * rows + [1.1],
                           hspace=0.5, wspace=0.35, left=0.05, right=0.97, top=_top(fig, 1.55), bottom=0.06)
    for i, (k, r) in enumerate(items):
        ax = fig.add_subplot(gs[i // cols, i % cols], projection="polar")
        g = r["gdf"]
        rose(ax, g["azimut"], g["longitud_m"], color=MODEL_COLORS.get(k, C_MED), highlight=0, alpha=0.75)
        ax.set_title(f"{k}\nn = {len(g)} · {g['longitud_m'].sum() / 1000:.1f} km", pad=16,
                     color=MODEL_COLORS.get(k, C_DARK))
    sub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[rows, :], wspace=0.3, width_ratios=[1, 1])
    ax = fig.add_subplot(sub[0, 0])
    names = [k for k, _ in items]
    dens = [r["gdf"]["longitud_m"].sum() / 1000 / max(area_km2, 1e-9) for _, r in items]
    med = [np.median(r["gdf"]["longitud_m"]) for _, r in items]
    x = np.arange(len(names))
    ax.bar(x, dens, color=[MODEL_COLORS.get(k, C_MED) for k in names], alpha=0.8, edgecolor=C_DARK, lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Densidad (km/km²)")
    ax2 = ax.twinx()
    ax2.plot(x, med, "D-", color=C_DARK, ms=5, lw=1)
    ax2.set_ylabel("Longitud mediana (m)")
    ax.set_title("Densidad y longitud mediana")
    ax = fig.add_subplot(sub[0, 1])
    if agreement is not None and len(agreement["names"]) > 1:
        M = agreement["f1"]
        im = ax.imshow(M, cmap=CMAP_DENS, vmin=0, vmax=1)
        nm = agreement["names"]
        ax.set_xticks(range(len(nm)))
        ax.set_xticklabels(nm, rotation=30, ha="right")
        ax.set_yticks(range(len(nm)))
        ax.set_yticklabels(nm)
        for i in range(len(nm)):
            for j in range(len(nm)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 else C_DARK)
        fig.colorbar(im, ax=ax, shrink=0.8, label="F1 (acuerdo por longitud)")
        ax.set_title(f"Acuerdo entre modelos (buffer {agreement['buffer']:g} m, {agreement['angle']:g}°)")
    else:
        ax.axis("off")
    return _save(fig, out)


def plate_field(out, field, gdf, fams, model, meta):
    """Estereograma de campo + comparación de rumbos con los lineamientos."""
    import importlib
    importlib.import_module("mplstereonet")   # registra la proyección "stereonet"
    from .attributes import find_families
    j = field[field["tipo"] == "diaclasa"]
    fo = field[field["tipo"] == "foliacion"]
    fa = field[field["tipo"] == "falla"]
    jv = j[j["dip"] >= 60]
    fig = plt.figure(figsize=(17, 10.8))
    _header(fig, f"Datos de campo vs lineamientos del DEM — {meta.get('zona', '')}".rstrip(" —"),
            f"n = {len(j)} diaclasas ({len(jv)} subverticales ≥ 60°)  ·  n = {len(fo)} foliaciones  ·  "
            f"{len(fa)} fallas  ·  modelo {model}: n = {len(gdf)}  ·  proyección equiárea, hemisferio inferior")
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.1, 0.62], hspace=0.42, wspace=0.28,
                           left=0.04, right=0.97, top=_top(fig, 1.6), bottom=0.1)
    # A: estereograma
    ax = fig.add_subplot(gs[0, 0], projection="stereonet")
    ax.set_facecolor(C_BG)
    if len(j) >= 3:
        ax.density_contourf(j["strike"], j["dip"], measurement="poles", method="exponential_kamb",
                            sigma=3, cmap=CMAP_DENS, alpha=0.85, zorder=1)
        ax.density_contour(j["strike"], j["dip"], measurement="poles", method="exponential_kamb",
                           sigma=3, colors=C_MED, linewidths=0.5, zorder=2)
    ax.pole(j["strike"], j["dip"], "o", ms=3.5, color=C_DARK, alpha=0.7, zorder=3,
            label=f"Polos diaclasas (n={len(j)})")
    for _, r in fa.iterrows():
        ax.plane(r["strike"], r["dip"], color=C_RED, lw=1.8, ls="--", zorder=4,
                 label=f"Falla {int(r['dd'])}/{int(r['dip'])}")
        ax.pole(r["strike"], r["dip"], "D", ms=6, color=C_RED, zorder=5)
    if len(fo):
        ax.plane(fo["strike"], fo["dip"], color="#1E88E5", lw=0.9, alpha=0.7, zorder=3)
        ax.pole(fo["strike"], fo["dip"], "s", ms=5, color="#1E88E5", mec=C_DARK, mew=0.5, zorder=4,
                label=f"Foliaciones (n={len(fo)})")
    ax.grid(color="#90A4AE", lw=0.35, alpha=0.5)
    ax.set_azimuth_ticks([])
    ax.set_title("A.  Polos de diaclasas + densidad (Kamb exponencial)", pad=22)
    ax.text(0.5, 1.02, "N", transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold", color=C_GREY)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.08, -0.02), fontsize=7.5, ncol=1)
    # B: roseta de rumbos de campo (subverticales)
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    _, ffams = find_families(jv["rumbo_axial"].values, np.ones(len(jv)), 6, kappa=12, min_rel=0.35)
    rose(ax, jv["rumbo_axial"], None, bin_deg=15, fams=None, highlight=0.6)
    for f in ffams[:6]:
        a = np.deg2rad(f["azimut_medio"])
        for off in (0, np.pi):
            ax.plot([a + off] * 2, [0, 1.1], color=C_DARK, lw=1, ls=":")
    ax.set_title(f"B.  Rumbo de diaclasas subverticales\n(dip ≥ 60°, n = {len(jv)}, intervalos 15°)", pad=18)
    ax.text(0.5, -0.1, "Familias de campo: " + " · ".join(f"{g['rumbo']} (n={g['n']})" for g in ffams[:6]),
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color=C_GREY)
    # C: superposición
    ax = fig.add_subplot(gs[0, 2], projection="polar")
    rose(ax, gdf["azimut"], gdf["longitud_m"], bin_deg=15, color=C_RED, highlight=0, alpha=0.45,
         label=f"Lineamientos DEM ({model})")
    rose(ax, jv["rumbo_axial"], None, bin_deg=15, outline_only=True, color=C_DARK,
         label="Diaclasas de campo")
    rc = st.rose_correlation(gdf["azimut"], gdf["longitud_m"], jv["rumbo_axial"], None, 15)
    rc_all = st.rose_correlation(gdf["azimut"], gdf["longitud_m"], j["rumbo_axial"], None, 15)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.15, -0.02), fontsize=7.5)
    ax.set_title(f"C.  Superposición DEM vs campo\ncorrelación de rosetas r = {rc:.2f}", pad=18)
    # D: tabla de correspondencia de familias
    ax = fig.add_subplot(gs[1, :2])
    rows = []
    fa_az = [g["azimut_medio"] for g in ffams]

    def adiff(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)
    for f in fams[:8]:
        if not ffams:
            break
        d = [adiff(f["azimut_medio"], a) for a in fa_az]
        k = int(np.argmin(d))
        # ¿la familia DEM es bisectriz de un par de familias de campo que la flanquean?
        bis, best = "—", None
        for i in range(len(fa_az)):
            for jj in range(i + 1, len(fa_az)):
                sep = adiff(fa_az[i], fa_az[jj])
                if not 20 <= sep <= 80:
                    continue
                b1 = (fa_az[i] + ((fa_az[jj] - fa_az[i] + 90) % 180 - 90) / 2) % 180
                e = adiff(b1, f["azimut_medio"])
                if e <= 10 and (best is None or e < best[0]):
                    best = (e, i, jj, sep)
        if best:
            e, i, jj, sep = best
            bis = f"{ffams[i]['rumbo']} / {ffams[jj]['rumbo']}  (separadas {sep:.0f}°, Δ {e:.0f}°)"
        rows.append({"familia": f["familia"], "rumbo": f["rumbo"], "pct": f["pct_longitud"],
                     "campo": f"{ffams[k]['rumbo']} (n={ffams[k]['n']})", "delta": d[k],
                     "ok": "Sí" if d[k] <= 15 else ("Parcial" if d[k] <= 25 else "No"), "bis": bis})
    _table(ax, rows, ["familia", "rumbo", "pct", "campo", "delta", "ok", "bis"],
           ["Familia DEM", "Rumbo", "% long.", "Campo más cercana", "Δ rumbo (°)",
            "¿Coincide? (≤15°)", "Entre las familias de campo (bisectriz)"],
           col_widths=[0.09, 0.07, 0.07, 0.15, 0.09, 0.12, 0.35], fontsize=7.8)
    ax.set_title("D.  Correspondencia entre familias de lineamientos y de diaclasas", pad=4)
    # E: histograma de buzamientos
    ax = fig.add_subplot(gs[1, 2])
    ax.hist(j["dip"], bins=np.arange(0, 95, 5), color=C_LIGHT, edgecolor=C_MED)
    ax.axvline(60, color=C_RED, ls="--", lw=1.2, label="umbral subvertical")
    ax.set_xlabel("Buzamiento (°)")
    ax.set_ylabel("N.º de diaclasas")
    ax.legend()
    ax.set_title("E.  Buzamiento de diaclasas")
    fam_txt = "  ·  ".join(f"{g['rumbo']} n={g['n']}" for g in ffams[:8])
    _footer(fig, f"Familias de campo (rumbo, diaclasas ≥ 60°): {fam_txt}  ·  r (todas las diaclasas) = {rc_all:.2f}")
    return _save(fig, out), {"r_subverticales": rc, "r_todas": rc_all, "familias_campo": ffams, "tabla": rows}


# ---------------------------------------------------------------- orquestación
def model_agreement(results, buffer, angle, names=None):
    from .validation import gdf_to_segments, match_scores
    names = names or [k for k, r in results.items() if len(r["gdf"])]
    if len(names) < 2:
        return None
    segs = {k: gdf_to_segments(results[k]["gdf"]) for k in names}
    M = np.eye(len(names))
    for i, a in enumerate(names):
        for jj, b in enumerate(names):
            if jj > i:
                M[i, jj] = M[jj, i] = match_scores(segs[a], segs[b], buffer, angle)["f1"]
    return {"names": names, "f1": M, "buffer": buffer, "angle": angle}


def make_all(outdir, results, main, bg, extent, area_km2, density_rasters, pts, P,
             field=None, meta=None, log=print):
    """Genera todas las láminas. Devuelve [(archivo, título)] y estadísticas."""
    _style()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta or {})
    figs, extra = [], {}
    g = results[main]["gdf"]
    fams = results[main]["familias"]
    ms = st.model_stats(g, area_km2, len(pts))
    meta.update(densidad=ms.get("densidad_km_km2", 0), cruces_por_lin=ms.get("cruces_por_lin", 0),
                frecuencia=ms.get("frecuencia_n_km2", 0))

    def add(fn, title, *a, **k):
        try:
            r = fn(*a, **k)
            p = r[0] if isinstance(r, tuple) else r
            figs.append((p.name, title))
            return r
        except Exception as e:  # una lámina fallida no detiene las demás
            log(f"    (lámina '{title}' omitida: {e})")
            return None

    if len(g):
        add(plate_map, "Mapa de lineamientos", outdir / "01_mapa_lineamientos.png", bg, extent, g, fams, main, meta)
        add(plate_directional, "Análisis direccional", outdir / "02_analisis_direccional.png", g, fams, main, meta)
        add(plate_lengths, "Estadística de longitudes", outdir / "03_longitudes.png", g, fams, main, meta)
        add(plate_density, "Densidad", outdir / "04_densidad.png", bg, extent, g, density_rasters,
            pts, main, meta, P.density_radius_m)
        if "clase_conf" in g and g["clase_conf"].notna().any():
            add(plate_confidence, "Confianza y tipo", outdir / "05_confianza_tipo.png", bg, extent, g, fams, main, meta)
    if sum(1 for r in results.values() if len(r["gdf"])) > 1:
        agr = model_agreement(results, P.val_buffer_m, P.val_angle_deg)
        add(plate_models, "Comparación de modelos", outdir / "06_comparacion_modelos.png", results, agr, area_km2)
        if agr:
            extra["acuerdo"] = {"names": agr["names"], "f1": agr["f1"].round(3).tolist()}
    if field is not None and len(g):
        r = add(plate_field, "Campo vs DEM", outdir / "07_campo_vs_dem.png", field, g, fams, main, meta)
        if r:
            extra["campo"] = r[1]
    # mapas individuales por modelo
    for k, r in results.items():
        if len(r["gdf"]) and k not in (main, "referencia"):
            add(plate_map, f"Mapa {k}", outdir / f"mapa_{k}.png", bg, extent, r["gdf"], r["familias"], k, meta)
    extra["stats_principal"] = ms
    return figs, extra


# ---------------------------------------------------------------- CLI
def _from_run(folder, field_path=None, zona=""):
    """Regenera las láminas de una corrida existente."""
    import json

    import geopandas as gpd
    import pyogrio
    import rasterio

    from .config import Params
    folder = Path(folder)
    P = Params.load(folder / "parametros.json")
    gpkg = folder / "lineamientos.gpkg"
    results, pts = {}, np.zeros((0, 2))
    from .attributes import find_families
    for name, _ in pyogrio.list_layers(gpkg):
        if name.startswith("lin_"):
            g = gpd.read_file(gpkg, layer=name)
            _, fams = find_families(g["azimut"].values, g["longitud_m"].values, P.max_families)
            results[name[4:]] = {"gdf": g, "familias": fams}
        elif name.startswith("cruces_"):
            c = gpd.read_file(gpkg, layer=name)
            pts = np.column_stack([c.geometry.x, c.geometry.y])
    main = "consenso" if "consenso" in results else next(iter(results))
    with rasterio.open(folder / "rasters" / "sombreado_multidireccional.tif") as s:
        f = max(1, int(np.ceil(max(s.shape) / 2000)))
        bg = s.read(1, out_shape=(s.height // f, s.width // f)) / 255.0
        b = s.bounds
        extent = (b.left, b.right, b.bottom, b.top)
        area = float((s.read(1) < 255).sum() * abs(s.transform.a * s.transform.e) / 1e6)

    def rd(name):
        with rasterio.open(folder / "rasters" / name) as s:
            b = s.bounds
            return s.read(1), (b.left, b.right, b.bottom, b.top)
    rasters = [(*rd(f"densidad_longitud_{main}.tif"), "km/km²", "Densidad de longitud")]
    if (folder / "rasters" / f"densidad_frecuencia_{main}.tif").exists():
        rasters.append((*rd(f"densidad_frecuencia_{main}.tif"), "lineamientos/km²",
                        "Densidad de frecuencia y cruces"))
    else:
        rasters.append((*rd(f"densidad_cruces_{main}.tif"), "cruces/km²", "Densidad de cruces"))
    field = st.read_field_data(field_path) if field_path else None
    figs, extra = make_all(folder / "figuras", results, main, bg, extent, area, rasters,
                           pts, P, field, {"zona": zona, "fuente": Path(P.dem_path).name})
    print(json.dumps([f for f, _ in figs], ensure_ascii=False))
    return figs


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Regenera las láminas de una corrida")
    ap.add_argument("carpeta")
    ap.add_argument("--campo", help="CSV/XLSX de datos estructurales de campo")
    ap.add_argument("--zona", default="")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    _from_run(a.carpeta, a.campo, a.zona)
