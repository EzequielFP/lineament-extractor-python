"""
Validación cuantitativa contra un shapefile de referencia (Catalyst).

Métricas:
  - Distribución de longitudes (estadísticos + Kolmogorov-Smirnov)
  - Distribución de azimuts (comparación de histogramas / roseta)
  - Buffer matching: % de lineamientos Catalyst con contraparte Python
  - Densidad direccional (roseta)

Genera un reporte de texto y un PNG con rosetas superpuestas.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


def _compute_azimuth(line) -> float:
    """Azimut geográfico de una LineString en [0, 180)."""
    coords = list(line.coords)
    x0, y0 = coords[0]
    x1, y1 = coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    return (np.degrees(np.arctan2(dx, dy))) % 180.0


def _prepare_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Asegura columnas length_m y azimuth_deg."""
    g = gdf.copy()
    if "length_m" not in g.columns:
        g["length_m"] = g.geometry.length
    if "azimuth_deg" not in g.columns:
        g["azimuth_deg"] = g.geometry.apply(_compute_azimuth)
    return g


# ───────────────────────────────────────────────────────────────────
# Métricas
# ───────────────────────────────────────────────────────────────────
def length_statistics(py: gpd.GeoDataFrame, ref: gpd.GeoDataFrame) -> dict:
    py, ref = _prepare_gdf(py), _prepare_gdf(ref)
    ks_stat, ks_p = ks_2samp(py["length_m"], ref["length_m"])
    return {
        "n_python": int(len(py)),
        "n_catalyst": int(len(ref)),
        "mean_py_m": float(py["length_m"].mean()),
        "mean_ref_m": float(ref["length_m"].mean()),
        "median_py_m": float(py["length_m"].median()),
        "median_ref_m": float(ref["length_m"].median()),
        "total_length_py_m": float(py["length_m"].sum()),
        "total_length_ref_m": float(ref["length_m"].sum()),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def azimuth_histogram_correlation(
    py: gpd.GeoDataFrame, ref: gpd.GeoDataFrame, n_bins: int = 18
) -> dict:
    py, ref = _prepare_gdf(py), _prepare_gdf(ref)
    bins = np.linspace(0, 180, n_bins + 1)
    h_py, _ = np.histogram(py["azimuth_deg"], bins=bins, weights=py["length_m"])
    h_ref, _ = np.histogram(ref["azimuth_deg"], bins=bins, weights=ref["length_m"])
    h_py = h_py / h_py.sum() if h_py.sum() > 0 else h_py
    h_ref = h_ref / h_ref.sum() if h_ref.sum() > 0 else h_ref
    corr = float(np.corrcoef(h_py, h_ref)[0, 1]) if h_py.sum() > 0 and h_ref.sum() > 0 else np.nan
    return {
        "azimuth_histogram_correlation": corr,
        "n_bins": n_bins,
    }


def buffer_matching(
    py: gpd.GeoDataFrame,
    ref: gpd.GeoDataFrame,
    buffer_m: float,
    angle_tol_deg: float,
) -> dict:
    """% de lineamientos de referencia que tienen al menos un match Python dentro
    del buffer y con diferencia angular < tolerancia."""
    py, ref = _prepare_gdf(py), _prepare_gdf(ref)

    py_buffered = py.copy()
    py_buffered["geom_buf"] = py.geometry.buffer(buffer_m)

    # Spatial index sobre buffers
    sindex = gpd.GeoDataFrame(py_buffered, geometry="geom_buf", crs=py.crs).sindex

    matched = 0
    for _, row in ref.iterrows():
        candidates_idx = list(sindex.intersection(row.geometry.bounds))
        if not candidates_idx:
            continue
        for idx in candidates_idx:
            py_row = py_buffered.iloc[idx]
            if py_row["geom_buf"].intersects(row.geometry):
                ang_diff = abs(py_row["azimuth_deg"] - row["azimuth_deg"]) % 180.0
                ang_diff = min(ang_diff, 180.0 - ang_diff)
                if ang_diff < angle_tol_deg:
                    matched += 1
                    break

    recall = matched / len(ref) if len(ref) > 0 else 0.0
    return {
        "matched_ref": int(matched),
        "total_ref": int(len(ref)),
        "recall": float(recall),
        "buffer_m": buffer_m,
        "angle_tol_deg": angle_tol_deg,
    }


# ───────────────────────────────────────────────────────────────────
# Visualización: roseta comparada
# ───────────────────────────────────────────────────────────────────
def plot_rose_comparison(
    py: gpd.GeoDataFrame,
    ref: gpd.GeoDataFrame,
    output_path: Path,
    n_bins: int = 18,
    title: str = "Rosa de lineamientos — Python vs Catalyst",
):
    py, ref = _prepare_gdf(py), _prepare_gdf(ref)
    bins = np.linspace(0, 180, n_bins + 1)

    h_py, _ = np.histogram(py["azimuth_deg"], bins=bins, weights=py["length_m"])
    h_ref, _ = np.histogram(ref["azimuth_deg"], bins=bins, weights=ref["length_m"])

    theta = np.deg2rad(bins[:-1] + np.diff(bins) / 2)
    width = np.deg2rad(180 / n_bins)

    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection="polar"), figsize=(12, 6))

    for ax, h, name, color in zip(axes, [h_py, h_ref], ["Python", "Catalyst"],
                                   ["#2E86AB", "#E63946"]):
        # Rosa bidireccional: duplicar simétricamente
        ax.bar(theta, h, width=width, color=color, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.bar(theta + np.pi, h, width=width, color=color, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_title(name, pad=15)
        ax.set_rticks([])

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────────────────────────────────────────────────
# Reporte completo
# ───────────────────────────────────────────────────────────────────
def full_validation_report(
    py: gpd.GeoDataFrame,
    ref: gpd.GeoDataFrame,
    output_dir: Path,
    extractor_name: str,
    buffer_m: float,
    angle_tol_deg: float,
) -> dict:
    results = {}
    results["length"] = length_statistics(py, ref)
    results["azimuth"] = azimuth_histogram_correlation(py, ref)
    results["buffer_matching"] = buffer_matching(py, ref, buffer_m, angle_tol_deg)

    rose_path = output_dir / f"roseta_comparada_{extractor_name}.png"
    plot_rose_comparison(py, ref, rose_path, title=f"Roseta — {extractor_name} vs Catalyst")
    results["rose_plot"] = str(rose_path)

    return results
