"""
Comparación de lineamientos contra una referencia (mapa de campo,
interpretación experta, o salida de otro software como PCI/Catalyst).

Métricas por LONGITUD (no por conteo, para no premiar la fragmentación):
  precisión = longitud detectada que coincide con la referencia / longitud detectada
  recall    = longitud de referencia recuperada / longitud de referencia
  F1        = media armónica
Coincidir = hay una línea de la otra capa a menos de `buffer` y con
diferencia de rumbo menor que `angle_deg`.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import ks_2samp


def _samples(segs: np.ndarray, step: float):
    if len(segs) == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    d = segs[:, 2:] - segs[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    ang = np.mod(np.arctan2(d[:, 1], d[:, 0]), np.pi)
    k = np.maximum(np.ceil(L / step).astype(int), 1)
    owner = np.repeat(np.arange(len(segs)), k)
    t = np.concatenate([(np.arange(kk) + 0.5) / kk for kk in k])
    P = segs[owner, :2] + t[:, None] * d[owner]
    w = (L / k)[owner]
    return P, ang[owner], w


def _matched(Pa, Aa, Pb, Ab, buffer, ang_tol, kmax=16):
    if len(Pa) == 0 or len(Pb) == 0:
        return np.zeros(len(Pa), bool)
    tree = cKDTree(Pb)
    dist, idx = tree.query(Pa, k=min(kmax, len(Pb)), distance_upper_bound=buffer)
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    ok = np.isfinite(dist)
    idx = np.where(ok, idx, 0)
    d = np.abs(Aa[:, None] - Ab[idx]) % np.pi
    d = np.minimum(d, np.pi - d)
    return (ok & (d <= ang_tol)).any(1)


def match_scores(pred: np.ndarray, ref: np.ndarray, buffer: float, angle_deg: float,
                 step: float | None = None) -> dict:
    """pred/ref: arrays (N,4) de segmentos en las mismas unidades que buffer."""
    step = step or max(buffer / 3.0, 1e-6)
    tol = np.deg2rad(angle_deg)
    Pp, Ap, Wp = _samples(np.asarray(pred, float).reshape(-1, 4), step)
    Pr, Ar, Wr = _samples(np.asarray(ref, float).reshape(-1, 4), step)
    mp = _matched(Pp, Ap, Pr, Ar, buffer, tol)
    mr = _matched(Pr, Ar, Pp, Ap, buffer, tol)
    prec = float((Wp * mp).sum() / Wp.sum()) if Wp.sum() > 0 else 0.0
    rec = float((Wr * mr).sum() / Wr.sum()) if Wr.sum() > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1,
            "len_pred": float(Wp.sum()), "len_ref": float(Wr.sum()),
            "buffer": buffer, "angle_deg": angle_deg}


def rose_hist(segs: np.ndarray, n_bins: int = 18, geo: bool = True) -> np.ndarray:
    """Histograma de rumbos ponderado por longitud (normalizado)."""
    d = segs[:, 2:] - segs[:, :2]
    L = np.hypot(d[:, 0], d[:, 1])
    az = np.degrees(np.arctan2(d[:, 0], d[:, 1] if geo else -d[:, 1])) % 180
    h, _ = np.histogram(az, bins=np.linspace(0, 180, n_bins + 1), weights=L)
    return h / h.sum() if h.sum() > 0 else h


def compare(pred_segs: np.ndarray, ref_segs: np.ndarray, buffer: float, angle_deg: float) -> dict:
    """Informe completo en coordenadas del mapa (metros)."""
    out = match_scores(pred_segs, ref_segs, buffer, angle_deg)
    hp, hr = rose_hist(pred_segs), rose_hist(ref_segs)
    out["rose_correlation"] = float(np.corrcoef(hp, hr)[0, 1]) if hp.sum() and hr.sum() else float("nan")
    Lp = np.hypot(*(pred_segs[:, 2:] - pred_segs[:, :2]).T)
    Lr = np.hypot(*(ref_segs[:, 2:] - ref_segs[:, :2]).T)
    if len(Lp) and len(Lr):
        ks = ks_2samp(Lp, Lr)
        out.update(ks_length_stat=float(ks.statistic),
                   median_len_pred=float(np.median(Lp)), median_len_ref=float(np.median(Lr)))
    out.update(n_pred=int(len(pred_segs)), n_ref=int(len(ref_segs)))
    return out


def gdf_to_segments(gdf) -> np.ndarray:
    """Descompone (Multi)LineStrings en segmentos (N,4)."""
    segs = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        parts = getattr(geom, "geoms", [geom])
        for part in parts:
            c = np.asarray(part.coords)[:, :2]
            if len(c) >= 2:
                segs.append(np.column_stack([c[:-1], c[1:]]))
    return np.vstack(segs) if segs else np.zeros((0, 4))


def read_vector(path):
    """Lee un vector; admite shapefiles sin nombre (".shp", como los que deja
    PCI Geomatica), que GDAL no reconoce, copiándolos a un nombre temporal."""
    import shutil
    import tempfile
    from pathlib import Path

    import geopandas as gpd
    p = Path(path)
    if p.name.lower() == ".shp":       # Path(".shp").suffix es ""
        tmp = Path(tempfile.mkdtemp(prefix="linpro_"))
        for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            src = p.with_name(ext)
            if src.exists():
                shutil.copy(src, tmp / f"referencia{ext}")
        try:
            return gpd.read_file(tmp / "referencia.shp")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return gpd.read_file(p)
