"""
Atributos geológicos de cada lineamiento.

  longitud_m, azimut (0-180, N->E), rumbo (N45E), sector (NE-SW...)
  familia    : familias direccionales automáticas (picos de la roseta)
  tipo       : evidencia dominante (valle, cresta, escarpe, borde)
  n_evid     : cuántas evidencias independientes lo respaldan
  fuerza     : respuesta media del filtro de rectitud
  soporte    : fracción de la traza con evidencia local
  drenaje    : fracción de la traza sobre la red de drenaje
  confianza  : 0-1 combinando lo anterior, y clase Alta/Media/Baja
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .vectorize import sample_mean

SECTORES = ["N-S", "NNE-SSW", "NE-SW", "ENE-WSW", "E-W", "WNW-ESE", "NW-SE", "NNW-SSE"]
TIPO_LABEL = {"valles": "valle", "crestas": "cresta", "escarpes": "escarpe", "bordes": "borde"}


def geo_azimuth(x0, y0, x1, y1):
    return np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 180.0


def rumbo_txt(az: float) -> str:
    az = float(az) % 180.0
    return f"N{az:02.0f}E" if az <= 90 else f"N{180 - az:02.0f}W"


def sector(az: float) -> str:
    return SECTORES[int(((az + 11.25) % 180) // 22.5)]


def find_families(az: np.ndarray, w: np.ndarray, max_families: int = 6, kappa: float = 25.0,
                  min_rel: float = 0.2, min_sep: float = 15.0):
    """Picos de la densidad circular (ángulo doble, núcleo von Mises)."""
    if len(az) == 0:
        return np.zeros(0, int), []
    grid = np.arange(0, 180, 1.0)
    d = np.deg2rad(2 * (grid[:, None] - az[None, :]))
    dens = (np.exp(kappa * (np.cos(d) - 1)) * w[None, :]).sum(1)
    peaks = [i for i in range(180) if dens[i] >= dens[i - 1] and dens[i] >= dens[(i + 1) % 180]]
    peaks = sorted(peaks, key=lambda i: -dens[i])
    keep = []
    for i in peaks:
        if dens[i] < min_rel * dens.max():
            break
        if all(min(abs(grid[i] - grid[j]), 180 - abs(grid[i] - grid[j])) >= min_sep for j in keep):
            keep.append(i)
        if len(keep) >= max_families:
            break
    centers = grid[keep]
    diff = np.abs(az[:, None] - centers[None, :]) % 180
    diff = np.minimum(diff, 180 - diff)
    lab = diff.argmin(1)
    # ordena familias por longitud acumulada
    tot = np.array([w[lab == k].sum() for k in range(len(centers))])
    order = np.argsort(-tot)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    fams = []
    for rank, k in enumerate(order):
        m = lab == k
        # media circular ponderada de los miembros
        a2 = np.deg2rad(2 * az[m])
        mu = (np.degrees(np.arctan2((w[m] * np.sin(a2)).sum(), (w[m] * np.cos(a2)).sum())) / 2) % 180
        fams.append({"familia": f"F{rank + 1}", "azimut_medio": float(mu), "rumbo": rumbo_txt(mu),
                     "n": int(m.sum()), "longitud_km": float(w[m].sum() / 1000),
                     "pct_longitud": float(100 * w[m].sum() / max(w.sum(), 1e-9))})
    return remap[lab], fams


def drainage_distance(grid, area_m2: float, workdir=None, log=print):
    """Distancia (px) a la red de drenaje D8 y máscara de cauces.
    Implementación propia (drainage.py): sin numba, funciona en el ejecutable."""
    from .drainage import stream_network
    try:
        streams, _ = stream_network(grid.z, grid.valid, grid.res, area_m2)
        return ndi.distance_transform_edt(~streams).astype(np.float32), streams
    except Exception as e:
        log(f"    (drenaje omitido: {e})")
        return None, None


def build_attributes(segs_px: np.ndarray, grid, P, model: str, strength_map=None,
                     support=None, evidences: dict | None = None, drain_dist=None):
    """Devuelve (lista de dicts de atributos, lista de familias)."""
    n = len(segs_px)
    if n == 0:
        return [], []
    X0, Y0 = grid.px2geo(segs_px[:, 0], segs_px[:, 1])
    X1, Y1 = grid.px2geo(segs_px[:, 2], segs_px[:, 3])
    L = np.hypot(X1 - X0, Y1 - Y0)
    az = geo_azimuth(X0, Y0, X1, Y1)
    lab, fams = find_families(az, L, P.max_families)
    fam_rumbo = {i: f["rumbo"] for i, f in enumerate(fams)}

    fuerza = sample_mean(segs_px, strength_map) if strength_map is not None else np.full(n, np.nan)
    sup = support if support is not None and len(support) == n else np.full(n, np.nan)

    ev_means = {}
    if evidences:
        for k, (S, _) in evidences.items():
            ev_means[k] = sample_mean(segs_px, S)
    if ev_means:
        keys = list(ev_means)
        M = np.column_stack([ev_means[k] for k in keys])
        tipo = [TIPO_LABEL.get(keys[i], keys[i]) for i in M.argmax(1)]
        n_evid = (M >= 0.35).sum(1)
    else:
        tipo = ["borde_sombreado"] * n
        n_evid = np.zeros(n, int)

    drain = np.full(n, np.nan)
    if drain_dist is not None:
        from .vectorize import sample_points
        owner, x, y, _ = sample_points(segs_px, 1.0)
        d = ndi.map_coordinates(drain_dist, [y, x], order=0, mode="nearest")
        inside = d <= P.drainage_buffer_m / grid.res
        drain = np.bincount(owner, weights=inside, minlength=n) / np.bincount(owner, minlength=n)

    # confianza
    Lmin = max(P.min_length_m, 1.0)
    s_len = np.clip(np.log2(np.maximum(L, 1) / Lmin) / 3.0, 0, 1)
    s_str = np.clip(np.nan_to_num(fuerza, nan=0.5), 0, 1)
    s_sup = np.clip(np.nan_to_num(sup, nan=0.7), 0, 1)
    s_ev = np.clip(n_evid / max(len(ev_means), 1), 0, 1) if ev_means else np.full(n, 0.5)
    conf = 0.35 * s_str + 0.25 * s_len + 0.2 * s_sup + 0.2 * s_ev

    rows = []
    for i in range(n):
        c = float(conf[i])
        rows.append({
            "modelo": model,
            "longitud_m": round(float(L[i]), 1),
            "azimut": round(float(az[i]), 1),
            "rumbo": rumbo_txt(az[i]),
            "sector": sector(az[i]),
            "familia": f"F{lab[i] + 1}",
            "fam_rumbo": fam_rumbo.get(int(lab[i]), ""),
            "tipo": tipo[i],
            "n_evid": int(n_evid[i]),
            "fuerza": round(float(fuerza[i]), 3) if np.isfinite(fuerza[i]) else None,
            "soporte": round(float(sup[i]), 3) if np.isfinite(sup[i]) else None,
            "drenaje": round(float(drain[i]), 3) if np.isfinite(drain[i]) else None,
            "sigue_dren": bool(drain[i] >= 0.6) if np.isfinite(drain[i]) else None,
            "confianza": round(c, 3),
            "clase_conf": "Alta" if c >= 0.6 else ("Media" if c >= 0.42 else "Baja"),
        })
    return rows, fams
