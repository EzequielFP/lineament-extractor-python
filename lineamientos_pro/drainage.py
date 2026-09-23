"""
Red de drenaje D8 sin dependencias compiladas en tiempo de ejecución (numba).

  1. Relleno de depresiones por reconstrucción morfológica (erosión).
  2. Zonas planas: se les impone una pendiente mínima hacia su salida
     (distancia a la celda más cercana que sí drena), como en Garbrecht y
     Martz (1997) simplificado.
  3. Dirección D8 de máxima pendiente.
  4. Acumulación de flujo con ordenamiento topológico vectorizado (Kahn):
     cada iteración propaga el flujo de todas las celdas cuyo aporte ya está
     completo.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import reconstruction

_OFF = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_depressions(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    seed = z.copy()
    inner = np.zeros_like(valid)
    inner[1:-1, 1:-1] = True
    # el borde (y el contorno de los datos válidos) actúa como salida
    edge = ~inner | ~ndi.binary_erosion(valid, np.ones((3, 3)), border_value=0)
    seed[~edge] = z.max()
    return reconstruction(seed, z, method="erosion").astype(np.float64)


def _shift(a, dr, dc, fill):
    out = np.full_like(a, fill)
    H, W = a.shape
    rs, re = max(dr, 0), H + min(dr, 0)
    cs, ce = max(dc, 0), W + min(dc, 0)
    out[rs - dr:re - dr, cs - dc:ce - dc] = a[rs:re, cs:ce]
    return out   # out[r, c] = a[r + dr, c + dc]


def flow_receivers(zf: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Índice plano de la celda receptora (-1 = salida/sumidero)."""
    H, W = zf.shape
    big = 1e30
    zv = np.where(valid, zf, big)
    # celdas planas (sin vecino más bajo): pendiente artificial hacia su SALIDA,
    # es decir, hacia celdas de la misma cota que sí tienen un vecino más bajo
    # (no hacia el borde alto de la zona plana, que crearía sumideros falsos)
    lower = np.zeros(zf.shape, bool)
    for dr, dc in _OFF:
        lower |= _shift(zv, dr, dc, big) < zv
    flat = valid & ~lower
    flat[[0, -1], :] = False
    flat[:, [0, -1]] = False
    if flat.any():
        eq_flat = np.zeros(zf.shape, bool)
        for dr, dc in _OFF:
            eq_flat |= _shift(flat, dr, dc, False) & (np.abs(_shift(zv, dr, dc, big) - zv) < 1e-9)
        outlet = lower & eq_flat & valid
        d = ndi.distance_transform_edt(~outlet)
        zv = np.where(flat, zv + 1e-6 * d, zv)
    best = np.zeros(zf.shape)
    rec = np.full(zf.shape, -1, np.int64)
    rr, cc = np.mgrid[0:H, 0:W]
    for dr, dc in _OFF:
        drop = (zv - _shift(zv, dr, dc, big)) / np.hypot(dr, dc)
        m = (drop > best) & valid
        best[m] = drop[m]
        rec[m] = ((rr + dr) * W + (cc + dc))[m]
    return rec


def accumulation(rec: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Número de celdas que drenan a cada celda (incluida ella)."""
    n = rec.size
    r = rec.ravel()
    acc = valid.ravel().astype(np.float64)
    has = r >= 0
    indeg = np.bincount(r[has], minlength=n)
    front = np.nonzero((indeg == 0) & valid.ravel())[0]
    while front.size:
        tgt = r[front]
        m = tgt >= 0
        src, tgt = front[m], tgt[m]
        np.add.at(acc, tgt, acc[src])
        dec = np.bincount(tgt, minlength=n)
        touched = np.unique(tgt)
        indeg[touched] -= dec[touched]
        front = touched[indeg[touched] == 0]
    return acc.reshape(rec.shape)


def stream_network(z: np.ndarray, valid: np.ndarray, res: float, area_m2: float):
    zf = fill_depressions(z.astype(np.float64), valid)
    rec = flow_receivers(zf, valid)
    acc = accumulation(rec, valid)
    return (acc * res ** 2 >= area_m2) & valid, acc
