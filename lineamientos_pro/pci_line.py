"""
Réplica del algoritmo LINE de PCI Geomatica (ayuda de Geomatica 2018).

  1. Canny: filtro gaussiano de radio RADI, gradiente y supresión de
     no-máximos -> imagen de intensidad de borde.
  2. Umbral GTHR (0-255) -> imagen binaria de bordes.
  3. Extracción de curvas: adelgazamiento, secuencias de píxeles, descarte de
     curvas con menos de LTHR píxeles, ajuste de polilínea con error máximo
     FTHR, partición donde el ángulo entre tramos supera ATHR (las piezas
     menores que LTHR se descartan), y enlace de
     pares de polilíneas cuyos tramos finales se enfrentan, con ángulo < ATHR
     y extremos a menos de DTHR píxeles.
  Entradas de 16/32 bits se escalan antes a 8 bits; las de 8 bits se usan tal cual.

PCI no publica la relación radio/sigma ni la escala del gradiente. Ambas
(PCI_SIGMA_DIV, PCI_GAIN) se calibraron contra una corrida real de PCI
Geomatica con parámetros por defecto sobre Ejemplos/0.tif (calibrate_pci.py).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

from .linefilter import nonmax_suppression
from .vectorize import trace_skeleton

# Calibrado contra PCI Geomatica 2018, LINE por defecto sobre 0.tif (2264 polilíneas):
# la réplica da 2015 polilíneas, mediana 45 m (PCI 47.7 m) y F1 0.68 con 1.5 px / 20°.
PCI_SIGMA_DIV = 2.5     # sigma = RADI / PCI_SIGMA_DIV
PCI_GAIN = 22.0         # intensidad de borde = PCI_GAIN * |gradiente| (niveles/píxel)


def to_uint8(img: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """8 bits: sin cambios. Otros tipos: estiramiento 0.5-99.5 % (PCI usa un
    escalado no lineal no documentado)."""
    if img.dtype == np.uint8:
        return img
    v = img[valid] if valid is not None else img.ravel()
    if v.size and np.all(np.mod(v, 1) == 0) and v.min() >= 0 and v.max() <= 255:
        return img.astype(np.uint8)          # imagen de 8 bits leída como float
    lo, hi = np.percentile(v, [0.5, 99.5]) if v.size else (0, 1)
    out = np.clip((img - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255)
    return out.astype(np.uint8)


def edge_strength(img8: np.ndarray, RADI: int, sigma_div: float | None = None,
                  gain: float | None = None):
    sigma_div = sigma_div or PCI_SIGMA_DIV
    gain = PCI_GAIN if gain is None else gain
    f = img8.astype(np.float32)
    if RADI > 0:
        f = cv2.GaussianBlur(f, (0, 0), max(RADI / sigma_div, 0.5), borderType=cv2.BORDER_REFLECT)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = np.hypot(gx, gy)
    theta_line = np.mod(np.arctan2(gy, gx) + np.pi / 2, np.pi).astype(np.float32)
    keep = nonmax_suppression(mag, theta_line)
    strength = np.clip(mag * gain, 0, 255)
    strength[~keep] = 0
    return strength.astype(np.float32)


def split_by_angle(poly: np.ndarray, ATHR: float) -> list[np.ndarray]:
    """Parte la polilínea donde el giro entre tramos consecutivos supera ATHR."""
    if len(poly) < 3:
        return [poly]
    d = np.diff(poly, axis=0)
    n = np.hypot(d[:, 0], d[:, 1])
    cosang = (d[:-1] * d[1:]).sum(1) / np.maximum(n[:-1] * n[1:], 1e-9)
    cut = np.nonzero(cosang < np.cos(np.deg2rad(ATHR)))[0] + 1   # índices de vértice
    parts, start = [], 0
    for c in cut:
        parts.append(poly[start:c + 1])
        start = c
    parts.append(poly[start:])
    return parts


def _link_polylines(polys: list[np.ndarray], DTHR: float, ATHR: float, max_iter: int = 20):
    """Enlaza extremos a menos de DTHR cuyos tramos finales se enfrentan y
    tienen orientación similar (ángulo < ATHR)."""
    cth = np.cos(np.deg2rad(ATHR))
    for _ in range(max_iter):
        if len(polys) < 2:
            break
        ends, dirs, owner, which = [], [], [], []
        for i, p in enumerate(polys):
            for w, (a, b) in enumerate(((p[0], p[1]), (p[-1], p[-2]))):
                d = a - b
                n = np.hypot(*d)
                ends.append(a)
                dirs.append(d / n if n > 0 else d)   # dirección de salida
                owner.append(i)
                which.append(w)
        ends, dirs = np.array(ends), np.array(dirs)
        owner, which = np.array(owner), np.array(which)
        pairs = cKDTree(ends).query_pairs(DTHR, output_type="ndarray")
        if len(pairs) == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        m = owner[i] != owner[j]
        i, j = i[m], j[m]
        # 1) tramos finales con orientación similar (salidas opuestas)
        ok = -(dirs[i] * dirs[j]).sum(1) >= cth
        # 2) se enfrentan: el conector sigue la salida de cada extremo
        con = ends[j] - ends[i]
        cl = np.hypot(con[:, 0], con[:, 1])
        u = con / np.maximum(cl, 1e-9)[:, None]
        face = ((dirs[i] * u).sum(1) >= cth) & ((-dirs[j] * u).sum(1) >= cth)
        ok &= face | (cl < 1.5)
        i, j = i[ok], j[ok]
        if len(i) == 0:
            break
        order = np.argsort(np.hypot(*(ends[i] - ends[j]).T))
        used = np.zeros(len(polys), bool)
        new, merged = [], False
        for k in order:
            a, b = i[k], j[k]
            pa, pb = owner[a], owner[b]
            if used[pa] or used[pb]:
                continue
            A = polys[pa] if which[a] == 1 else polys[pa][::-1]   # A termina en a
            B = polys[pb] if which[b] == 0 else polys[pb][::-1]   # B empieza en b
            new.append(np.vstack([A, B]))
            used[[pa, pb]] = True
            merged = True
        new.extend(p for k, p in enumerate(polys) if not used[k])
        polys = new
        if not merged:
            break
    return polys


def pci_line(img8: np.ndarray, valid: np.ndarray, RADI=10, GTHR=100, LTHR=30, FTHR=3,
             ATHR=30, DTHR=20, sigma_div=None, gain=None):
    """Devuelve (polilíneas [(K,2) x,y en píxeles], intensidad de borde)."""
    strength = edge_strength(img8, RADI, sigma_div, gain)
    strength[~valid] = 0
    skel = skeletonize(strength >= max(GTHR, 1e-6))
    curves = [c for c in trace_skeleton(skel, min_len=2) if len(c) >= LTHR]
    polys = []
    for c in curves:
        ap = approximate_polygon(c, tolerance=max(FTHR, 0.5))
        # las piezas resultantes también deben superar LTHR (PCI nunca entrega
        # polilíneas más cortas que LTHR)
        polys.extend(p for p in split_by_angle(ap, ATHR)
                     if len(p) >= 2 and np.hypot(*np.diff(p, axis=0).T).sum() >= LTHR)
    return _link_polylines(polys, DTHR, ATHR), strength


def polylines_to_segments(polys) -> np.ndarray:
    segs = [(a[0], a[1], b[0], b[1]) for p in polys for a, b in zip(p[:-1], p[1:])]
    return np.array(segs, dtype=np.float64).reshape(-1, 4)
