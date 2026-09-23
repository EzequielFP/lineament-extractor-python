"""
Esqueleto binario -> segmentos rectos.

  trace_skeleton   : caminos ordenados de píxeles (rompe en bifurcaciones)
  link_segments    : une tramos colineales (hueco, ángulo y desfase lateral)
                     con índice espacial; también absorbe duplicados paralelos
  trim_by_support  : recorta extremos sin evidencia y mide el soporte

Segmentos: array (N, 4) float64 = x0, y0, x1, y1 en píxeles (col, fila).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

_NB8 = np.ones((3, 3), dtype=np.uint8)
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def trace_skeleton(skel: np.ndarray, min_len: int = 3) -> list[np.ndarray]:
    """Devuelve caminos como arrays (K, 2) de (x, y). Se eliminan los píxeles de
    bifurcación para obtener componentes simples; el enlace posterior vuelve a
    unir lo que sea colineal."""
    skel = skel.astype(bool)
    nb = ndi.convolve(skel.astype(np.uint8), _NB8, mode="constant") - skel
    simple = skel & (nb <= 2)
    lab, n = ndi.label(simple, structure=_NB8)
    if n == 0:
        return []
    paths = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl] == i
        rr, cc = np.nonzero(sub)
        if rr.size < min_len:
            continue
        r0, c0 = sl[0].start, sl[1].start
        if rr.size == 2:
            paths.append(np.column_stack([cc + c0, rr + r0]).astype(np.float64))
            continue
        # vecinos dentro del componente
        pts = set(zip(rr.tolist(), cc.tolist()))
        deg = {p: sum((p[0] + dr, p[1] + dc) in pts for dr, dc in _OFFS) for p in pts}
        ends = [p for p, d in deg.items() if d <= 1]
        start = ends[0] if ends else next(iter(pts))
        order = [start]
        seen = {start}
        cur = start
        while True:
            nxt = None
            # prioriza vecinos 4-conectados para no saltarse píxeles
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                q = (cur[0] + dr, cur[1] + dc)
                if q in pts and q not in seen:
                    nxt = q
                    break
            if nxt is None:
                break
            order.append(nxt)
            seen.add(nxt)
            cur = nxt
        o = np.array(order, dtype=np.float64)
        paths.append(np.column_stack([o[:, 1] + c0, o[:, 0] + r0]))
    return paths


# ---------------------------------------------------------------------------
def seg_lengths(S: np.ndarray) -> np.ndarray:
    return np.hypot(S[:, 2] - S[:, 0], S[:, 3] - S[:, 1])


def seg_angles(S: np.ndarray) -> np.ndarray:
    """Ángulo de la línea en [0, pi) (marco de imagen)."""
    return np.mod(np.arctan2(S[:, 3] - S[:, 1], S[:, 2] - S[:, 0]), np.pi)


def _angdiff(a, b):
    d = np.abs(a - b) % np.pi
    return np.minimum(d, np.pi - d)


class _DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _fit_group(segs: np.ndarray):
    """Recta de mínimos cuadrados totales ponderada por longitud."""
    L = seg_lengths(segs)
    pts = np.vstack([segs[:, :2], segs[:, 2:]])
    w = np.concatenate([L, L]) + 1e-6
    c = (pts * w[:, None]).sum(0) / w.sum()
    # dirección: media de ángulo doble ponderada (robusta para pocos puntos)
    ang = seg_angles(segs)
    d2 = np.array([(L * np.cos(2 * ang)).sum(), (L * np.sin(2 * ang)).sum()])
    th = 0.5 * np.arctan2(d2[1], d2[0])
    u = np.array([np.cos(th), np.sin(th)])
    t = (pts - c) @ u
    lat = np.abs((pts - c) @ np.array([-u[1], u[0]]))
    a, b = c + t.min() * u, c + t.max() * u
    return np.array([a[0], a[1], b[0], b[1]]), float(lat.max())


def link_segments(segs: np.ndarray, gap_px: float, angle_deg: float, lateral_px: float,
                  max_iter: int = 6, gap_rel: float = 0.0, lateral_rel: float = 0.0) -> np.ndarray:
    """Une iterativamente segmentos colineales cercanos (incluye solapados).

    gap_rel / lateral_rel permiten huecos y desfases proporcionales a la
    longitud del tramo más corto: dos trazas largas y alineadas separadas por
    un tramo erosionado se reconocen como la misma estructura."""
    if len(segs) < 2:
        return segs
    ang_tol = np.deg2rad(angle_deg)
    for _ in range(max_iter):
        n = len(segs)
        L = seg_lengths(segs)
        th = seg_angles(segs)
        # puntos de muestreo a lo largo de cada segmento para búsqueda espacial
        step = max(gap_px / 2.0, 1.0)
        k = np.maximum(np.ceil(L / step).astype(int), 1) + 1
        owner = np.repeat(np.arange(n), k)
        t = np.concatenate([np.linspace(0, 1, kk) for kk in k])
        P = segs[owner, :2] + t[:, None] * (segs[owner, 2:] - segs[owner, :2])
        reach = gap_px + lateral_px
        if gap_rel > 0:
            reach = max(reach, gap_rel * float(np.percentile(L, 99)) + lateral_px)
        pairs = cKDTree(P).query_pairs(reach, output_type="ndarray")
        if len(pairs) == 0:
            break
        a, b = owner[pairs[:, 0]], owner[pairs[:, 1]]
        m = a != b
        a, b = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        if len(a) == 0:
            break
        ab = np.unique(np.column_stack([a, b]), axis=0)
        a, b = ab[:, 0], ab[:, 1]
        ok = _angdiff(th[a], th[b]) <= ang_tol
        a, b = a[ok], b[ok]
        # desfase lateral y hueco longitudinal en el marco del segmento más largo
        ref = np.where(L[a] >= L[b], a, b)
        oth = np.where(ref == a, b, a)
        u = np.column_stack([np.cos(th[ref]), np.sin(th[ref])])
        nrm = np.column_stack([-u[:, 1], u[:, 0]])
        p0 = segs[ref, :2]
        q0, q1 = segs[oth, :2] - p0, segs[oth, 2:] - p0
        lat = np.maximum(np.abs((q0 * nrm).sum(1)), np.abs((q1 * nrm).sum(1)))
        tq = np.column_stack([(q0 * u).sum(1), (q1 * u).sum(1)])
        tr = np.column_stack([np.zeros(len(ref)), ((segs[ref, 2:] - p0) * u).sum(1)])
        gap = np.maximum(0, np.maximum(tq.min(1) - tr.max(1), tr.min(1) - tq.max(1)))
        short = np.minimum(L[a], L[b])
        gmax = np.maximum(gap_px, gap_rel * short)
        lmax = np.maximum(lateral_px, lateral_rel * (L[a] + L[b]))
        ok = (lat <= lmax) & (gap <= gmax)
        a, b = a[ok], b[ok]
        if len(a) == 0:
            break
        dsu = _DSU(n)
        for i, j in zip(a, b):
            dsu.union(i, j)
        roots = np.array([dsu.find(i) for i in range(n)])
        new = []
        merged = False
        for r in np.unique(roots):
            members = np.nonzero(roots == r)[0]
            if len(members) == 1:
                new.append(segs[members[0]])
                continue
            fit, dev = _fit_group(segs[members])
            span = np.hypot(fit[2] - fit[0], fit[3] - fit[1])
            if dev <= 1.5 * max(lateral_px, lateral_rel * span):
                new.append(fit)
                merged = True
            else:  # grupo quebrado (cadena en zigzag): se conservan por separado
                new.extend(segs[members])
        segs = np.array(new).reshape(-1, 4)
        if not merged:
            break
    return segs


# ---------------------------------------------------------------------------
def sample_points(segs: np.ndarray, step: float = 1.0):
    """Puntos equiespaciados sobre cada segmento: (owner, x, y)."""
    L = seg_lengths(segs)
    k = np.maximum(np.ceil(L / step).astype(int), 1) + 1
    owner = np.repeat(np.arange(len(segs)), k)
    t = np.concatenate([np.linspace(0, 1, kk) for kk in k]) if len(k) else np.zeros(0)
    xy = segs[owner, :2] + t[:, None] * (segs[owner, 2:] - segs[owner, :2])
    return owner, xy[:, 0], xy[:, 1], t


def sample_mean(segs: np.ndarray, raster: np.ndarray, step: float = 1.0) -> np.ndarray:
    if len(segs) == 0:
        return np.zeros(0)
    owner, x, y, _ = sample_points(segs, step)
    v = ndi.map_coordinates(raster, [y, x], order=1, mode="nearest")
    s = np.bincount(owner, weights=v, minlength=len(segs))
    c = np.bincount(owner, minlength=len(segs))
    return s / np.maximum(c, 1)


def trim_by_support(segs: np.ndarray, S: np.ndarray, thr: float, min_len_px: float,
                    min_support: float, max_nudge_px: float = 2.0):
    """Recorta los extremos sin evidencia; descarta líneas con poco soporte.
    La evidencia se busca en una franja de +/- max_nudge_px (tolerancia a
    pequeñas desviaciones del trazo recto)."""
    if len(segs) == 0:
        return segs, np.zeros(0)
    Sb = ndi.maximum_filter(S, size=int(2 * max_nudge_px + 1))
    out, sup = [], []
    for sg in segs:
        L = np.hypot(sg[2] - sg[0], sg[3] - sg[1])
        n = max(int(np.ceil(L)), 1) + 1
        t = np.linspace(0, 1, n)
        x = sg[0] + t * (sg[2] - sg[0])
        y = sg[1] + t * (sg[3] - sg[1])
        v = ndi.map_coordinates(Sb, [y, x], order=1, mode="nearest") >= thr
        if not v.any():
            continue
        i0, i1 = np.argmax(v), n - 1 - np.argmax(v[::-1])
        if (i1 - i0) * L / (n - 1) < min_len_px:
            continue
        frac = v[i0:i1 + 1].mean()
        if frac < min_support:
            continue
        a, b = t[i0], t[i1]
        out.append((sg[0] + a * (sg[2] - sg[0]), sg[1] + a * (sg[3] - sg[1]),
                    sg[0] + b * (sg[2] - sg[0]), sg[1] + b * (sg[3] - sg[1])))
        sup.append(frac)
    return np.array(out, dtype=np.float64).reshape(-1, 4), np.array(sup)
