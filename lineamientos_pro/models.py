"""
Modelos de extracción. Trabajan en píxeles sobre arrays; la E/S y los
atributos se hacen en pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evidence import gblur, hillshade, robust_scale
from .linefilter import detect_line_pixels, sensitivity_thresholds
from .pci_line import pci_line, polylines_to_segments, to_uint8
from .vectorize import link_segments, trace_skeleton, trim_by_support


@dataclass
class ModelResult:
    name: str
    segs: np.ndarray                         # (N,4) píxeles
    support: np.ndarray | None = None        # fracción con evidencia
    strength: np.ndarray | None = None       # respuesta media de línea
    polylines: list | None = None            # solo PCI
    rasters: dict = field(default_factory=dict)


def fuse_evidence(ev: dict, weights: dict, valid: np.ndarray):
    """Consenso: suma vectorial (ángulo doble) ponderada de las evidencias.
    Donde varias evidencias coinciden en posición y rumbo la señal se refuerza;
    donde discrepan en rumbo se cancela parcialmente."""
    num_c = np.zeros(valid.shape, np.float32)
    num_s = np.zeros(valid.shape, np.float32)
    mag = np.zeros(valid.shape, np.float32)
    for k, (S, T) in ev.items():
        w = float(weights.get(k, 1.0))
        if w <= 0:
            continue
        num_c += w * S * np.cos(2 * T)
        num_s += w * S * np.sin(2 * T)
        mag += w * S
    T = np.mod(0.5 * np.arctan2(num_s, num_c), np.pi).astype(np.float32)
    coh = np.hypot(num_c, num_s)          # = mag si todas concuerdan
    S = robust_scale(0.5 * (coh + mag), valid)
    return S, T


def split_straight(path: np.ndarray, tol: float, rel: float = 0.015) -> list[np.ndarray]:
    """Douglas-Peucker con tolerancia relativa: tramos largos pueden ondular
    un poco más (max(tol, rel * cuerda)) sin romperse."""
    out, stack = [], [(0, len(path) - 1)]
    while stack:
        i, j = stack.pop()
        a, b = path[i], path[j]
        d = b - a
        L = np.hypot(*d)
        if j - i < 2 or L == 0:
            out.append((i, j))
            continue
        seg = path[i:j + 1] - a
        dev = np.abs(seg[:, 0] * d[1] - seg[:, 1] * d[0]) / L
        k = int(np.argmax(dev))
        if dev[k] > max(tol, rel * L):
            stack.append((i + k, j))
            stack.append((i, i + k))
        else:
            out.append((i, j))
    out.sort()
    return [np.array([path[i][0], path[i][1], path[j][0], path[j][1]]) for i, j in out]


def vectorize_skeleton(skel, fit_tol_px, min_seed_px):
    segs = []
    for p in trace_skeleton(skel, min_len=max(3, int(min_seed_px // 2))):
        for s in split_straight(p, fit_tol_px):
            if np.hypot(s[2] - s[0], s[3] - s[1]) >= min_seed_px:
                segs.append(s)
    return np.array(segs, dtype=np.float64).reshape(-1, 4)


def run_evidence_model(name, S, T, valid, res, P, log=print) -> ModelResult:
    m2p = lambda m: float(m) / res
    border = P.border_margin_m or 1.5 * max(P.scales_m)
    L = m2p(P.line_length_m)
    lengths = [L, L * P.long_factor] if P.long_factor and P.long_factor > 1 else L
    skel, Rn, Theta = detect_line_pixels(
        S, T, valid, lengths, int(P.n_orient), P.angle_tol_deg,
        P.sensitivity, m2p(border), m2p(P.min_seed_m))
    segs = vectorize_skeleton(skel, max(m2p(P.fit_tol_m), 1.0), m2p(P.min_seed_m))
    n_raw = len(segs)
    segs = link_segments(segs, m2p(P.link_gap_m), P.link_angle_deg, max(m2p(P.link_lateral_m), 1.5))
    low, _ = sensitivity_thresholds(P.sensitivity)
    Sb = gblur(S, 1.0)
    segs, sup = trim_by_support(segs, Sb, 0.5 * low, m2p(P.min_length_m), P.min_support)
    # segunda pasada de enlace tras el recorte (une tramos ya depurados)
    segs = link_segments(segs, m2p(P.link_gap_m), P.link_angle_deg, max(m2p(P.link_lateral_m), 1.5),
                         gap_rel=P.link_gap_rel, lateral_rel=P.link_lateral_rel)
    segs, sup = trim_by_support(segs, Sb, 0.5 * low, m2p(P.min_length_m), P.min_support)
    log(f"    {name}: {n_raw} trazos -> {len(segs)} lineamientos")
    return ModelResult(name, segs, support=sup, rasters={"respuesta": Rn, "theta": Theta})


def run_pci_model(img8, valid, P, name="pci_line", log=print) -> ModelResult:
    polys, strength = pci_line(img8, valid, int(P.RADI), float(P.GTHR), int(P.LTHR),
                               float(P.FTHR), float(P.ATHR), float(P.DTHR))
    log(f"    {name}: {len(polys)} polilíneas")
    return ModelResult(name, polylines_to_segments(polys), polylines=polys,
                       rasters={"bordes": strength})


def run_pci_multi(z, valid, res, P, log=print) -> ModelResult:
    """PCI LINE sobre varios azimuts; la unión se depura fusionando duplicados
    paralelos (un mismo rasgo aparece con cada iluminación)."""
    allsegs = []
    for az in P.pci_multi_azimuths:
        hs = hillshade(z, res, az, P.pci_altitude, P.z_factor)
        r = run_pci_model(to_uint8(hs, valid), valid, P, name=f"pci az {az:g}", log=log)
        allsegs.append(r.segs)
    segs = np.vstack(allsegs) if allsegs else np.zeros((0, 4))
    segs = link_segments(segs, gap_px=2.0, angle_deg=15.0, lateral_px=3.0)
    log(f"    pci_multi: unión depurada -> {len(segs)} segmentos")
    return ModelResult("pci_multi", segs)
