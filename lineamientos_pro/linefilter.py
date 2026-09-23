"""
Filtro orientado de rectitud ("stick filter").

Idea: un lineamiento es una alineación RECTA y CONTINUA de evidencia con
orientación coherente. Para cada orientación theta_k se integra la evidencia
cuyo ángulo local concuerda con theta_k a lo largo de un núcleo lineal de
longitud L. Resultado:
  * premia la rectitud (rasgos sinuosos -drenaje meándrico, bordes de
    textura- no acumulan respuesta en ninguna orientación),
  * puentea huecos cortos (erosión, vegetación, sombras),
  * no depende del azimut de iluminación.

Después se aplica supresión de no-máximos perpendicular a la línea y
umbral por histéresis, igual que Canny pero sobre la respuesta de línea.
"""
from __future__ import annotations

import cv2
import numpy as np
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, skeletonize


def line_kernel(length_px: float, theta: float, sigma_across: float = 1.0) -> np.ndarray:
    half = int(np.ceil(length_px / 2.0 + 2 * sigma_across))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    c, s = np.cos(theta), np.sin(theta)
    u = xx * c + yy * s                   # a lo largo
    v = -xx * s + yy * c                  # a través
    sa = max(length_px / 4.0, 1.0)
    k = np.exp(-v ** 2 / (2 * sigma_across ** 2)) * np.exp(-u ** 2 / (2 * sa ** 2))
    k[np.abs(u) > length_px / 2.0] = 0
    return (k / k.sum()).astype(np.float32)


def _power_for_tolerance(tol_deg: float) -> float:
    """Exponente n de cos^(2n) con media altura en +/- tol."""
    c = np.cos(np.deg2rad(max(tol_deg, 1.0)))
    return float(np.log(0.5) / (2.0 * np.log(c)))


def _pct(R, valid, q=99.5):
    vals = R[valid & (R > 0)]
    if vals.size == 0:
        return 1.0
    if vals.size > 2_000_000:
        vals = np.random.default_rng(0).choice(vals, 2_000_000, replace=False)
    return float(np.percentile(vals, q)) or 1.0


def oriented_line_response(S: np.ndarray, T: np.ndarray, length_px, n_orient: int,
                           angle_tol_deg: float, valid: np.ndarray | None = None):
    """Devuelve (R, Theta): respuesta máxima entre orientaciones y su ángulo.

    length_px puede ser una lista (multi-longitud): cada longitud se normaliza
    por su percentil 99.5 y se toma el máximo. Los núcleos largos rescatan
    fallas largas y tenues (la relación señal/ruido crece con la raíz de L);
    los cortos conservan los rasgos cortos y nítidos."""
    lengths = list(length_px) if np.ndim(length_px) else [length_px]
    valid = np.ones(S.shape, bool) if valid is None else valid
    n = _power_for_tolerance(angle_tol_deg)
    thetas = np.arange(n_orient) * np.pi / n_orient
    Rbest = np.zeros(S.shape, np.float32)
    idx = np.zeros(S.shape, np.int16)
    for L in lengths:
        R = np.zeros(S.shape, np.float32)
        ik = np.zeros(S.shape, np.int16)
        for k, th in enumerate(thetas):
            w = np.cos(T - th) ** 2                   # concordancia angular (periodo pi)
            A = S * np.power(w, n, dtype=np.float32)
            O = cv2.filter2D(A, cv2.CV_32F, line_kernel(L, th), borderType=cv2.BORDER_REFLECT)
            m = O > R
            R[m] = O[m]
            ik[m] = k
        if len(lengths) > 1:
            R /= _pct(R, valid)
        m = R > Rbest
        Rbest[m] = R[m]
        idx[m] = ik[m]
    return Rbest, thetas[idx].astype(np.float32)


def nonmax_suppression(R: np.ndarray, Theta: np.ndarray) -> np.ndarray:
    """Conserva máximos locales en la dirección normal a la línea."""
    normal = np.mod(Theta + np.pi / 2, np.pi)
    q = (np.round(normal / (np.pi / 4)).astype(np.int8)) % 4   # 0,45,90,135°
    P = np.pad(R, 1, mode="edge")
    H, W = R.shape
    offs = {0: (0, 1), 1: (1, 1), 2: (1, 0), 3: (1, -1)}       # (dy, dx) en imagen
    keep = np.zeros(R.shape, bool)
    for k, (dy, dx) in offs.items():
        a = P[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
        b = P[1 - dy:1 - dy + H, 1 - dx:1 - dx + W]
        m = q == k
        keep |= m & (R >= a) & (R >= b)
    return keep & (R > 0)


def sensitivity_thresholds(sensitivity: float) -> tuple[float, float]:
    s = float(np.clip(sensitivity, 0.0, 1.0))
    high = 0.75 - 0.5 * s          # s=0.5 -> 0.50 ; s=1 -> 0.25
    return 0.5 * high, high


def detect_line_pixels(S, T, valid, length_px, n_orient, angle_tol_deg, sensitivity,
                       border_px, min_seed_px):
    """Evidencia -> esqueleto binario de lineamientos + mapas R/Theta."""
    R, Theta = oriented_line_response(S, T, length_px, n_orient, angle_tol_deg, valid)
    # normaliza R contra la respuesta de un lineamiento "típico fuerte"
    Rn = (R / _pct(R, valid)).astype(np.float32)
    Rs = cv2.GaussianBlur(Rn, (0, 0), 1.0)          # rompe empates en mesetas

    inner = valid.copy()
    b = int(np.ceil(border_px))
    if b > 0:
        inner[:b, :] = inner[-b:, :] = False
        inner[:, :b] = inner[:, -b:] = False
        inner = cv2.erode(inner.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=b).astype(bool)
    nms = nonmax_suppression(Rs, Theta) & inner
    low, high = sensitivity_thresholds(sensitivity)
    mask = apply_hysteresis_threshold(np.where(nms, Rs, 0), low, high)
    # cierra cortes de 1 px que deja la NMS en diagonales y adelgaza
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((3, 3), np.uint8)).astype(bool)
    skel = skeletonize(mask)
    skel = remove_small_objects(skel, max(3, int(min_seed_px)), connectivity=2)
    return skel, Rn, Theta
