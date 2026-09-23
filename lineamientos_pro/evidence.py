"""
Mapas de evidencia lineal calculados directamente sobre el DEM.

Cada función devuelve pares (S, theta):
  S     : intensidad de evidencia, normalizada (~1 = percentil 99)
  theta : dirección de la LÍNEA en el marco de la imagen, radianes [0, pi),
          medida desde +x (columnas, Este) hacia +y (filas, Sur).

Evidencias:
  valles   : concavidad transversal (autovalor mayor del Hessiano > 0)
  crestas  : convexidad transversal (autovalor menor del Hessiano < 0)
  escarpes : crestas del mapa de pendiente (quiebres/escalones)
  bordes   : gradiente de sombreados en varios azimuts (lo que "ve" un
             intérprete o PCI LINE, pero sin sesgo de iluminación)
"""
from __future__ import annotations

import cv2
import numpy as np

cv2.setNumThreads(0)  # usa todos los núcleos

_K_XX = np.array([[1.0, -2.0, 1.0]], dtype=np.float32)
_K_YY = _K_XX.T.copy()
_K_XY = np.array([[0.25, 0.0, -0.25], [0.0, 0.0, 0.0], [-0.25, 0.0, 0.25]], dtype=np.float32)
_K_X = np.array([[-0.5, 0.0, 0.5]], dtype=np.float32)
_K_Y = _K_X.T.copy()


def _f(img, k):
    return cv2.filter2D(img, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT)


def gblur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3:
        return img.astype(np.float32, copy=False)
    return cv2.GaussianBlur(img.astype(np.float32, copy=False), (0, 0), sigmaX=sigma,
                            sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def robust_scale(S: np.ndarray, valid: np.ndarray, q: float = 99.0, cap: float = 5.0,
                 rng=np.random.default_rng(0)) -> np.ndarray:
    """Divide por el percentil q de los valores positivos válidos."""
    vals = S[valid]
    vals = vals[vals > 0]
    if vals.size == 0:
        return np.zeros_like(S, dtype=np.float32)
    if vals.size > 2_000_000:
        vals = rng.choice(vals, 2_000_000, replace=False)
    p = float(np.percentile(vals, q))
    if p <= 0:
        return np.zeros_like(S, dtype=np.float32)
    out = S / p
    np.clip(out, 0, cap, out=out)
    out[~valid] = 0
    return out.astype(np.float32, copy=False)


def _hessian(zs: np.ndarray):
    hxx, hyy, hxy = _f(zs, _K_XX), _f(zs, _K_YY), _f(zs, _K_XY)
    tr = hxx + hyy
    tmp = np.sqrt((hxx - hyy) ** 2 + 4.0 * hxy ** 2)
    l1 = 0.5 * (tr + tmp)          # autovalor mayor
    l2 = 0.5 * (tr - tmp)          # autovalor menor
    psi = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)  # dirección del autovector de l1
    return l1, l2, psi


def _wrap(theta):
    return np.mod(theta, np.pi).astype(np.float32)


def hessian_evidence(z: np.ndarray, valid: np.ndarray, scales_px: list[float]):
    """Valles y crestas multiescala. Normaliza cada escala por separado para
    que rasgos pequeños (trazas de falla) no queden opacados por los grandes
    valles del relieve."""
    out = {}
    best = {k: (np.zeros(z.shape, np.float32), np.zeros(z.shape, np.float32))
            for k in ("valles", "crestas")}
    for s in scales_px:
        zs = gblur(z, s)
        l1, l2, psi = _hessian(zs)
        # valle: l1 >> 0 y |l2| pequeño ; línea perpendicular al autovector de l1
        rv = np.maximum(l1 - np.abs(l2), 0)
        # cresta: l2 << 0 y |l1| pequeño ; línea paralela al autovector de l1
        rr = np.maximum(-l2 - np.abs(l1), 0)
        for key, r, th in (("valles", rv, psi + np.pi / 2), ("crestas", rr, psi)):
            r = robust_scale(r, valid)
            S, T = best[key]
            m = r > S
            S[m] = r[m]
            T[m] = _wrap(th)[m]
    for k, (S, T) in best.items():
        out[k] = (S, T)
    return out


def slope_evidence(z: np.ndarray, valid: np.ndarray, scales_px: list[float]):
    """Escarpes: crestas del mapa de pendiente (zonas lineales más empinadas
    que su entorno: escarpes de falla, facetas, rupturas de pendiente)."""
    S = np.zeros(z.shape, np.float32)
    T = np.zeros(z.shape, np.float32)
    for s in scales_px:
        zs = gblur(z, s)
        g = np.hypot(_f(zs, _K_X), _f(zs, _K_Y))
        gs = gblur(g, max(1.0, s / 2.0))
        l1, l2, psi = _hessian(gs)
        r = robust_scale(np.maximum(-l2 - np.abs(l1), 0), valid)
        m = r > S
        S[m] = r[m]
        T[m] = _wrap(psi)[m]
    return S, T


def hillshade(z: np.ndarray, res: float, azimuth: float, altitude: float = 45.0,
              z_factor: float = 1.0) -> np.ndarray:
    """Sombreado analítico [0,1] por producto normal·sol (azimut horario desde N)."""
    p = _f(z, _K_X) * (z_factor / res)          # dz/dEste
    q = -_f(z, _K_Y) * (z_factor / res)         # dz/dNorte (filas crecen al Sur)
    az, alt = np.deg2rad(azimuth), np.deg2rad(altitude)
    sx, sy, sz = np.sin(az) * np.cos(alt), np.cos(az) * np.cos(alt), np.sin(alt)
    hs = (-p * sx - q * sy + sz) / np.sqrt(1.0 + p * p + q * q)
    return np.clip(hs, 0, 1).astype(np.float32)


def multidirectional_hillshade(z, res, azimuths, altitude=45.0, z_factor=1.0):
    acc = np.zeros(z.shape, np.float32)
    for a in azimuths:
        acc += hillshade(z, res, a, altitude, z_factor)
    return acc / max(len(azimuths), 1)


def edge_evidence(z: np.ndarray, valid: np.ndarray, res: float, sigma_px: float,
                  azimuths, altitude: float, z_factor: float = 1.0):
    """Bordes de sombreado: máximo del gradiente entre azimuts."""
    zs = gblur(z, sigma_px)
    S = np.zeros(z.shape, np.float32)
    T = np.zeros(z.shape, np.float32)
    for a in azimuths:
        hs = gblur(hillshade(zs, res, a, altitude, z_factor), 1.0)
        gx, gy = _f(hs, _K_X), _f(hs, _K_Y)
        mag = np.hypot(gx, gy)
        m = mag > S
        S[m] = mag[m]
        T[m] = _wrap(np.arctan2(gy, gx) + np.pi / 2)[m]
    return robust_scale(S, valid), T


def compute_evidences(z, valid, res, params, wanted, log=print):
    """Calcula solo las evidencias necesarias para los modelos pedidos."""
    scales_px = [max(0.7, s / res) for s in params.scales_m]
    ev = {}
    if {"valles", "crestas"} & wanted:
        log(f"  · Hessiano multiescala (σ = {', '.join(f'{s:.1f}' for s in scales_px)} px)")
        ev.update(hessian_evidence(z, valid, scales_px))
    if "escarpes" in wanted:
        log("  · Escarpes / quiebres de pendiente")
        ev["escarpes"] = slope_evidence(z, valid, scales_px[:3] if len(scales_px) > 3 else scales_px)
    if "bordes" in wanted:
        log(f"  · Bordes de sombreado en {len(params.hs_azimuths)} azimuts")
        ev["bordes"] = edge_evidence(z, valid, res, scales_px[0], params.hs_azimuths,
                                     params.hs_altitude, params.z_factor)
    return {k: v for k, v in ev.items() if k in wanted}
