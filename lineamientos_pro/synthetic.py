"""
Banco de pruebas sintético con verdad conocida.

Genera un DEM montañoso (ruido fractal) con:
  * FALLAS rectas de rumbo aleatorio (verdad de terreno): valles lineales,
    crestas y escarpes sutiles (2-8 m), algunas con tramos erosionados.
  * DISTRACTORES que no son lineamientos: cauces sinuosos más profundos que
    las fallas, rasgos rectos muy cortos (< 40 m) y ruido.
Permite medir precisión/recall/F1 de cada modelo de forma objetiva.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from rasterio.transform import from_origin


@dataclass
class Scene:
    z: np.ndarray
    truth: np.ndarray          # (N,4) segmentos en píxeles
    truth_type: list
    res: float
    transform: object


def _fbm(n, beta, rng):
    kx = np.fft.fftfreq(n)[None, :]
    ky = np.fft.fftfreq(n)[:, None]
    k = np.hypot(kx, ky)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0
    ph = np.exp(2j * np.pi * rng.random((n, n)))
    f = np.real(np.fft.ifft2(amp * ph))
    return (f - f.mean()) / f.std()


def _dist_to_path(shape, pts):
    m = np.full(shape, 255, np.uint8)
    cv2.polylines(m, [np.round(pts).astype(np.int32)], False, 0, 1)
    return cv2.distanceTransform(m, cv2.DIST_L2, 5)


def make_scene(n=1024, res=2.0, seed=0, n_faults=14, n_channels=10, n_short=40,
               relief=80.0) -> Scene:
    rng = np.random.default_rng(seed)
    # relieve fractal suavizado (pendiente media ~18°, como montaña andina)
    z = cv2.GaussianBlur((relief * _fbm(n, 4.2, rng)).astype(np.float32), (0, 0), 3.0)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)

    # cauces sinuosos (distractores fuertes)
    for _ in range(n_channels):
        p = rng.uniform(0.1 * n, 0.9 * n, 2)
        th = rng.uniform(0, 2 * np.pi)
        pts = [p.copy()]
        for _ in range(int(rng.uniform(300, 700))):
            th += rng.normal(0, 0.12)
            p = p + 1.5 * np.array([np.cos(th), np.sin(th)])
            pts.append(p.copy())
        d = _dist_to_path((n, n), np.array(pts))
        depth, w = rng.uniform(4, 10), rng.uniform(3, 6)
        z -= depth * np.exp(-(d ** 2) / (2 * w ** 2))

    # rasgos rectos cortos (no son lineamientos: < 40 m)
    for _ in range(n_short):
        c = rng.uniform(0, n, 2)
        th = rng.uniform(0, np.pi)
        L = rng.uniform(8, 18)   # px -> 16-36 m
        a = c - L / 2 * np.array([np.cos(th), np.sin(th)])
        b = c + L / 2 * np.array([np.cos(th), np.sin(th)])
        d = _dist_to_path((n, n), np.array([a, b]))
        z -= rng.uniform(1.5, 3) * np.exp(-(d ** 2) / (2 * 1.5 ** 2))

    # fallas (verdad)
    truth, types = [], []
    for _ in range(n_faults):
        L = rng.uniform(100, 700)           # px -> 200-1400 m
        th = rng.uniform(0, np.pi)
        u = np.array([np.cos(th), np.sin(th)])
        c = rng.uniform(0.15 * n, 0.85 * n, 2)
        a = c - L / 2 * u
        b = c + L / 2 * u
        a, b = np.clip(a, 5, n - 6), np.clip(b, 5, n - 6)
        L = np.hypot(*(b - a))
        if L < 60:
            continue
        u = (b - a) / L
        t = (xx - a[0]) * u[0] + (yy - a[1]) * u[1]
        s = -(xx - a[0]) * u[1] + (yy - a[1]) * u[0]
        taper = np.clip(np.minimum(t, L - t) / 15.0, 0, 1)
        taper = np.where((t < 0) | (t > L), 0, taper)
        # tramos erosionados (huecos) en ~40 % de las fallas
        if rng.random() < 0.4:
            g0 = rng.uniform(0.3, 0.6) * L
            gl = rng.uniform(0.08, 0.18) * L
            taper = taper * np.clip(np.abs(t - (g0 + gl / 2)) / (gl / 2), 0, 1) ** 2
        kind = rng.choice(["valle", "cresta", "escarpe"], p=[0.45, 0.2, 0.35])
        w = rng.uniform(2, 5)
        amp = rng.uniform(2, 8)
        if kind == "valle":
            z -= amp * taper * np.exp(-s ** 2 / (2 * w ** 2))
        elif kind == "cresta":
            z += amp * taper * np.exp(-s ** 2 / (2 * w ** 2))
        else:
            z += amp * taper * np.tanh(s / w) * np.exp(-s ** 2 / (2 * (12 * w) ** 2))
        truth.append((a[0], a[1], b[0], b[1]))
        types.append(kind)

    z += rng.normal(0, 0.03, z.shape)
    z = (z - z.min() + 1500).astype(np.float32)
    return Scene(z=z, truth=np.array(truth), truth_type=types, res=res,
                 transform=from_origin(1_000_000, 2_000_000 + n * res, res, res))
