"""
Detectores de estructuras lineales.

Tres implementaciones intercambiables:
  - canny            : bordes clásicos con histéresis
  - phase_congruency : detección por congruencia de fase (Kovesi)
  - frangi           : realce de crestas vía Hessiano multi-escala

Todas devuelven una imagen binaria (H, W) bool donde True = píxel de estructura.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, uniform_filter
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import thin

# Phase Congruency: intento phasepack, fallback a implementación manual
try:
    from phasepack import phasecong
    _HAS_PHASEPACK = True
except ImportError:
    _HAS_PHASEPACK = False


# -------------------------------------------------------------------
# Suavizado previo (RADI)
# -------------------------------------------------------------------
def apply_radi(img: np.ndarray, radi_pixels: int, mode: str = "gaussian") -> np.ndarray:
    """Suavizado previo al detector.

    mode:
      'gaussian' -> Gaussian con sigma = RADI/3  (suavizado fuerte)
      'box'      -> Filtro de media (uniform) con ventana 2*RADI+1
                    Esto es lo que hace Catalyst: suavizado suave que
                    preserva mejor los gradientes.
    """
    if radi_pixels <= 0:
        return img
    if mode == "box":
        size = 2 * radi_pixels + 1
        return uniform_filter(img, size=size)
    else:
        sigma = max(radi_pixels / 3.0, 0.5)
        return gaussian_filter(img, sigma=sigma)


# -------------------------------------------------------------------
# Canny
# -------------------------------------------------------------------
def detect_canny(
    img: np.ndarray,
    gthr_percentile: float = 70.0,
    gthr_ratio_high: float = 3.0,
    gthr_absolute: float | None = None,
    canny_sigma: float = 1.0,
) -> tuple[np.ndarray, dict]:
    """Canny con histeresis.

    Calcula gradientes internamente para determinar umbrales, luego
    delega a skimage.feature.canny que hace su propio Gaussian+Sobel
    con sigma=canny_sigma para la deteccion real de bordes.

    Si gthr_absolute no es None, se interpreta como umbral absoluto
    sobre el gradiente de la imagen SIN suavizar (como hace Catalyst).
    El umbral se mapea al percentil equivalente para pasarlo a skimage.
    """
    from skimage.feature import canny as sk_canny

    # Calcular gradiente para establecer umbrales
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    mag = np.hypot(gx, gy)
    valid = mag[np.isfinite(mag) & (mag > 0)]

    if valid.size == 0:
        return np.zeros_like(img, dtype=bool), {"method": "canny", "error": "no valid gradients"}

    if gthr_absolute is not None:
        # Mapear umbral absoluto al percentil equivalente del gradiente
        pct_equiv = 100.0 * np.searchsorted(np.sort(valid), gthr_absolute) / valid.size
        low = float(np.percentile(valid, min(pct_equiv, 99.0)))
    else:
        low = float(np.percentile(valid, gthr_percentile))

    high = low * gthr_ratio_high

    # Proteccion: asegurar que high > low
    grad_max = float(np.nanmax(mag))
    if high <= low:
        high = low * 1.5
    # No dejar que high supere el maximo real
    high = min(high, grad_max * 0.99)
    if high <= low:
        high = low + 0.01

    edges = sk_canny(img, sigma=canny_sigma,
                     low_threshold=low, high_threshold=high)

    meta = {
        "method": "canny",
        "canny_sigma": canny_sigma,
        "low_threshold": float(low),
        "high_threshold": float(high),
        "gthr_type": "absolute" if gthr_absolute is not None else "percentile",
        "gradient_max": grad_max,
        "gradient_p50": float(np.percentile(valid, 50)),
        "gradient_p90": float(np.percentile(valid, 90)),
    }
    return edges, meta


# ───────────────────────────────────────────────────────────────────
# Phase Congruency
# ───────────────────────────────────────────────────────────────────
def _phase_congruency_manual(img: np.ndarray, nscale: int, norient: int,
                              min_wavelength: int, mult: float,
                              sigma_onf: float, k: float) -> np.ndarray:
    """Implementación simplificada de PC vía banco de filtros log-Gabor.

    Si phasepack está instalado se usa esa implementación (más robusta y
    calibrada). Este fallback produce resultados razonables pero no
    idénticos al paper de Kovesi 2000.
    """
    rows, cols = img.shape
    img_fft = np.fft.fft2(img)

    # Grillas de frecuencia
    u = np.fft.fftfreq(cols)[None, :]
    v = np.fft.fftfreq(rows)[:, None]
    radius = np.sqrt(u**2 + v**2)
    radius[0, 0] = 1.0  # evitar log(0)
    theta = np.arctan2(v, u)

    # Filtro pasa-bajos (suprime DC)
    lp = 1.0 / (1.0 + (radius / 0.45) ** (2 * 15))

    energy_sum = np.zeros((rows, cols))
    amplitude_sum = np.zeros((rows, cols))

    for o in range(norient):
        angle = o * np.pi / norient
        # Diferencia angular (circular)
        d_theta = np.abs(np.angle(np.exp(1j * (theta - angle))))
        spread = np.exp(-(d_theta ** 2) / (2 * (np.pi / norient / 1.5) ** 2))

        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        sum_amp = np.zeros((rows, cols))

        for s in range(nscale):
            wavelength = min_wavelength * (mult ** s)
            fo = 1.0 / wavelength
            log_gabor = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_onf) ** 2))
            log_gabor *= lp
            log_gabor[0, 0] = 0.0

            filt = log_gabor * spread
            response = np.fft.ifft2(img_fft * filt)
            sum_e += np.real(response)
            sum_o += np.imag(response)
            sum_amp += np.abs(response)

        energy = np.sqrt(sum_e ** 2 + sum_o ** 2)
        # Supresión de ruido (threshold k)
        tau = np.median(sum_amp) / np.sqrt(np.log(4))
        noise_thresh = tau * (1.0 + k * 0.5)
        energy = np.maximum(energy - noise_thresh, 0.0)

        energy_sum += energy
        amplitude_sum += sum_amp

    pc = energy_sum / (amplitude_sum + 1e-6)
    return pc


def detect_phase_congruency(
    img: np.ndarray,
    nscale: int = 5,
    norient: int = 6,
    min_wavelength: int = 3,
    mult: float = 2.1,
    sigma_onf: float = 0.55,
    k: float = 2.0,
    threshold: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """Binariza el mapa de phase congruency por umbral fijo."""
    if _HAS_PHASEPACK:
        pc_result = phasecong(
            img, nscale=nscale, norient=norient,
            minWaveLength=min_wavelength, mult=mult,
            sigmaOnf=sigma_onf, k=k,
        )
        # phasepack devuelve tupla; el primer elemento es el PC máximo
        pc_map = pc_result[0] if isinstance(pc_result, tuple) else pc_result
        backend = "phasepack"
    else:
        pc_map = _phase_congruency_manual(
            img, nscale, norient, min_wavelength, mult, sigma_onf, k
        )
        backend = "manual"

    edges = pc_map > threshold
    edges = thin(edges)  # adelgazamiento morfológico

    meta = {
        "method": "phase_congruency",
        "backend": backend,
        "threshold": threshold,
        "pc_max": float(np.nanmax(pc_map)),
        "pc_mean": float(np.nanmean(pc_map)),
    }
    return edges, meta


# ───────────────────────────────────────────────────────────────────
# Frangi (ridge detection)
# ───────────────────────────────────────────────────────────────────
def detect_frangi(
    img: np.ndarray,
    scales: list[int] = (2, 4, 6, 8, 12),
    beta: float = 0.5,
    black_ridges: bool = False,
) -> tuple[np.ndarray, dict]:
    """Realce de crestas con Frangi. Binariza con Otsu sobre la respuesta."""
    # skimage.frangi espera sigmas; usamos las escalas directamente como sigmas
    response = frangi(
        img,
        sigmas=list(scales),
        beta=beta,
        black_ridges=black_ridges,
    )

    # Binarización: Otsu sobre valores > 0
    nonzero = response[response > 0]
    if nonzero.size > 100:
        thr = threshold_otsu(nonzero)
    else:
        thr = np.percentile(response, 95)

    edges = response > thr
    edges = thin(edges)

    meta = {
        "method": "frangi",
        "scales": list(scales),
        "threshold_otsu": float(thr),
        "response_max": float(response.max()),
    }
    return edges, meta


# ───────────────────────────────────────────────────────────────────
# Dispatcher
# ───────────────────────────────────────────────────────────────────
def run_detector(
    img: np.ndarray,
    detector: str,
    radi_by_detector: dict,
    cfg,
) -> tuple[np.ndarray, dict]:
    """Aplica suavizado RADI + detector seleccionado."""
    radi = radi_by_detector.get(detector, 0)
    # Para Canny usa box filter (como Catalyst), para otros usa gaussian
    smooth_mode = getattr(cfg, "RADI_FILTER_MODE", "box" if detector == "canny" else "gaussian")
    img_smooth = apply_radi(img, radi, mode=smooth_mode)

    if detector == "canny":
        gthr_abs = getattr(cfg, "GTHR_ABSOLUTE", None)
        canny_sigma = getattr(cfg, "CANNY_SIGMA", 1.0)
        edges, meta = detect_canny(
            img_smooth,
            gthr_percentile=cfg.GTHR_PERCENTILE,
            gthr_ratio_high=cfg.GTHR_RATIO_HIGH,
            gthr_absolute=gthr_abs,
            canny_sigma=canny_sigma,
        )
    elif detector == "phase_congruency":
        edges, meta = detect_phase_congruency(
            img_smooth,
            nscale=cfg.PC_NSCALE, norient=cfg.PC_NORIENT,
            min_wavelength=cfg.PC_MIN_WAVELENGTH, mult=cfg.PC_MULT,
            sigma_onf=cfg.PC_SIGMA_ONF, k=cfg.PC_K,
            threshold=cfg.PC_THRESHOLD,
        )
    elif detector == "frangi":
        edges, meta = detect_frangi(
            img_smooth,
            scales=cfg.FRANGI_SCALES,
            beta=cfg.FRANGI_BETA,
            black_ridges=cfg.FRANGI_BLACK_RIDGES,
        )
    else:
        raise ValueError(f"Detector desconocido: {detector}")

    meta["radi_applied"] = radi
    return edges, meta
