"""
Feature stack para extracción de lineamientos.

Calcula hillshade multidireccional, slope y curvatura, y los combina
con PCA para obtener una imagen realzada donde la estructura lineal
queda mejor expuesta que en un hillshade único.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import sobel, laplace
from sklearn.decomposition import PCA

from .io_dem import DEMData


def _gradients(elev: np.ndarray, res_x: float, res_y: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradientes dz/dx, dz/dy usando Sobel escalado a unidades físicas."""
    dzdx = sobel(elev, axis=1, mode="reflect") / (8.0 * res_x)
    dzdy = sobel(elev, axis=0, mode="reflect") / (8.0 * res_y)
    return dzdx, dzdy


def hillshade(
    elev: np.ndarray,
    res_x: float,
    res_y: float,
    azimuth_deg: float,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Hillshade estándar (Horn 1981). Salida en [0, 1]."""
    dzdx, dzdy = _gradients(elev * z_factor, res_x, res_y)
    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(-dzdy, dzdx)

    az_rad = np.deg2rad(360.0 - azimuth_deg + 90.0)
    alt_rad = np.deg2rad(altitude_deg)

    hs = (
        np.sin(alt_rad) * np.cos(slope)
        + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )
    return np.clip(hs, 0.0, 1.0)


def hillshade_multidir_max(
    dem: DEMData,
    azimuths: list[float],
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """Máximo por píxel de hillshades en múltiples azimuts."""
    stack = np.stack(
        [hillshade(dem.elevation, dem.res_x, dem.res_y, az, altitude_deg, z_factor)
         for az in azimuths],
        axis=0,
    )
    return np.nanmax(stack, axis=0)


def slope_map(dem: DEMData) -> np.ndarray:
    """Pendiente en grados."""
    dzdx, dzdy = _gradients(dem.elevation, dem.res_x, dem.res_y)
    return np.rad2deg(np.arctan(np.hypot(dzdx, dzdy)))


def curvature_map(dem: DEMData) -> np.ndarray:
    """Laplaciano del DEM como proxy de curvatura total.
    Positivo en crestas/divisorias, negativo en valles."""
    return laplace(dem.elevation) / (dem.pixel_size_m ** 2)


def _normalize01(arr: np.ndarray) -> np.ndarray:
    """Normalización robusta por percentiles 2-98 a [0,1]."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    p2, p98 = np.percentile(valid, [2, 98])
    if p98 - p2 < 1e-10:
        return np.zeros_like(arr)
    out = (arr - p2) / (p98 - p2)
    return np.clip(out, 0.0, 1.0)


def build_feature_stack(
    dem: DEMData,
    azimuths: list[float],
    altitude_deg: float,
    z_factor: float,
    include_slope: bool,
    include_curvature: bool,
) -> tuple[np.ndarray, list[str]]:
    """Construye el stack (N, H, W) y devuelve nombres de capas."""
    layers, names = [], []

    hs_max = hillshade_multidir_max(dem, azimuths, altitude_deg, z_factor)
    layers.append(_normalize01(hs_max))
    names.append("hillshade_max")

    if include_slope:
        layers.append(_normalize01(slope_map(dem)))
        names.append("slope")

    if include_curvature:
        layers.append(_normalize01(curvature_map(dem)))
        names.append("curvature")

    return np.stack(layers, axis=0), names


def reduce_with_pca(stack: np.ndarray, component: int = 2) -> np.ndarray:
    """Aplica PCA sobre el stack y devuelve la componente solicitada (1-indexed).
    Si solo hay una capa, la devuelve tal cual."""
    n_layers, h, w = stack.shape
    if n_layers == 1:
        return stack[0]

    flat = stack.reshape(n_layers, -1).T            # (pixels, features)
    mask = np.isfinite(flat).all(axis=1)
    pca = PCA(n_components=min(n_layers, component))
    pca.fit(flat[mask])
    transformed = pca.transform(np.nan_to_num(flat))
    pc = transformed[:, component - 1].reshape(h, w)

    # Signo: asegurar correlación positiva con hillshade original
    # (PCA puede invertir componentes; lo alineamos visualmente)
    if np.corrcoef(pc.ravel(), stack[0].ravel())[0, 1] < 0:
        pc = -pc

    return _normalize01(pc)


def make_detector_input(
    dem: DEMData,
    use_pca: bool,
    pca_component: int,
    azimuths: list[float],
    altitude_deg: float,
    z_factor: float,
    include_slope: bool,
    include_curvature: bool,
) -> tuple[np.ndarray, dict]:
    """Orquesta el armado del input al detector. Devuelve imagen y metadata."""
    stack, names = build_feature_stack(
        dem, azimuths, altitude_deg, z_factor, include_slope, include_curvature
    )
    meta = {"layers": names, "pca_applied": False, "pca_component": None}

    if use_pca and stack.shape[0] > 1:
        img = reduce_with_pca(stack, pca_component)
        meta["pca_applied"] = True
        meta["pca_component"] = pca_component
    else:
        img = stack[0]

    return img.astype(np.float32), meta


def make_detector_input_from_config(dem: DEMData, cfg) -> tuple[np.ndarray, dict]:
    """Despacha la construcción del input según cfg.MODE."""
    mode = getattr(cfg, "MODE", "enhanced")

    if mode == "external_hillshade":
        # Carga directa del hillshade externo — mismo archivo que entra a Catalyst
        import rasterio
        hs_path = cfg.HILLSHADE_PATH
        with rasterio.open(hs_path) as src:
            img = src.read(1).astype(np.float32)
        img = _normalize01(img)
        meta = {
            "mode": "external_hillshade",
            "layers": [str(hs_path.name)],
            "pca_applied": False,
            "pca_component": None,
        }
        return img, meta

    if mode == "catalyst_replica":
        # Hillshade único al azimut de la corrida Catalyst
        hs = hillshade(
            dem.elevation,
            dem.res_x, dem.res_y,
            azimuth_deg=cfg.CATALYST_AZIMUTH,
            altitude_deg=cfg.CATALYST_ALTITUDE,
            z_factor=cfg.CATALYST_Z_FACTOR,
        )
        img = _normalize01(hs).astype(np.float32)
        meta = {
            "mode": "catalyst_replica",
            "layers": [f"hillshade_{cfg.CATALYST_AZIMUTH}deg"],
            "pca_applied": False,
            "pca_component": None,
        }
        return img, meta

    # mode == "enhanced"
    img, meta = make_detector_input(
        dem,
        use_pca=cfg.USE_MULTI_FEATURE_PCA,
        pca_component=cfg.PCA_COMPONENT,
        azimuths=cfg.AZIMUTHS,
        altitude_deg=cfg.ALTITUDE,
        z_factor=cfg.Z_FACTOR,
        include_slope=cfg.INCLUDE_SLOPE,
        include_curvature=cfg.INCLUDE_CURVATURE,
    )
    meta["mode"] = "enhanced"
    return img, meta
