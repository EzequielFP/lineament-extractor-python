"""Lectura/escritura de rásteres y geometría de la grilla de trabajo."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from scipy import ndimage as ndi


@dataclass
class Grid:
    """DEM en la grilla de trabajo."""
    z: np.ndarray            # elevación float32, huecos rellenados
    valid: np.ndarray        # máscara bool de datos originales válidos
    transform: Affine
    crs: object
    res: float               # tamaño de píxel (m)
    path: Path

    @property
    def shape(self):
        return self.z.shape

    def m2px(self, meters: float) -> float:
        return float(meters) / self.res

    def px2geo(self, cols, rows, center: bool = True):
        """Coordenadas de píxel (col, fila) -> CRS. center=True ubica el vértice en
        el centro del píxel; False en la esquina (convención de PCI Geomatica)."""
        T = self.transform
        off = 0.5 if center else 0.0
        cols = np.asarray(cols, dtype=np.float64) + off
        rows = np.asarray(rows, dtype=np.float64) + off
        return T.a * cols + T.b * rows + T.c, T.d * cols + T.e * rows + T.f

    def geo2px(self, xs, ys):
        inv = ~self.transform
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        c = inv.a * xs + inv.b * ys + inv.c - 0.5
        r = inv.d * xs + inv.e * ys + inv.f - 0.5
        return c, r


def _nodata_mask(arr: np.ndarray, nodata) -> np.ndarray:
    bad = ~np.isfinite(arr) | (np.abs(arr) > 1e20)
    if nodata is not None and np.isfinite(nodata):
        bad |= np.isclose(arr, nodata, rtol=0, atol=abs(nodata) * 1e-6 + 1e-6)
    return ~bad


def fill_nodata(z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Rellena huecos con el valor válido más cercano (sin crear bordes falsos)."""
    if valid.all():
        return z
    if not valid.any():
        raise ValueError("El ráster no contiene datos válidos.")
    idx = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def load_dem(path: str | Path, work_res_m: float = 0.0) -> Grid:
    path = Path(path)
    with rasterio.open(path) as src:
        if src.crs is not None and src.crs.is_geographic:
            raise ValueError(
                "El DEM está en coordenadas geográficas (grados). Reproyecte a un "
                "sistema proyectado en metros (p. ej. MAGNA-SIRGAS Origen Nacional).")
        native = (abs(src.transform.a) + abs(src.transform.e)) / 2.0
        factor = work_res_m / native if work_res_m and work_res_m > native * 1.01 else 1.0
        if factor > 1.0:
            h, w = max(1, int(round(src.height / factor))), max(1, int(round(src.width / factor)))
            raw = src.read(1, out_shape=(h, w), resampling=Resampling.average,
                           masked=False).astype(np.float32)
            transform = src.transform * Affine.scale(src.width / w, src.height / h)
        else:
            raw = src.read(1).astype(np.float32)
            transform = src.transform
        valid = _nodata_mask(raw, src.nodata)
        crs = src.crs
    res = (abs(transform.a) + abs(transform.e)) / 2.0
    z = fill_nodata(np.where(valid, raw, 0).astype(np.float32), valid)
    return Grid(z=z, valid=valid, transform=transform, crs=crs, res=res, path=path)


def save_raster(path: str | Path, arr: np.ndarray, grid: Grid, nodata=None,
                transform: Affine | None = None) -> Path:
    path = Path(path)
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    profile = dict(
        driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=arr.dtype, crs=grid.crs, transform=transform or grid.transform,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    if arr.shape[0] < 256 or arr.shape[1] < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize"), profile.pop("blockysize")
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    return path


def load_image_native(path: str | Path, fill: bool = True) -> Grid:
    """Imagen de una banda en su grilla nativa (p. ej. sombreado para PCI).
    fill=False conserva los valores originales (la máscara marca el NoData)."""
    path = Path(path)
    with rasterio.open(path) as src:
        raw = src.read(1).astype(np.float32)
        valid = _nodata_mask(raw, src.nodata)
        transform, crs = src.transform, src.crs
    res = (abs(transform.a) + abs(transform.e)) / 2.0
    z = fill_nodata(np.where(valid, raw, 0).astype(np.float32), valid) if fill else raw
    return Grid(z=z, valid=valid, transform=transform, crs=crs, res=res, path=path)
