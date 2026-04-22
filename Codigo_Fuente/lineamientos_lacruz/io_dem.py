"""Carga de DEM y metadatos geoespaciales."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class DEMData:
    elevation: np.ndarray          # (H, W) float32, nodata → np.nan
    transform: Affine              # affine rasterio
    crs: object                    # rasterio CRS
    res_x: float                   # resolución en X (unidades del CRS)
    res_y: float                   # resolución en Y (positiva)
    nodata: float | None
    path: Path

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def pixel_size_m(self) -> float:
        """Tamaño medio de píxel en metros. Asume CRS proyectado en metros."""
        return (self.res_x + self.res_y) / 2.0


def load_dem(path: str | Path) -> DEMData:
    path = Path(path)
    with rasterio.open(path) as src:
        elev = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            elev[elev == nodata] = np.nan
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        return DEMData(
            elevation=elev,
            transform=src.transform,
            crs=src.crs,
            res_x=res_x,
            res_y=res_y,
            nodata=nodata,
            path=path,
        )


def pixels_to_meters(n_pixels: float, dem: DEMData) -> float:
    return n_pixels * dem.pixel_size_m


def meters_to_pixels(n_meters: float, dem: DEMData) -> float:
    return n_meters / dem.pixel_size_m
