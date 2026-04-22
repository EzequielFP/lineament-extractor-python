"""
Filtrado opcional de lineamientos que coinciden con la red de drenaje.

Usa pysheds para extraer la red, la vectoriza, genera buffer, y descarta
lineamientos cuya fracción dentro del buffer supera un umbral.
"""
from __future__ import annotations
from pathlib import Path
import tempfile
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString

from .io_dem import DEMData


def extract_drainage_buffer(
    dem: DEMData,
    accum_threshold: int,
    buffer_m: float,
) -> gpd.GeoSeries | None:
    """Extrae red de drenaje desde el DEM y genera un buffer.

    Devuelve una GeoSeries con la geometría de buffer unificada, o None si
    pysheds no está disponible o falla.
    """
    try:
        from pysheds.grid import Grid
    except ImportError:
        print("  [drainage] pysheds no disponible — se omite filtrado.")
        return None

    # pysheds necesita leer desde archivo; usamos el DEM original
    dem_path = str(dem.path)

    try:
        grid = Grid.from_raster(dem_path)
        elev = grid.read_raster(dem_path)

        pit_filled = grid.fill_pits(elev)
        flooded = grid.fill_depressions(pit_filled)
        inflated = grid.resolve_flats(flooded)

        fdir = grid.flowdir(inflated)
        acc = grid.accumulation(fdir)

        # Red como máscara booleana
        stream_mask = acc > accum_threshold

        # Vectorizar: cada píxel de stream → línea a su vecino aguas abajo.
        # Aquí tomamos el camino simple: convertir la máscara a polígonos y
        # luego a buffer, que es suficiente para el objetivo de filtrar.
        from rasterio.features import shapes
        from shapely.geometry import shape

        mask_u8 = stream_mask.astype(np.uint8)
        polys = [
            shape(geom) for geom, val in shapes(mask_u8, transform=dem.transform)
            if val == 1
        ]
        if not polys:
            return None

        gs = gpd.GeoSeries(polys, crs=dem.crs)
        buffered = gs.buffer(buffer_m).union_all()
        return gpd.GeoSeries([buffered], crs=dem.crs)

    except Exception as e:
        print(f"  [drainage] Error extrayendo red de drenaje: {e}")
        return None


def filter_lineaments_by_drainage(
    gdf: gpd.GeoDataFrame,
    drainage_buffer: gpd.GeoSeries,
    overlap_threshold: float,
) -> gpd.GeoDataFrame:
    """Descarta lineamientos cuya fracción dentro del buffer > umbral."""
    if drainage_buffer is None or len(drainage_buffer) == 0:
        return gdf

    buf_geom = drainage_buffer.iloc[0]

    def overlap_frac(line: LineString) -> float:
        if line.length == 0:
            return 0.0
        inter = line.intersection(buf_geom)
        return inter.length / line.length if not inter.is_empty else 0.0

    fracs = gdf.geometry.apply(overlap_frac)
    gdf = gdf.copy()
    gdf["drainage_overlap"] = fracs
    keep = fracs < overlap_threshold
    return gdf[keep].reset_index(drop=True)
