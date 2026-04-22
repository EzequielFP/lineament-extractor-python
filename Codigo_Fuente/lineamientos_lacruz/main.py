"""
Pipeline de extraccion automatica de lineamientos - La Cruz.
Corre: python -m lineamientos_lacruz.main

Modos (cfg.MODE):
  "enhanced"            -> stack PCA multidireccional + detector configurable
  "catalyst_replica"    -> hillshade generado internamente (azimut 0°) + Canny
  "external_hillshade"  -> hillshade externo .tif + Canny
  "batch"               -> corre todas las combinaciones definidas en BATCH_RUNS
                          de una sola vez. Genera un shapefile por combinacion.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import os

from . import config as cfg
from .io_dem import load_dem
from .features import make_detector_input_from_config
from .detectors import run_detector
from .extractors import run_extractor
from .postprocess import (
    merge_colinear, filter_by_length_m, pixels_to_geo,
    compute_attributes, classify_families,
)
from .validation import full_validation_report


# -------------------------------------------------------------------
# BATCH: combinaciones a probar en una sola corrida
# Cada entrada es un dict con overrides sobre config.py.
# "label" define el nombre del shapefile de salida: lin_{label}_{extractor}.shp
# -------------------------------------------------------------------
BATCH_RUNS = [
    # ---- REPLICA CATALYST: Canny sobre hillshade externo ----
    # Variamos percentil para encontrar el optimo vs los 3502 de Catalyst
    {
        "MODE": "external_hillshade",
        "DETECTOR": "canny",
        "EXTRACTORS": ["skeleton"],
        "GTHR_ABSOLUTE": None,
        "GTHR_PERCENTILE": 50,
        "CANNY_SIGMA": 1.0,
        "RADI_BY_DETECTOR": {"canny": 3},
        "LTHR": 15, "FTHR": 2, "ATHR": 30, "DTHR": 25,
        "DILATION_RADIUS": 1,
        "MIN_LENGTH_M": 20,
        "label": "HS_CANNY_P50",
    },
    {
        "MODE": "external_hillshade",
        "DETECTOR": "canny",
        "EXTRACTORS": ["skeleton"],
        "GTHR_ABSOLUTE": None,
        "GTHR_PERCENTILE": 60,
        "CANNY_SIGMA": 1.0,
        "RADI_BY_DETECTOR": {"canny": 3},
        "LTHR": 15, "FTHR": 2, "ATHR": 30, "DTHR": 25,
        "DILATION_RADIUS": 1,
        "MIN_LENGTH_M": 20,
        "label": "HS_CANNY_P60",
    },
    {
        "MODE": "external_hillshade",
        "DETECTOR": "canny",
        "EXTRACTORS": ["skeleton"],
        "GTHR_ABSOLUTE": None,
        "GTHR_PERCENTILE": 70,
        "CANNY_SIGMA": 1.0,
        "RADI_BY_DETECTOR": {"canny": 3},
        "LTHR": 15, "FTHR": 2, "ATHR": 30, "DTHR": 25,
        "DILATION_RADIUS": 1,
        "MIN_LENGTH_M": 20,
        "label": "HS_CANNY_P70",
    },

    # ---- FRANGI mejorado (multi-escala mas fina) ----
    {
        "MODE": "external_hillshade",
        "DETECTOR": "frangi",
        "EXTRACTORS": ["skeleton"],
        "FRANGI_SCALES": [1, 2, 3, 4, 6, 8],
        "RADI_BY_DETECTOR": {"frangi": 0},
        "LTHR": 15, "FTHR": 2, "ATHR": 30, "DTHR": 25,
        "DILATION_RADIUS": 1,
        "MIN_LENGTH_M": 20,
        "label": "HS_FRANGI_FINE",
    },

    # ---- PCA multidireccional + Canny ----
    {
        "MODE": "enhanced",
        "DETECTOR": "canny",
        "EXTRACTORS": ["skeleton"],
        "GTHR_ABSOLUTE": None,
        "GTHR_PERCENTILE": 60,
        "CANNY_SIGMA": 1.0,
        "RADI_BY_DETECTOR": {"canny": 3},
        "LTHR": 15, "FTHR": 2, "ATHR": 30, "DTHR": 25,
        "DILATION_RADIUS": 1,
        "MIN_LENGTH_M": 20,
        "label": "PCA_CANNY_P60",
    },
]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _setup_output(dem_path: Path, output_dir) -> Path:
    out = dem_path.parent / "Resultados" if output_dir is None else Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_raster(arr: np.ndarray, path: Path, dem) -> None:
    import rasterio
    profile = {
        "driver": "GTiff", "height": arr.shape[0], "width": arr.shape[1],
        "count": 1, "dtype": arr.dtype, "crs": dem.crs,
        "transform": dem.transform, "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)


def segments_to_gdf(segments, dem, attrs) -> gpd.GeoDataFrame:
    geoms = [LineString([(s[0], s[1]), (s[2], s[3])]) for s in segments]
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs=dem.crs)


class _CfgOverride:
    """Aplica overrides sobre cfg sin modificar el módulo original."""
    def __init__(self, base, overrides: dict):
        self._base = base
        self._ov = overrides

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._ov[name] if name in self._ov else getattr(self._base, name)


# -------------------------------------------------------------------
# Núcleo: procesar una combinación
# -------------------------------------------------------------------
def _run_single(dem, out_dir, run_cfg, label: str,
                ref_gdf, drainage_buf, cached_inputs: dict) -> dict:
    """cached_inputs evita recalcular features+detector para la misma combinación."""
    mode = run_cfg.MODE
    active_detector = run_cfg.DETECTOR if mode != "catalyst_replica" else "canny"
    
    # Generar cache_key incluyendo parámetros críticos
    gthr_p = getattr(run_cfg, "GTHR_PERCENTILE", cfg.GTHR_PERCENTILE)
    gthr_a = getattr(run_cfg, "GTHR_ABSOLUTE", getattr(cfg, "GTHR_ABSOLUTE", None))
    cache_key = (mode, active_detector, gthr_p, gthr_a, label)

    if cache_key not in cached_inputs:
        detector_input, feat_meta = make_detector_input_from_config(dem, run_cfg)
        edges, det_meta = run_detector(
            detector_input, active_detector, run_cfg.RADI_BY_DETECTOR, run_cfg
        )
        # Guardar raster de bordes para inspección visual
        _save_raster(
            edges.astype(np.uint8),
            out_dir / f"edges_{label}.tif",
            dem,
        )
        cached_inputs[cache_key] = (feat_meta, edges, det_meta)
        pct = 100 * edges.sum() / edges.size
        print(f"    [{label}] detector={active_detector} "
              f"bordes={int(edges.sum()):,} ({pct:.2f}%)")
    else:
        feat_meta, edges, det_meta = cached_inputs[cache_key]

    results = {}
    for extractor in run_cfg.EXTRACTORS:
        raw = run_extractor(edges, extractor, run_cfg)
        if not raw:
            print(f"    [{label}/{extractor}] sin segmentos crudos")
            continue

        merged = merge_colinear(raw, run_cfg.ATHR, run_cfg.DTHR)
        filtered = filter_by_length_m(merged, run_cfg.MIN_LENGTH_M, dem)

        if not filtered:
            print(f"    [{label}/{extractor}] 0 tras filtro longitud")
            continue

        geo_segs = pixels_to_geo(filtered, dem)
        attrs = compute_attributes(geo_segs)
        labels_fam, centroids = classify_families(attrs, n_families=4)

        for a, lf, (x0, y0, x1, y1) in zip(attrs, labels_fam, geo_segs):
            a["family"] = lf
            a["detector"] = active_detector
            a["extractor"] = extractor
            a["mode"] = mode
            a["label"] = label

        gdf = segments_to_gdf(geo_segs, dem, attrs)

        if drainage_buf is not None:
            from .drainage import filter_lineaments_by_drainage
            gdf = filter_lineaments_by_drainage(
                gdf, drainage_buf, run_cfg.DRAINAGE_OVERLAP_PCT
            )

        shp_path = out_dir / f"lin_{label}_{extractor}.shp"
        gdf.to_file(shp_path)
        print(f"    [{label}/{extractor}] -> {shp_path.name}  ({len(gdf)} lin.)")

        ext_result = {
            "n_final": int(len(gdf)),
            "centroids_deg": [float(c) for c in centroids],
            "output": str(shp_path),
        }

        if ref_gdf is not None and len(gdf) > 0:
            val = full_validation_report(
                gdf, ref_gdf, out_dir,
                extractor_name=f"{label}_{extractor}",
                buffer_m=run_cfg.VALIDATION_BUFFER_M,
                angle_tol_deg=run_cfg.VALIDATION_ANGLE_TOL,
            )
            ext_result["validation"] = val
            recall = val["buffer_matching"]["recall"]
            corr = val["azimuth"]["azimuth_histogram_correlation"]
            print(f"           recall={recall:.1%}  corr_az={corr:.3f}")

        results[extractor] = ext_result

    return results


# -------------------------------------------------------------------
# Entrada principal
# -------------------------------------------------------------------
def run_pipeline():
    t0 = datetime.now()
    print("=" * 60)
    print(f"  LINEAMIENTOS - PIPELINE PYTHON / {t0:%Y-%m-%d %H:%M}")
    print("=" * 60)

    # Overrides desde la GUI
    gui_env = os.environ.get("GUI_CONFIG_OVERRIDE")
    if gui_env:
        try:
            overrides = json.loads(gui_env)
            print(f"[GUI] Aplicando overrides: {list(overrides.keys())}")
            for k, v in overrides.items():
                if k == "DEM_PATH" or k == "HILLSHADE_PATH":
                    v = Path(v)
                setattr(cfg, k, v)
        except Exception as e:
            print(f"[GUI] Error cargando overrides: {e}")

    print(f"[1] Cargando DEM: {cfg.DEM_PATH}")
    dem = load_dem(cfg.DEM_PATH)
    print(f"    {dem.shape}  |  {dem.pixel_size_m:.3f} m/px  |  {dem.crs}")

    out_dir = _setup_output(Path(cfg.DEM_PATH), cfg.OUTPUT_DIR)
    print(f"    Output -> {out_dir}")

    mode = getattr(cfg, "MODE", "enhanced")

    drainage_buf = None
    if cfg.FILTER_DRAINAGE:
        from .drainage import extract_drainage_buffer
        drainage_buf = extract_drainage_buffer(
            dem, cfg.DRAINAGE_ACCUM_THRESHOLD, cfg.DRAINAGE_BUFFER_M
        )

    ref_gdf = None
    if cfg.CATALYST_REF is not None:
        try:
            ref_gdf = gpd.read_file(cfg.CATALYST_REF)
            if ref_gdf.crs != dem.crs:
                ref_gdf = ref_gdf.to_crs(dem.crs)
            print(f"    Catalyst ref: {len(ref_gdf)} lineamientos")
        except Exception as e:
            print(f"    Catalyst ref no disponible: {e}")

    batch_report = {"timestamp": t0.isoformat(), "runs": {}}
    cached_inputs: dict = {}

    if mode == "batch":
        print(f"\n[BATCH] {len(BATCH_RUNS)} combinaciones")
        print("-" * 60)
        for run_def in BATCH_RUNS:
            label = run_def.get("label", "run")
            overrides = {k: v for k, v in run_def.items() if k != "label"}
            run_cfg = _CfgOverride(cfg, overrides)
            print(f"\n  [RUNNING] {label}")
            results = _run_single(
                dem, out_dir, run_cfg, label, ref_gdf, drainage_buf, cached_inputs
            )
            batch_report["runs"][label] = results
    else:
        active_detector = cfg.DETECTOR if mode == "enhanced" else "canny"
        run_cfg = _CfgOverride(cfg, {"MODE": mode, "DETECTOR": active_detector})
        print(f"\n[RUN] modo={mode}  detector={active_detector}")
        results = _run_single(
            dem, out_dir, run_cfg, mode, ref_gdf, drainage_buf, cached_inputs
        )
        batch_report["runs"][mode] = results

    report_path = out_dir / f"batch_report_{t0:%Y%m%d_%H%M}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(batch_report, f, indent=2, ensure_ascii=False, default=str)

    dt = (datetime.now() - t0).total_seconds()
    print(f"\n{'=' * 60}")
    print(f"  Reporte -> {report_path.name}")
    print(f"  Tiempo total: {dt:.1f} s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_pipeline()
