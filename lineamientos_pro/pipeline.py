"""
Orquestador: DEM -> modelos -> atributos -> productos.

Estructura de salida (carpeta por corrida):
  lineamientos.gpkg          capas lin_<modelo>, cruces_<principal>
  shp/lin_<modelo>.shp       mismas capas en shapefile (ArcGIS / PCI)
  rasters/                   sombreado, respuesta de línea, densidades, drenaje
  figuras/                   mapas, rosetas, histogramas
  tablas/                    resumen, familias, validación (CSV)
  reporte.html               reporte autocontenido
  parametros.json            configuración exacta (reproducible)
  preview/                   datos ligeros para el visor de la interfaz
"""
from __future__ import annotations

import csv
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from . import figures as figs_mod
from . import products as prod
from . import stats as st
from .attributes import build_attributes, drainage_distance, geo_azimuth, rumbo_txt, sector
from .config import EVIDENCIAS, Params
from .evidence import compute_evidences, hillshade, multidirectional_hillshade
from .models import fuse_evidence, run_evidence_model, run_pci_model, run_pci_multi
from .pci_line import to_uint8
from .raster import load_dem, load_image_native, save_raster
from .validation import compare, gdf_to_segments, read_vector


class Cancelled(Exception):
    pass


def _out_dir(P: Params, dem: Path) -> Path:
    if P.output_dir:
        base = Path(P.output_dir)
    else:
        base = dem.parent
    out = base / f"Lineamientos_{dem.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    for sub in ("shp", "rasters", "figuras", "tablas", "preview"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    return out


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        w.writeheader()
        w.writerows(rows)


def _segs_to_geo(segs_px, grid):
    X0, Y0 = grid.px2geo(segs_px[:, 0], segs_px[:, 1])
    X1, Y1 = grid.px2geo(segs_px[:, 2], segs_px[:, 3])
    return np.column_stack([X0, Y0, X1, Y1])


def _poly_to_geo(poly, grid):
    X, Y = grid.px2geo(poly[:, 0], poly[:, 1], center=False)   # convención PCI
    return LineString(np.column_stack([X, Y]))


def run(P: Params, log=print, progress=None, cancel=None) -> dict:
    """Ejecuta el flujo completo. Devuelve un resumen (dict)."""
    t0 = time.time()
    prog = progress or (lambda f, m="": None)

    def step(frac, msg):
        if cancel is not None and cancel():
            raise Cancelled()
        prog(frac, msg)
        log(msg)

    dem_path = Path(P.dem_path)
    if not dem_path.exists():
        raise FileNotFoundError(f"No existe el DEM: {dem_path}")
    models = [m for m in P.models if m]
    if not models:
        raise ValueError("Seleccione al menos un modelo.")

    step(0.02, f"[1/6] Cargando DEM {dem_path.name}")
    grid = load_dem(dem_path, P.work_res_m)
    out = _out_dir(P, dem_path)
    P.save(out / "parametros.json")
    area_km2 = grid.valid.sum() * grid.res ** 2 / 1e6
    log(f"    {grid.shape[1]} x {grid.shape[0]} px · {grid.res:.2f} m/px · "
        f"{area_km2:.2f} km² válidos · salida: {out}")

    step(0.08, "[2/6] Sombreado multidireccional")
    hs_multi = multidirectional_hillshade(grid.z, grid.res, [315, 45, 135, 225, 0, 90, 180, 270],
                                          45, P.z_factor)
    hs_multi[~grid.valid] = 1.0
    files = []
    files.append(save_raster(out / "rasters" / "sombreado_multidireccional.tif",
                             (hs_multi * 255).astype(np.uint8), grid))

    evid_models = [m for m in models if m in EVIDENCIAS or m == "consenso"]
    wanted = set(EVIDENCIAS) if "consenso" in models else set(evid_models)
    ev = {}
    if wanted:
        step(0.12, "[3/6] Evidencias topográficas: " + ", ".join(sorted(wanted)))
        ev = compute_evidences(grid.z, grid.valid, grid.res, P, wanted, log)

    drain_dist, streams = None, None
    if P.drainage and evid_models:
        step(0.30, "    Red de drenaje (D8) para marcar lineamientos de drenaje")
        drain_dist, streams = drainage_distance(grid, P.drainage_area_m2, out, log)
        if streams is not None:
            files.append(save_raster(out / "rasters" / "red_drenaje.tif", streams, grid))

    results = {}
    cache = {}
    n_models = len(models)
    for i, m in enumerate(models):
        step(0.35 + 0.35 * i / n_models, f"[4/6] Modelo {m}")
        try:
            polys = None
            if m == "consenso":
                S, T = fuse_evidence(ev, P.consensus_weights, grid.valid)
                r = run_evidence_model(m, S, T, grid.valid, grid.res, P, log)
                strength = r.rasters["respuesta"]
            elif m in EVIDENCIAS:
                r = run_evidence_model(m, *ev[m], grid.valid, grid.res, P, log)
                strength = r.rasters["respuesta"]
            elif m in ("pci_line", "pci_multi"):
                # PCI trabaja en píxeles: se ejecuta a resolución nativa, como en Geomatica
                if m == "pci_line" and P.hillshade_path:
                    mgrid = load_image_native(P.hillshade_path, fill=False)
                    img = mgrid.z
                    log(f"    imagen PCI: {Path(P.hillshade_path).name} ({mgrid.res:.2f} m/px)")
                else:
                    mgrid = grid if not P.work_res_m else _native_grid(dem_path, grid, cache)
                    img = hillshade(mgrid.z, mgrid.res, P.pci_azimuth, P.pci_altitude, P.z_factor)
                    if m == "pci_line":
                        log(f"    imagen PCI: sombreado azimut {P.pci_azimuth:g}°, altitud "
                            f"{P.pci_altitude:g}° ({mgrid.res:.2f} m/px)")
                if m == "pci_line":
                    r = run_pci_model(to_uint8(img, mgrid.valid), mgrid.valid, P, log=log)
                    strength = r.rasters["bordes"] / 255.0
                    polys = r.polylines
                else:
                    r = run_pci_multi(mgrid.z, mgrid.valid, mgrid.res, P, log)
                    strength = None
            else:
                log(f"    modelo desconocido: {m}")
                continue
        except Cancelled:
            raise
        except Exception as e:
            log(f"    ERROR en {m}: {e}")
            log(traceback.format_exc())
            continue

        segs = r.segs
        is_pci = m in ("pci_line", "pci_multi")
        agrid = mgrid if is_pci else grid
        rows, fams = build_attributes(
            segs, agrid, P, m, strength_map=strength, support=r.support,
            evidences=None if is_pci else ev, drain_dist=None if is_pci else drain_dist)
        if polys is not None:
            # PCI: se conservan polilíneas originales (como Catalyst); atributos
            # agregados por polilínea a partir de sus tramos.
            geoms, prow = [], []
            k = 0
            for p in polys:
                nseg = len(p) - 1
                sub = rows[k:k + nseg]
                k += nseg
                g = _poly_to_geo(p, agrid)
                X0, Y0 = g.coords[0]
                X1, Y1 = g.coords[-1]
                az = float(geo_azimuth(X0, Y0, X1, Y1))
                fam = max(sub, key=lambda s: s["longitud_m"])["familia"] if sub else ""
                prow.append({"modelo": m, "longitud_m": round(g.length, 1), "azimut": round(az, 1),
                             "rumbo": rumbo_txt(az), "sector": sector(az), "familia": fam,
                             "tipo": "borde_sombreado", "n_vert": len(p),
                             "fuerza": round(float(np.mean([s["fuerza"] or 0 for s in sub])), 3) if sub else None,
                             "confianza": round(float(np.mean([s["confianza"] for s in sub])), 3) if sub else None})
                geoms.append(g)
            gdf = gpd.GeoDataFrame(prow, geometry=geoms, crs=grid.crs)
        else:
            geo = _segs_to_geo(segs, agrid) if len(segs) else np.zeros((0, 4))
            geoms = [LineString([(a, b), (c, d)]) for a, b, c, d in geo]
            gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs=grid.crs)
        if len(gdf):
            gdf.insert(0, "id", np.arange(1, len(gdf) + 1))
        results[m] = {"gdf": gdf, "familias": fams, "strength": strength, "grid": agrid}
        log(f"    -> {len(gdf)} lineamientos · {gdf.length.sum() / 1000 if len(gdf) else 0:.2f} km")

    if not results:
        raise RuntimeError("Ningún modelo produjo resultados.")

    # ------------------------------------------------------------ vectores
    step(0.72, "[5/6] Guardando vectores, rásteres y tablas")
    gpkg = out / "lineamientos.gpkg"
    for m, r in results.items():
        g = r["gdf"]
        if len(g) == 0:
            continue
        g.to_file(gpkg, layer=f"lin_{m}", driver="GPKG")
        if P.save_shapefiles:
            g.to_file(out / "shp" / f"lin_{m}.shp", encoding="utf-8")
    files.append(gpkg)

    main = "consenso" if "consenso" in results else next(iter(results))
    gmain = results[main]["gdf"]
    segs_main = gdf_to_segments(gmain) if len(gmain) else np.zeros((0, 4))
    pts = prod.intersections(segs_main)
    if len(pts):
        gp = gpd.GeoDataFrame({"id": np.arange(1, len(pts) + 1)},
                              geometry=[Point(x, y) for x, y in pts], crs=grid.crs)
        gp.to_file(gpkg, layer=f"cruces_{main}", driver="GPKG")
    dens, Td = prod.length_density(segs_main, grid, P.density_cell_m, P.density_radius_m)
    files.append(save_raster(out / "rasters" / f"densidad_longitud_{main}.tif", dens, grid, transform=Td))
    cdens, Tc = prod.point_density(pts, grid, P.density_cell_m, P.density_radius_m)
    files.append(save_raster(out / "rasters" / f"densidad_cruces_{main}.tif", cdens, grid, transform=Tc))
    mids = (segs_main[:, :2] + segs_main[:, 2:]) / 2 if len(segs_main) else np.zeros((0, 2))
    if len(gmain):   # centro de cada lineamiento (no de cada tramo de polilínea)
        mids = np.array([[g.interpolate(0.5, normalized=True).x, g.interpolate(0.5, normalized=True).y]
                         for g in gmain.geometry])
    fdens, Tf = prod.point_density(mids, grid, P.density_cell_m, P.density_radius_m)
    files.append(save_raster(out / "rasters" / f"densidad_frecuencia_{main}.tif", fdens, grid, transform=Tf))
    if P.save_evidence_rasters:
        for m, r in results.items():
            if r["strength"] is not None:
                files.append(save_raster(out / "rasters" / f"respuesta_{m}.tif",
                                         np.clip(r["strength"], 0, 3).astype(np.float32), r["grid"]))

    # ------------------------------------------------------------ tablas
    summary, fam_tables, full_stats = [], {}, []
    for m, r in results.items():
        g = r["gdf"]
        ms = st.model_stats(g, area_km2, len(pts) if m == main else None)
        full_stats.append({"modelo": m, **ms})
        row = {"modelo": m, "n": ms.get("n", 0), "long_total_km": ms.get("total_km", 0),
               "long_media_m": ms.get("media_m", 0), "long_mediana_m": ms.get("mediana_m", 0),
               "densidad_km_km2": ms.get("densidad_km_km2", 0),
               "n_conf_alta": int((g["clase_conf"] == "Alta").sum()) if "clase_conf" in g else 0,
               "familias": ", ".join(f"{f['familia']} {f['rumbo']}" for f in r["familias"][:4])}
        if "pct_long_drenaje" in ms:
            row["pct_long_drenaje"] = ms["pct_long_drenaje"]
        summary.append(row)
        fam_tables[m] = st.family_stats(g)
        _write_csv(out / "tablas" / f"familias_{m}.csv", fam_tables[m])
    _write_csv(out / "tablas" / "resumen_modelos.csv", summary)
    _write_csv(out / "tablas" / "estadisticas_modelos.csv", full_stats)

    validation = []
    ref_segs = None
    if P.reference_path:
        try:
            ref = read_vector(P.reference_path)
            if ref.crs is not None and grid.crs is not None and ref.crs != grid.crs:
                ref = ref.to_crs(grid.crs)
            ref_segs = gdf_to_segments(ref)
            results_ref = {"gdf": ref.assign(
                longitud_m=ref.length, azimut=[_az_line(g) for g in ref.geometry]), "familias": []}
            log(f"    referencia: {Path(P.reference_path).name} ({len(ref)} entidades)")
        except Exception as e:
            log(f"    no se pudo leer la referencia: {e}")
            ref_segs = None
    for m, r in results.items():
        if ref_segs is not None and len(r["gdf"]):
            c = compare(gdf_to_segments(r["gdf"]), ref_segs, P.val_buffer_m, P.val_angle_deg)
            validation.append({"modelo": m, "contra": "referencia", **_fmt_val(c)})
    if "consenso" in results and "pci_line" in results and len(results["pci_line"]["gdf"]):
        c = compare(gdf_to_segments(results["consenso"]["gdf"]),
                    gdf_to_segments(results["pci_line"]["gdf"]), P.val_buffer_m, P.val_angle_deg)
        validation.append({"modelo": "consenso", "contra": "pci_line", **_fmt_val(c)})
    _write_csv(out / "tablas" / "validacion.csv", validation)

    # ------------------------------------------------------------ figuras
    step(0.85, "[6/6] Figuras, visor y reporte")
    T = grid.transform
    extent = (T.c, T.c + T.a * grid.shape[1], T.f + T.e * grid.shape[0], T.f)
    f = max(1, int(np.ceil(max(grid.shape) / 2000)))
    bg = hs_multi[::f, ::f]

    def _ext(T_, a):
        return (T_.c, T_.c + T_.a * a.shape[1], T_.f + T_.e * a.shape[0], T_.f)
    dens_r = [(dens, _ext(Td, dens), "km/km²", "Densidad de longitud"),
              (fdens, _ext(Tf, fdens), "lineamientos/km²", "Densidad de frecuencia y cruces")]
    field = None
    if P.field_path:
        try:
            field = st.read_field_data(P.field_path)
            log(f"    datos de campo: {Path(P.field_path).name} ({len(field)} medidas)")
        except Exception as e:
            log(f"    no se pudieron leer los datos de campo: {e}")
    plot_results = dict(results)
    if ref_segs is not None:
        from .attributes import find_families
        _, rf = find_families(results_ref["gdf"]["azimut"].values % 180,
                              results_ref["gdf"]["longitud_m"].values, P.max_families)
        plot_results["referencia"] = {**results_ref, "familias": rf}
    fig_list, fig_extra = figs_mod.make_all(
        out / "figuras", plot_results, main, bg, extent, area_km2, dens_r, pts, P, field,
        {"zona": P.zona, "fuente": dem_path.name}, log)
    _write_excel(out / "tablas" / "estadisticas.xlsx", summary, full_stats, fam_tables, gmain, main,
                 validation, fig_extra, log)
    figs = {}
    for fname, title in fig_list:
        figs[title] = _thumb(out / "figuras" / fname)

    _write_preview(out / "preview", bg, extent, results)

    elapsed = time.time() - t0
    meta = {"dem": str(dem_path), "fecha": f"{datetime.now():%Y-%m-%d %H:%M}", "res": grid.res,
            "shape": f"{grid.shape[1]}×{grid.shape[0]}", "tiempo_s": elapsed}
    all_files = [str(p.relative_to(out)) for p in sorted(out.rglob("*"))
                 if p.is_file() and p.parent.name != "preview"]
    prod.html_report(out / "reporte.html", meta, summary, fam_tables, figs, validation,
                     all_files, P.to_dict())
    step(1.0, f"Listo en {elapsed:.0f} s -> {out}")
    res = {"output_dir": str(out), "summary": summary, "validation": validation,
           "families": fam_tables, "elapsed_s": elapsed, "main": main,
           "figures": fig_list, "stats": full_stats,
           "campo": _jsonable(fig_extra.get("campo"))}
    (out / "preview" / "resumen.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                  encoding="utf-8")
    return res


def _thumb(path, max_px=1600) -> bytes:
    """JPEG reducido para incrustar en el reporte HTML (las láminas son PNG de 300 dpi)."""
    import io

    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def _write_excel(path, summary, full_stats, fam_tables, gmain, main, validation, extra, log):
    """Libro de Excel con todas las tablas de la corrida."""
    try:
        import pandas as pd
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            pd.DataFrame(summary).to_excel(xw, sheet_name="Resumen", index=False)
            pd.DataFrame(full_stats).to_excel(xw, sheet_name="Estadisticas", index=False)
            for m, rows in fam_tables.items():
                if rows:
                    pd.DataFrame(rows).to_excel(xw, sheet_name=f"Familias_{m}"[:31], index=False)
            if len(gmain):
                gmain.drop(columns="geometry").to_excel(xw, sheet_name=f"Lineamientos_{main}"[:31], index=False)
            if validation:
                pd.DataFrame(validation).to_excel(xw, sheet_name="Validacion", index=False)
            if extra.get("acuerdo"):
                a = extra["acuerdo"]
                pd.DataFrame(a["f1"], index=a["names"], columns=a["names"]).to_excel(xw, sheet_name="Acuerdo_modelos")
            if extra.get("campo") and extra["campo"].get("tabla"):
                pd.DataFrame(extra["campo"]["tabla"]).to_excel(xw, sheet_name="Campo_vs_DEM", index=False)
                pd.DataFrame(extra["campo"]["familias_campo"]).to_excel(xw, sheet_name="Familias_campo", index=False)
            for ws in xw.book.worksheets:          # ancho de columnas legible
                for col in ws.columns:
                    w = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                    ws.column_dimensions[col[0].column_letter].width = min(max(10, w + 2), 60)
    except Exception as e:
        log(f"    (Excel no generado: {e})")


def _native_grid(dem_path, grid, cache):
    if "native" not in cache:
        cache["native"] = load_dem(dem_path, 0)
    return cache["native"]


def _az_line(g):
    try:
        c = np.asarray(g.coords if hasattr(g, "coords") else g.geoms[0].coords)
        return float(geo_azimuth(c[0, 0], c[0, 1], c[-1, 0], c[-1, 1]))
    except Exception:
        return np.nan


def _fmt_val(c: dict) -> dict:
    keys = ("precision", "recall", "f1", "rose_correlation", "n_pred", "n_ref",
            "median_len_pred", "median_len_ref")
    return {k: (round(c[k], 3) if isinstance(c.get(k), float) else c.get(k)) for k in keys}


def _write_preview(folder: Path, bg, extent, results):
    """Fondo PNG + líneas en coordenadas del PNG para el visor interactivo."""
    import cv2
    img = np.clip(bg * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(folder / "fondo.png"), img)
    h, w = img.shape
    x0, x1, y0, y1 = extent
    sx, sy = w / (x1 - x0), h / (y1 - y0)
    data = {"width": w, "height": h, "extent": extent, "layers": {}}
    for m, r in results.items():
        feats = []
        for row, geom in zip(r["gdf"].drop(columns="geometry").to_dict("records"), r["gdf"].geometry):
            c = np.asarray(geom.coords)
            px = np.column_stack([(c[:, 0] - x0) * sx, (y1 - c[:, 1]) * sy]).round(1).tolist()
            feats.append({"c": px, "p": {k: v for k, v in row.items()
                                          if k in ("id", "longitud_m", "rumbo", "familia", "tipo",
                                                   "confianza", "clase_conf", "drenaje", "n_evid")}})
        data["layers"][m] = {"features": feats, "familias": r["familias"]}
    (folder / "lineas.json").write_text(json.dumps(data, ensure_ascii=False, default=float),
                                        encoding="utf-8")
