"""
Línea de comandos:

  python -m lineamientos_pro --dem LaCruz_DEM.tif
  python -m lineamientos_pro --dem dem.tif --preset completo --ref catalyst.shp
  python -m lineamientos_pro --dem dem.tif --modelos consenso valles --res 2 --sens 0.7
  python -m lineamientos_pro --config parametros.json
  python -m lineamientos_pro --benchmark 6
"""
from __future__ import annotations

import argparse
import sys

from .config import MODELOS, PRESETS, Params, params_from_preset


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="lineamientos_pro",
                                 description="Extracción automática de lineamientos desde DEM")
    ap.add_argument("--dem", help="DEM GeoTIFF en CRS proyectado (metros)")
    ap.add_argument("--preset", default="recomendado", choices=list(PRESETS))
    ap.add_argument("--config", help="JSON de parámetros (p. ej. parametros.json de otra corrida)")
    ap.add_argument("--modelos", nargs="+", choices=list(MODELOS))
    ap.add_argument("--salida", help="carpeta de salida")
    ap.add_argument("--ref", help="lineamientos de referencia (shp/gpkg) para validar")
    ap.add_argument("--hillshade", help="imagen para la réplica PCI (p. ej. 0.tif)")
    ap.add_argument("--campo", help="datos estructurales de campo (CSV/XLSX: tipo, dir_buz, buzamiento)")
    ap.add_argument("--zona", help="nombre de la zona para los títulos")
    ap.add_argument("--res", type=float, help="resolución de trabajo (m); 0 = nativa")
    ap.add_argument("--sens", type=float, help="sensibilidad 0-1")
    ap.add_argument("--min-long", type=float, help="longitud mínima (m)")
    ap.add_argument("--benchmark", type=int, metavar="N", help="corre el banco sintético con N escenas")
    a = ap.parse_args(argv)

    if a.benchmark:
        from .bench import main as bench
        bench(a.benchmark)
        return 0
    P = Params.load(a.config) if a.config else params_from_preset(a.preset)
    for k, v in (("dem_path", a.dem), ("models", a.modelos), ("output_dir", a.salida),
                 ("reference_path", a.ref), ("hillshade_path", a.hillshade),
                 ("field_path", a.campo), ("zona", a.zona),
                 ("work_res_m", a.res), ("sensitivity", a.sens), ("min_length_m", a.min_long)):
        if v is not None:
            setattr(P, k, v)
    if not P.dem_path:
        ap.error("falta --dem (o dem_path en --config)")
    from .pipeline import run
    res = run(P)
    print("\nResumen:")
    for r in res["summary"]:
        print(f"  {r['modelo']:10s} n={r['n']:5d}  {r['long_total_km']:8.2f} km  "
              f"densidad {r['densidad_km_km2']:.2f} km/km²  {r['familias']}")
    print(f"\nReporte: {res['output_dir']}\reporte.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
