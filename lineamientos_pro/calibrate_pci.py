"""
Calibra la réplica de PCI LINE contra una corrida real de PCI Geomatica.

  python -m lineamientos_pro.calibrate_pci imagen.tif salida_pci.shp [RADI GTHR LTHR FTHR ATHR DTHR]

Las coordenadas de la salida de PCI se interpretan en la grilla de la imagen
(vértices en esquinas de píxel). Barre sigma = RADI/k y la ganancia del
gradiente y reporta el acuerdo (F1 por longitud, buffer 1.5 px, 20°).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import rasterio

from .pci_line import pci_line, polylines_to_segments, to_uint8
from .validation import gdf_to_segments, match_scores, read_vector


def load(img_path, shp_path):
    with rasterio.open(img_path) as s:
        img = s.read(1)
        nod = s.nodata
        T = s.transform
    valid = img != nod if nod is not None else np.ones(img.shape, bool)
    ref = gdf_to_segments(read_vector(shp_path))
    inv = ~T
    c0, r0 = inv * (ref[:, 0], ref[:, 1])
    c1, r1 = inv * (ref[:, 2], ref[:, 3])
    return to_uint8(img, valid), valid, np.column_stack([c0, r0, c1, r1])


def score(polys, ref, n_ref, L_ref):
    segs = polylines_to_segments(polys)
    m = match_scores(segs, ref, 1.5, 20.0)
    L = np.array([np.hypot(*np.diff(p, axis=0).T).sum() for p in polys]) if polys else np.zeros(1)
    return {"n": len(polys), "n_ratio": len(polys) / n_ref, "med": float(np.median(L)),
            "med_ref": float(np.median(L_ref)), **{k: m[k] for k in ("precision", "recall", "f1")}}


def main(img_path, shp_path, pci=(10, 100, 30, 3, 30, 20),
         sigma_divs=(1.5, 2.0, 2.5, 3.0, 4.0), gains=(0.5, 1, 2, 3, 4, 6, 8)):
    RADI, GTHR, LTHR, FTHR, ATHR, DTHR = pci
    img, valid, ref = load(img_path, shp_path)
    g = read_vector(shp_path)
    n_ref, L_ref = len(g), g.length.values
    print(f"Referencia PCI: {n_ref} polilíneas, mediana {np.median(L_ref):.1f}")
    best = None
    for sd in sigma_divs:
        for gain in gains:
            t = time.time()
            polys, _ = pci_line(img, valid, RADI, GTHR, LTHR, FTHR, ATHR, DTHR, sigma_div=sd, gain=gain)
            s = score(polys, ref, n_ref, L_ref)
            print(f"sigma=RADI/{sd:<4} gain={gain:<4} n={s['n']:6d} ({s['n_ratio']:.2f}x) "
                  f"mediana={s['med']:6.1f}  P={s['precision']:.3f} R={s['recall']:.3f} "
                  f"F1={s['f1']:.3f}  ({time.time() - t:.1f}s)", flush=True)
            if best is None or s["f1"] > best[2]["f1"]:
                best = (sd, gain, s)
    print(f"\nMejor: sigma = RADI/{best[0]}, ganancia {best[1]} -> F1 {best[2]['f1']:.3f}")
    return best


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) < 2:
        print(__doc__)
        sys.exit(1)
    main(a[0], a[1], tuple(int(x) for x in a[2:8]) if len(a) >= 8 else (10, 100, 30, 3, 30, 20))
