"""
Benchmark objetivo: python -m lineamientos_pro.bench [n_escenas]

Compara todos los modelos contra la verdad de terreno de escenas sintéticas.
Para ser justos con PCI, además de sus valores por defecto se reporta el
mejor GTHR encontrado por barrido (ventaja que en la práctica no tiene).
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .config import Params
from .evidence import compute_evidences, hillshade
from .models import fuse_evidence, run_evidence_model, run_pci_model, run_pci_multi
from .pci_line import to_uint8
from .synthetic import make_scene
from .validation import match_scores

BUFFER_M = 8.0
ANGLE = 15.0


def run_scene(seed: int, P: Params, pci_sweep=(30, 50, 70, 100, 130, 170, 220), quiet=True):
    sc = make_scene(seed=seed)
    res = sc.res
    valid = np.ones(sc.z.shape, bool)
    log = (lambda *a: None) if quiet else print
    buf = BUFFER_M / res
    out = {}
    t = time.time()
    ev = compute_evidences(sc.z, valid, res, P, {"valles", "crestas", "escarpes", "bordes"}, log)
    for k in ("valles", "crestas", "escarpes", "bordes"):
        r = run_evidence_model(k, *ev[k], valid, res, P, log)
        out[k] = match_scores(r.segs, sc.truth, buf, ANGLE)
    S, T = fuse_evidence(ev, P.consensus_weights, valid)
    r = run_evidence_model("consenso", S, T, valid, res, P, log)
    out["consenso"] = match_scores(r.segs, sc.truth, buf, ANGLE)
    out["consenso"]["n"] = len(r.segs)
    t_pro = time.time() - t

    for az in (315, 0):
        img = to_uint8(hillshade(sc.z, res, az, 45), valid)
        r = run_pci_model(img, valid, P, log=log)
        out[f"pci_line az{az}"] = match_scores(r.segs, sc.truth, buf, ANGLE)
        best = None
        for g in pci_sweep:
            Pg = Params.from_dict({**P.to_dict(), "GTHR": g})
            rr = run_pci_model(img, valid, Pg, log=log)
            m = match_scores(rr.segs, sc.truth, buf, ANGLE)
            if best is None or m["f1"] > best["f1"]:
                best = {**m, "GTHR": g}
        out[f"pci_line az{az} (GTHR óptimo)"] = best
    r = run_pci_multi(sc.z, valid, res, P, log)
    out["pci_multi"] = match_scores(r.segs, sc.truth, buf, ANGLE)
    out["_t_pro"] = t_pro
    return out


def main(n=3, P: Params | None = None, verbose=True):
    P = P or Params()
    rows = {}
    for seed in range(n):
        o = run_scene(seed, P)
        for k, v in o.items():
            if k.startswith("_"):
                continue
            rows.setdefault(k, []).append(v)
    table = {k: {m: float(np.mean([x[m] for x in v])) for m in ("precision", "recall", "f1")}
             for k, v in rows.items()}
    if verbose:
        print(f"\nBenchmark sintético: {n} escenas, tolerancia {BUFFER_M} m / {ANGLE}°")
        print(f"{'modelo':34s} {'precisión':>9s} {'recall':>7s} {'F1':>6s}")
        for k, v in sorted(table.items(), key=lambda kv: -kv[1]["f1"]):
            print(f"{k:34s} {v['precision']:9.2f} {v['recall']:7.2f} {v['f1']:6.2f}")
    return table


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
