"""
Post-procesamiento de segmentos extraídos.

Funciones:
  - merge_colinear     : encadena segmentos colineales cercanos (ATHR + DTHR)
  - filter_by_length   : descarta segmentos más cortos que un umbral en metros
  - pixels_to_geo      : transforma coords píxel → coords del CRS del DEM
  - compute_attributes : longitud, azimut, familia direccional
  - classify_families  : agrupa azimuts en familias direccionales
"""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans

from .io_dem import DEMData

Segment = tuple[float, float, float, float]


# ───────────────────────────────────────────────────────────────────
# Merge colineales (equivalente a ATHR + DTHR de PCI)
# ───────────────────────────────────────────────────────────────────
def _segment_angle_deg(seg: Segment) -> float:
    """Azimut geográfico en [0, 180). 0° = N (vertical en imagen), 90° = E."""
    x0, y0, x1, y1 = seg
    # En coordenadas de píxel, y aumenta hacia abajo. Convertimos a azimut
    # geográfico: ángulo desde Norte, sentido horario.
    dx = x1 - x0
    dy = -(y1 - y0)  # invertir eje y para que Norte sea arriba
    angle = (np.degrees(np.arctan2(dx, dy))) % 180.0
    return angle


def _angular_diff(a: float, b: float) -> float:
    """Diferencia angular mínima en [0, 90] para líneas sin dirección."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def _segment_length(seg: Segment) -> float:
    x0, y0, x1, y1 = seg
    return float(np.hypot(x1 - x0, y1 - y0))


def _endpoint_distance(s1: Segment, s2: Segment) -> tuple[float, str]:
    """Distancia mínima entre extremos y qué pareja es. Devuelve (dist, tipo)."""
    p1a, p1b = (s1[0], s1[1]), (s1[2], s1[3])
    p2a, p2b = (s2[0], s2[1]), (s2[2], s2[3])
    pairs = {
        "ab": (p1a, p2b, np.hypot(p1a[0] - p2b[0], p1a[1] - p2b[1])),
        "ba": (p1b, p2a, np.hypot(p1b[0] - p2a[0], p1b[1] - p2a[1])),
        "aa": (p1a, p2a, np.hypot(p1a[0] - p2a[0], p1a[1] - p2a[1])),
        "bb": (p1b, p2b, np.hypot(p1b[0] - p2b[0], p1b[1] - p2b[1])),
    }
    best = min(pairs.items(), key=lambda kv: kv[1][2])
    return best[1][2], best[0]


def _merge_two(s1: Segment, s2: Segment) -> Segment:
    """Une dos segmentos tomando los dos extremos más lejanos entre sí."""
    points = [(s1[0], s1[1]), (s1[2], s1[3]), (s2[0], s2[1]), (s2[2], s2[3])]
    max_d = 0.0
    best = (points[0], points[1])
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            if d > max_d:
                max_d = d
                best = (points[i], points[j])
    return (best[0][0], best[0][1], best[1][0], best[1][1])


def merge_colinear(
    segments: list[Segment],
    angle_tol_deg: float,
    dist_tol_px: float,
    max_iterations: int = 10,
) -> list[Segment]:
    """Fusiona iterativamente pares de segmentos colineales y cercanos.

    Orden O(N²) por iteración. Aceptable para miles de segmentos; si crece
    mucho, conviene indexar por R-tree.
    """
    segs = list(segments)
    for _ in range(max_iterations):
        merged_any = False
        used = [False] * len(segs)
        new_segs = []

        for i in range(len(segs)):
            if used[i]:
                continue
            current = segs[i]
            used[i] = True
            ang_i = _segment_angle_deg(current)

            for j in range(i + 1, len(segs)):
                if used[j]:
                    continue
                ang_j = _segment_angle_deg(segs[j])
                if _angular_diff(ang_i, ang_j) > angle_tol_deg:
                    continue
                dist, _ = _endpoint_distance(current, segs[j])
                if dist > dist_tol_px:
                    continue
                current = _merge_two(current, segs[j])
                ang_i = _segment_angle_deg(current)
                used[j] = True
                merged_any = True

            new_segs.append(current)

        segs = new_segs
        if not merged_any:
            break

    return segs


# ───────────────────────────────────────────────────────────────────
# Filtros
# ───────────────────────────────────────────────────────────────────
def filter_by_length_m(
    segments: list[Segment],
    min_length_m: float,
    dem: DEMData,
) -> list[Segment]:
    min_px = min_length_m / dem.pixel_size_m
    return [s for s in segments if _segment_length(s) >= min_px]


# ───────────────────────────────────────────────────────────────────
# Transformación píxel → coordenadas geográficas
# ───────────────────────────────────────────────────────────────────
def pixels_to_geo(segments: list[Segment], dem: DEMData) -> list[Segment]:
    """Transforma (col, row) → (X, Y) usando el affine de rasterio."""
    T = dem.transform
    out = []
    for x0, y0, x1, y1 in segments:
        X0, Y0 = T * (x0, y0)
        X1, Y1 = T * (x1, y1)
        out.append((X0, Y0, X1, Y1))
    return out


# ───────────────────────────────────────────────────────────────────
# Atributos y familias direccionales
# ───────────────────────────────────────────────────────────────────
def _geo_azimuth_deg(x0, y0, x1, y1) -> float:
    """Azimut geográfico N→E en [0, 180)."""
    dx = x1 - x0
    dy = y1 - y0
    # En CRS proyectado, Y aumenta hacia Norte
    angle = (np.degrees(np.arctan2(dx, dy))) % 180.0
    return angle


def compute_attributes(geo_segments: list[Segment]) -> list[dict]:
    attrs = []
    for x0, y0, x1, y1 in geo_segments:
        length = float(np.hypot(x1 - x0, y1 - y0))
        azimuth = _geo_azimuth_deg(x0, y0, x1, y1)
        attrs.append({"length_m": length, "azimuth_deg": azimuth})
    return attrs


def classify_families(
    attrs: list[dict],
    n_families: int = 4,
    weighted_by_length: bool = True,
) -> tuple[list[int], list[float]]:
    """Clasifica lineamientos en familias direccionales vía k-means circular.

    Devuelve (labels, centroids_deg). Los centroides están ordenados por
    azimut ascendente.
    """
    if len(attrs) < n_families:
        return [0] * len(attrs), []

    azimuths = np.array([a["azimuth_deg"] for a in attrs])
    lengths = np.array([a["length_m"] for a in attrs])

    # Proyectamos a círculo duplicado (2*azimuth) para manejar circularidad
    theta = np.deg2rad(2.0 * azimuths)
    X = np.column_stack([np.cos(theta), np.sin(theta)])

    weights = lengths if weighted_by_length else None
    km = KMeans(n_clusters=n_families, n_init=10, random_state=42)
    km.fit(X, sample_weight=weights)
    labels = km.labels_

    # Recuperar ángulos de centroides
    centroids_deg = []
    for c in km.cluster_centers_:
        angle = (np.degrees(np.arctan2(c[1], c[0])) / 2.0) % 180.0
        centroids_deg.append(angle)

    # Reordenar labels por ángulo ascendente
    order = np.argsort(centroids_deg)
    remap = {old: new for new, old in enumerate(order)}
    labels = np.array([remap[l] for l in labels])
    centroids_deg = sorted(centroids_deg)

    return labels.tolist(), centroids_deg
