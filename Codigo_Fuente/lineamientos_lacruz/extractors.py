"""
Extractores de segmentos de línea a partir de imagen binaria de bordes.

Tres métodos:
  - hough    : Hough Probabilístico (skimage)
  - lsd      : Line Segment Detector (OpenCV contrib, con fallback)
  - skeleton : skeletonización + trazado de caminos + Douglas-Peucker

Todos devuelven lista de segmentos [(x0, y0, x1, y1), ...] en coordenadas píxel
(columna, fila).
"""
from __future__ import annotations
import numpy as np
from skimage.transform import probabilistic_hough_line
from skimage.morphology import skeletonize
from skimage.measure import approximate_polygon


Segment = tuple[float, float, float, float]   # (x0, y0, x1, y1) en píxeles


# ───────────────────────────────────────────────────────────────────
# Hough Probabilístico
# ───────────────────────────────────────────────────────────────────
def extract_hough(
    edges: np.ndarray,
    threshold: int,
    line_length: int,
    line_gap: int,
) -> list[Segment]:
    lines = probabilistic_hough_line(
        edges,
        threshold=threshold,
        line_length=line_length,
        line_gap=line_gap,
    )
    # skimage devuelve ((x0,y0),(x1,y1))
    return [(float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]))
            for p0, p1 in lines]


# ───────────────────────────────────────────────────────────────────
# LSD (OpenCV) con fallback
# ───────────────────────────────────────────────────────────────────
def extract_lsd(edges: np.ndarray, min_length_px: float) -> list[Segment]:
    """LSD de OpenCV. Si no está disponible, cae a Hough como fallback.

    Nota: LSD espera imagen en escala de grises, no binaria. Multiplicamos
    la máscara por 255 para darle una entrada razonable.
    """
    img_u8 = (edges.astype(np.uint8) * 255)

    try:
        import cv2
        # LSD fue removido en OpenCV 4.1+ por licencia. Disponible vía
        # cv2.ximgproc en opencv-contrib-python.
        if hasattr(cv2, "createLineSegmentDetector"):
            lsd = cv2.createLineSegmentDetector()
        elif hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector"):
            lsd = cv2.ximgproc.createFastLineDetector(
                length_threshold=int(min_length_px),
                do_merge=True,
            )
        else:
            raise AttributeError("LSD/FLD no disponible en esta build de OpenCV")

        result = lsd.detect(img_u8)
        if result is None:
            return []
        lines = result[0] if isinstance(result, tuple) else result
        if lines is None:
            return []

        segments = []
        for ln in lines.reshape(-1, 4):
            x0, y0, x1, y1 = ln
            length = np.hypot(x1 - x0, y1 - y0)
            if length >= min_length_px:
                segments.append((float(x0), float(y0), float(x1), float(y1)))
        return segments

    except (ImportError, AttributeError) as e:
        print(f"  [LSD] No disponible ({e}); usando Hough como fallback.")
        return extract_hough(edges, threshold=30, line_length=int(min_length_px), line_gap=10)


# ───────────────────────────────────────────────────────────────────
# Skeleton + path tracing + Douglas-Peucker
# ───────────────────────────────────────────────────────────────────
def _trace_skeleton_paths(skel: np.ndarray) -> list[list[tuple[int, int]]]:
    """Traza caminos del esqueleto partiendo de endpoints y bifurcaciones.

    Algoritmo:
      1. Clasificar cada píxel del esqueleto por su número de vecinos 8-conectados:
         - 1 vecino: endpoint
         - 2 vecinos: píxel de camino
         - ≥3 vecinos: bifurcación
      2. Partir caminos en endpoints y bifurcaciones.
      3. Devolver listas de (row, col) por camino.
    """
    skel = skel.astype(bool)
    H, W = skel.shape

    # Contar vecinos 8-conectados vía convolución
    from scipy.ndimage import convolve
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbors = convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)
    neighbors[~skel] = 0

    endpoints = np.argwhere((neighbors == 1) & skel)
    junctions = np.argwhere((neighbors >= 3) & skel)

    visited = np.zeros_like(skel, dtype=bool)
    # Las bifurcaciones rompen caminos pero no se recorren como parte de ellos
    for r, c in junctions:
        visited[r, c] = True

    paths = []

    def walk(start: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start]
        r, c = start
        visited[r, c] = True
        while True:
            found = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and skel[nr, nc] and not visited[nr, nc]:
                        found = (nr, nc)
                        break
                if found:
                    break
            if not found:
                break
            path.append(found)
            r, c = found
            visited[r, c] = True
            # Si llegamos a una bifurcación (marcada como visitada previamente),
            # el loop se detiene en la siguiente iteración porque todos los
            # vecinos no visitados ya están agotados.
            if neighbors[r, c] >= 3:
                break
        return path

    # Desde endpoints primero
    for r, c in endpoints:
        if not visited[r, c]:
            paths.append(walk((int(r), int(c))))

    # Luego desde vecinos de bifurcaciones (para caminos entre junctions)
    for jr, jc in junctions:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = jr + dr, jc + dc
                if 0 <= nr < skel.shape[0] and 0 <= nc < skel.shape[1] \
                        and skel[nr, nc] and not visited[nr, nc]:
                    paths.append(walk((int(nr), int(nc))))

    # Finalmente, loops aislados (ciclos sin endpoints ni junctions)
    remaining = np.argwhere(skel & ~visited)
    for r, c in remaining:
        if not visited[r, c]:
            paths.append(walk((int(r), int(c))))

    return paths


def extract_skeleton(
    edges: np.ndarray,
    min_length_px: float,
    fitting_error_px: float,
    dilation_radius: int = 0,
) -> list[Segment]:
    """Skeletonización + tracing + aproximación poligonal Douglas-Peucker.

    dilation_radius > 0 aplica dilatación morfológica antes de esqueletonizar,
    cerrando pequeños gaps en los bordes detectados. Es el fix principal para
    mejorar la continuidad de lineamientos respecto a la versión original.
    """
    if dilation_radius > 0:
        from skimage.morphology import dilation, disk
        edges = dilation(edges, disk(dilation_radius))

    skel = skeletonize(edges)
    paths = _trace_skeleton_paths(skel)

    segments: list[Segment] = []
    for path in paths:
        if len(path) < 2:
            continue
        # path viene como (row, col); para Douglas-Peucker usamos (x, y) = (col, row)
        coords = np.array([(c, r) for r, c in path], dtype=np.float64)
        approx = approximate_polygon(coords, tolerance=fitting_error_px)
        if len(approx) < 2:
            continue
        for i in range(len(approx) - 1):
            x0, y0 = approx[i]
            x1, y1 = approx[i + 1]
            length = np.hypot(x1 - x0, y1 - y0)
            if length >= min_length_px:
                segments.append((float(x0), float(y0), float(x1), float(y1)))

    return segments


# ───────────────────────────────────────────────────────────────────
# Dispatcher
# ───────────────────────────────────────────────────────────────────
def run_extractor(edges: np.ndarray, extractor: str, cfg) -> list[Segment]:
    if extractor == "hough":
        return extract_hough(
            edges,
            threshold=cfg.HOUGH_THRESHOLD,
            line_length=cfg.HOUGH_LINE_LENGTH,
            line_gap=cfg.HOUGH_LINE_GAP,
        )
    if extractor == "lsd":
        return extract_lsd(edges, min_length_px=cfg.LTHR)
    if extractor == "skeleton":
        return extract_skeleton(
            edges,
            min_length_px=cfg.LTHR,
            fitting_error_px=cfg.FTHR,
            dilation_radius=getattr(cfg, "DILATION_RADIUS", 0),
        )
    raise ValueError(f"Extractor desconocido: {extractor}")
