"""
Configuración del pipeline de extracción automática de lineamientos.
Parámetros expuestos al inicio, estilo módulo LINE de PCI Geomatica.

Modos disponibles (MODE):
  "catalyst_replica" → hillshade único azimut 0°, Canny con RADI=10,
                        parámetros idénticos a la corrida de Catalyst.
                        Usar para validación y comparación directa.
  "enhanced"         → stack multidireccional + slope + curvatura + PCA,
                        Frangi con dilatación morfológica. Producto final.

Unidades:
  - RADI_*, LTHR, FTHR, DTHR, FRANGI_SCALES, DILATION_RADIUS: píxeles
  - MIN_LENGTH_M, DRAINAGE_BUFFER_M: metros
  - ATHR: grados
  - GTHR_PERCENTILE: percentil [0-100]
"""
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# RUTAS
# ═══════════════════════════════════════════════════════════════════
DEM_PATH = Path(r"E:\Trabajo\Antigravity\Codigos\LIneamientos_Python\LaCruz_DEM.tif")
CATALYST_REF = Path(r"E:\Trabajo\Antigravity\Codigos\LIneamientos_Python\Catalyst_0_Original.shp")
OUTPUT_DIR = None    # None -> crea "Resultados/" junto al DEM

# ═══════════════════════════════════════════════════════════════════
# MODO DE OPERACIÓN  ← cambiar aquí para alternar entre modos
# ═══════════════════════════════════════════════════════════════════
MODE = "batch"   # "catalyst_replica" | "enhanced" | "batch"

# Ruta al hillshade externo — solo se usa si MODE = "external_hillshade"
# Ajustá el nombre del archivo al tuyo
HILLSHADE_PATH = Path(r"E:\Trabajo\Antigravity\Codigos\LIneamientos_Python\0.tif")

# ═══════════════════════════════════════════════════════════════════
# SELECCIÓN DE MÉTODOS
# (en catalyst_replica el detector se fuerza a "canny")
# ═══════════════════════════════════════════════════════════════════
DETECTOR = "canny"                           # "canny" | "phase_congruency" | "frangi"
EXTRACTORS = ["skeleton"]     # subconjunto no vacío

# ═══════════════════════════════════════════════════════════════════
# FEATURE STACK — modo "enhanced"
# ═══════════════════════════════════════════════════════════════════
USE_MULTI_FEATURE_PCA = True
PCA_COMPONENT = 2                 # 1 o 2 (PC2 suele aislar estructura lineal)
AZIMUTHS = [0, 45, 90, 135, 180, 225, 270, 315]
ALTITUDE = 45                     # grados
Z_FACTOR = 1.0
INCLUDE_SLOPE = True
INCLUDE_CURVATURE = True

# ═══════════════════════════════════════════════════════════════════
# FEATURE STACK — modo "catalyst_replica"
# ═══════════════════════════════════════════════════════════════════
CATALYST_AZIMUTH = 0              # azimut del hillshade único (grados)
CATALYST_ALTITUDE = 45            # altitud solar (grados)
CATALYST_Z_FACTOR = 1.0

# ═══════════════════════════════════════════════════════════════════
# PARÁMETROS TIPO PCI (comparables con corrida Catalyst)
# ═══════════════════════════════════════════════════════════════════
RADI_BY_DETECTOR = {
    "canny": 3,               # radio box-filter (ventana 7x7) - preserva gradientes
    "phase_congruency": 0,    # PC hace analisis multi-escala interno
    "frangi": 0,              # Frangi integra suavizado por escala
}

# Canny
CANNY_SIGMA = 1.0                 # sigma interno de skimage.canny
GTHR_ABSOLUTE = None              # None -> usa percentil. 150 = valor Catalyst original
GTHR_PERCENTILE = 60              # percentil del gradiente para umbral bajo
GTHR_RATIO_HIGH = 2.5             # umbral alto = ratio * umbral bajo

# Parametros PCI directos
LTHR = 15                         # longitud minima de curva (px)
FTHR = 2                          # error de ajuste Douglas-Peucker (px)
ATHR = 30                         # diferencia angular merge colineal (grados)
DTHR = 25                         # distancia maxima linkeo (px)
MIN_LENGTH_M = 20

# ═══════════════════════════════════════════════════════════════════
# SKELETON — dilatación morfológica para cerrar gaps
# ═══════════════════════════════════════════════════════════════════
DILATION_RADIUS = 1               # px - engrosa bordes antes de esqueletonizar
                                  # 0 = desactivado (comportamiento anterior)

# ═══════════════════════════════════════════════════════════════════
# FRANGI
# ═══════════════════════════════════════════════════════════════════
FRANGI_SCALES = [2, 4, 6, 8, 12]
FRANGI_BETA = 0.5
FRANGI_BLACK_RIDGES = False

# ═══════════════════════════════════════════════════════════════════
# PHASE CONGRUENCY (Kovesi)
# ═══════════════════════════════════════════════════════════════════
PC_NSCALE = 5
PC_NORIENT = 6
PC_MIN_WAVELENGTH = 3
PC_MULT = 2.1
PC_SIGMA_ONF = 0.55
PC_K = 2.0
PC_THRESHOLD = 0.3

# ═══════════════════════════════════════════════════════════════════
# HOUGH PROBABILÍSTICO
# ═══════════════════════════════════════════════════════════════════
HOUGH_THRESHOLD = 30
HOUGH_LINE_LENGTH = LTHR
HOUGH_LINE_GAP = 25               # era 10 — permite saltar gaps más grandes

# ═══════════════════════════════════════════════════════════════════
# FILTRADO DE DRENAJES (opcional)
# ═══════════════════════════════════════════════════════════════════
FILTER_DRAINAGE = False
DRAINAGE_ACCUM_THRESHOLD = 1000
DRAINAGE_BUFFER_M = 15
DRAINAGE_OVERLAP_PCT = 0.5

# ═══════════════════════════════════════════════════════════════════
# VALIDACIÓN (si CATALYST_REF no es None)
# ═══════════════════════════════════════════════════════════════════
VALIDATION_BUFFER_M = 30
VALIDATION_ANGLE_TOL = 20
