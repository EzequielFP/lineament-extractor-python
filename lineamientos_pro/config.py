"""
Parámetros del motor de extracción de lineamientos.

Todas las distancias van en METROS (se convierten a píxeles según la
resolución de trabajo), los ángulos en GRADOS. Los parámetros de la
réplica PCI LINE (RADI, GTHR, LTHR, FTHR, ATHR, DTHR) conservan las
unidades de PCI Geomatica (píxeles / niveles 0-255) para poder comparar.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# Modelos disponibles -------------------------------------------------------
MODELOS = {
    "consenso": "Consenso multi-evidencia (recomendado)",
    "valles": "Valles rectilíneos (Hessiano, concavidades)",
    "crestas": "Crestas rectilíneas (Hessiano, convexidades)",
    "escarpes": "Escarpes / quiebres de pendiente",
    "bordes": "Bordes de sombreado multiazimut",
    "pci_line": "Réplica PCI LINE (1 azimut)",
    "pci_multi": "PCI LINE multiazimut (unión depurada)",
}
EVIDENCIAS = ("valles", "crestas", "escarpes", "bordes")


@dataclass
class Params:
    # --- Entradas / salidas --------------------------------------------
    dem_path: str = ""
    hillshade_path: str = ""          # opcional: imagen para la réplica PCI
    reference_path: str = ""          # opcional: shapefile/gpkg de referencia
    field_path: str = ""              # opcional: CSV/XLSX de datos estructurales de campo
    zona: str = ""                    # nombre de la zona para los títulos de las láminas
    output_dir: str = ""              # vacío -> <carpeta DEM>/Lineamientos_<fecha>
    models: list = field(default_factory=lambda: ["consenso", "pci_line"])

    # --- Preprocesamiento ------------------------------------------------
    work_res_m: float = 0.0           # 0 = resolución nativa del DEM
    z_factor: float = 1.0

    # --- Evidencia topográfica ------------------------------------------
    scales_m: list = field(default_factory=lambda: [3.0, 6.0, 12.0, 24.0])
    hs_azimuths: list = field(default_factory=lambda: [0, 45, 90, 135, 180, 225, 270, 315])
    hs_altitude: float = 35.0
    consensus_weights: dict = field(default_factory=lambda: {
        "valles": 1.0, "crestas": 0.4, "escarpes": 0.8, "bordes": 1.2})

    # --- Filtro orientado de rectitud -----------------------------------
    line_length_m: float = 40.0       # longitud de integración a lo largo
    long_factor: float = 5.0          # añade núcleo de longitud L*factor (fallas largas y tenues)
    n_orient: int = 18                # orientaciones (10° c/u)
    angle_tol_deg: float = 12.0       # tolerancia de concordancia angular
    sensitivity: float = 0.65         # 0 = conservador, 1 = muy sensible

    # --- Vectorización y enlace -----------------------------------------
    fit_tol_m: float = 3.0            # error de ajuste Douglas-Peucker
    min_seed_m: float = 10.0          # longitud mínima de trazo antes de enlazar
    link_gap_m: float = 30.0          # hueco máximo para unir segmentos colineales
    link_angle_deg: float = 12.0      # diferencia angular máxima al unir
    link_lateral_m: float = 4.0       # desplazamiento lateral máximo al unir
    link_gap_rel: float = 0.6         # hueco extra = fracción de la longitud del tramo corto
    link_lateral_rel: float = 0.02    # desfase extra = fracción de la longitud conjunta
    min_length_m: float = 100.0       # longitud mínima final
    min_support: float = 0.45         # fracción mínima de la línea con evidencia
    border_margin_m: float = 0.0      # 0 = automático (1.5 x escala máxima)

    # --- Atributos -------------------------------------------------------
    max_families: int = 6
    drainage: bool = True             # marca lineamientos que siguen drenajes
    drainage_area_m2: float = 20000.0 # área de captación mínima para cauce
    drainage_buffer_m: float = 6.0

    # --- Productos -------------------------------------------------------
    density_cell_m: float = 10.0
    density_radius_m: float = 200.0
    save_shapefiles: bool = True
    save_evidence_rasters: bool = True

    # --- Réplica PCI LINE (unidades PCI) --------------------------------
    pci_azimuth: float = 315.0
    pci_altitude: float = 45.0
    RADI: int = 10
    GTHR: int = 100
    LTHR: int = 30
    FTHR: int = 3
    ATHR: int = 30
    DTHR: int = 20
    pci_multi_azimuths: list = field(default_factory=lambda: [0, 45, 90, 135])

    # --- Validación ------------------------------------------------------
    val_buffer_m: float = 15.0
    val_angle_deg: float = 20.0

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        names = {f.name for f in fields(cls)}
        p = cls()
        for k, v in (d or {}).items():
            if k in names and v is not None:
                setattr(p, k, v)
        return p

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Params":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# Preajustes ------------------------------------------------------------------
PRESETS = {
    "recomendado": {
        "label": "Recomendado: consenso + réplica PCI para comparar",
        "models": ["consenso", "pci_line"],
    },
    "completo": {
        "label": "Completo: todos los modelos y evidencias",
        "models": list(MODELOS),
    },
    "regional": {
        "label": "Regional: estructuras largas, DEM remuestreado",
        "models": ["consenso"],
        "work_res_m": 5.0, "scales_m": [10.0, 20.0, 40.0, 80.0],
        "line_length_m": 150.0, "min_length_m": 300.0, "link_gap_m": 120.0,
        "link_lateral_m": 12.0, "fit_tol_m": 10.0, "min_seed_m": 40.0,
        "density_cell_m": 50.0, "density_radius_m": 1000.0,
    },
    "detalle": {
        "label": "Detalle: LiDAR / DEM < 2 m, rasgos cortos",
        "models": ["consenso", "valles", "escarpes"],
        "scales_m": [2.0, 4.0, 8.0], "line_length_m": 25.0, "long_factor": 4.0,
        "min_length_m": 40.0, "sensitivity": 0.6, "link_gap_m": 15.0, "link_lateral_m": 3.0,
        "fit_tol_m": 2.0, "min_seed_m": 8.0,
    },
    "rapido": {
        "label": "Rápido: vista previa con DEM a 4 m",
        "models": ["consenso"], "work_res_m": 4.0,
        "scales_m": [8.0, 16.0, 32.0], "line_length_m": 80.0,
        "min_length_m": 150.0, "n_orient": 12, "save_evidence_rasters": False,
    },
    "pci": {
        "label": "Solo réplica PCI LINE",
        "models": ["pci_line"], "drainage": False,
    },
}


def params_from_preset(name: str, **overrides) -> Params:
    base = {k: v for k, v in PRESETS.get(name, {}).items() if k != "label"}
    base.update(overrides)
    return Params.from_dict(base)
