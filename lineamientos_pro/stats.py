"""
Estadística estructural de lineamientos y de datos de campo.

Direcciones axiales (0-180°): se trabajan con ángulo doble.
  media axial, longitud resultante R, desviación circular, prueba de
  Rayleigh (¿hay orientación preferente?).
Longitudes: descriptivos + ajuste de ley de potencia a la distribución
acumulada complementaria (N(>L) ~ L^-a), habitual en redes de fracturas.
Red: densidad (km/km²), frecuencia (n/km²), cruces por lineamiento.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Estadística circular axial
# ---------------------------------------------------------------------------
def axial_stats(az_deg, w=None) -> dict:
    az = np.asarray(az_deg, float) % 180
    n = len(az)
    if n == 0:
        return {"n": 0}
    w = np.ones(n) if w is None else np.asarray(w, float)
    a2 = np.deg2rad(2 * az)
    C, S = (w * np.cos(a2)).sum() / w.sum(), (w * np.sin(a2)).sum() / w.sum()
    R = float(np.hypot(C, S))
    mean = float(np.degrees(np.arctan2(S, C)) / 2 % 180)
    circ_std = float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-12)))) / 2)
    # Rayleigh sobre ángulo doble, con n = número de datos (sin ponderar)
    Ru = np.hypot(np.cos(a2).mean(), np.sin(a2).mean())
    z = n * Ru ** 2
    p = float(np.exp(-z) * (1 + (2 * z - z ** 2) / (4 * n) - (24 * z - 132 * z ** 2 + 76 * z ** 3 - 9 * z ** 4) / (288 * n ** 2)))
    return {"n": n, "azimut_medio": round(mean, 1), "R": round(R, 3),
            "desv_circular": round(circ_std, 1), "rayleigh_z": round(float(z), 2),
            "rayleigh_p": float(np.clip(p, 0, 1))}


def rose_counts(az_deg, w=None, bin_deg=10):
    bins = np.arange(0, 180 + bin_deg, bin_deg)
    h, _ = np.histogram(np.asarray(az_deg) % 180, bins=bins, weights=w)
    return h, bins


def rose_correlation(az_a, w_a, az_b, w_b, bin_deg=10) -> float:
    ha, _ = rose_counts(az_a, w_a, bin_deg)
    hb, _ = rose_counts(az_b, w_b, bin_deg)
    if ha.sum() == 0 or hb.sum() == 0:
        return float("nan")
    return float(np.corrcoef(ha / ha.sum(), hb / hb.sum())[0, 1])


# ---------------------------------------------------------------------------
# Longitudes
# ---------------------------------------------------------------------------
def power_law_fit(L, xmin=None) -> dict:
    """Ajuste log-log de N(>L) para L >= xmin (por defecto la mediana)."""
    L = np.sort(np.asarray(L, float))
    L = L[L > 0]
    if len(L) < 8:
        return {"exponente": float("nan"), "r2": float("nan"), "xmin": float("nan")}
    xmin = float(np.median(L)) if xmin is None else xmin
    x = L[L >= xmin]
    ccdf = 1.0 - np.arange(len(x)) / len(x)
    lx, ly = np.log10(x), np.log10(ccdf)
    b, a = np.polyfit(lx, ly, 1)
    pred = a + b * lx
    r2 = 1 - ((ly - pred) ** 2).sum() / max(((ly - ly.mean()) ** 2).sum(), 1e-12)
    # MLE (Clauset) como referencia
    alpha = 1 + len(x) / np.log(x / xmin).sum() if (x > xmin).any() else float("nan")
    return {"exponente": round(float(-b), 3), "r2": round(float(r2), 3), "xmin": round(xmin, 1),
            "alpha_mle": round(float(alpha), 3), "intercepto": float(a)}


def length_stats(L) -> dict:
    L = np.asarray(L, float)
    if len(L) == 0:
        return {}
    return {"n": int(len(L)), "total_km": round(L.sum() / 1000, 3), "media_m": round(L.mean(), 1),
            "mediana_m": round(float(np.median(L)), 1), "desv_m": round(L.std(), 1),
            "p10_m": round(float(np.percentile(L, 10)), 1), "p90_m": round(float(np.percentile(L, 90)), 1),
            "max_m": round(L.max(), 1)}


# ---------------------------------------------------------------------------
# Tablas por modelo y por familia
# ---------------------------------------------------------------------------
def model_stats(gdf, area_km2: float, n_intersections: int | None = None) -> dict:
    if len(gdf) == 0:
        return {"n": 0}
    L = gdf["longitud_m"].values
    az = gdf["azimut"].values
    out = {**length_stats(L)}
    ax = axial_stats(az, L)
    out.update({"azimut_medio": ax["azimut_medio"], "R": ax["R"],
                "desv_circular": ax["desv_circular"], "rayleigh_p": ax["rayleigh_p"]})
    out["densidad_km_km2"] = round(L.sum() / 1000 / max(area_km2, 1e-9), 3)
    out["frecuencia_n_km2"] = round(len(L) / max(area_km2, 1e-9), 2)
    pl = power_law_fit(L)
    out["exp_ley_potencia"] = pl["exponente"]
    out["r2_ley_potencia"] = pl["r2"]
    if n_intersections is not None:
        out["cruces"] = int(n_intersections)
        out["cruces_por_lin"] = round(2 * n_intersections / max(len(L), 1), 2)
    if "clase_conf" in gdf:
        for c in ("Alta", "Media", "Baja"):
            m = (gdf["clase_conf"] == c).values
            out[f"pct_long_{c.lower()}"] = round(100 * L[m].sum() / max(L.sum(), 1e-9), 1)
    if "sigue_dren" in gdf and gdf["sigue_dren"].notna().any():
        out["pct_long_drenaje"] = round(100 * L[gdf["sigue_dren"].fillna(False).values.astype(bool)].sum()
                                        / max(L.sum(), 1e-9), 1)
    return out


def family_stats(gdf) -> list[dict]:
    rows = []
    if len(gdf) == 0 or "familia" not in gdf:
        return rows
    Ltot = gdf["longitud_m"].sum()
    for fam, g in sorted(gdf.groupby("familia"), key=lambda kv: int(str(kv[0])[1:] or 0)):
        L = g["longitud_m"].values
        ax = axial_stats(g["azimut"].values, L)
        r = {"familia": fam, "azimut_medio": ax["azimut_medio"], "rumbo": _rumbo(ax["azimut_medio"]),
             "desv_circular": ax["desv_circular"], "n": int(len(g)),
             "long_km": round(L.sum() / 1000, 2), "pct_long": round(100 * L.sum() / max(Ltot, 1e-9), 1),
             "long_media_m": round(L.mean(), 1), "long_max_m": round(L.max(), 1)}
        if "tipo" in g:
            r["tipo_dominante"] = g.groupby("tipo")["longitud_m"].sum().idxmax()
        if "confianza" in g and g["confianza"].notna().any():
            r["confianza_media"] = round(float(g["confianza"].mean()), 2)
        rows.append(r)
    return rows


def _rumbo(az):
    az = float(az) % 180
    return f"N{az:02.0f}E" if az <= 90 else f"N{180 - az:02.0f}W"


# ---------------------------------------------------------------------------
# Datos estructurales de campo
# ---------------------------------------------------------------------------
_ALIASES = {
    "tipo": ("tipo", "type", "estructura", "clase"),
    "dd": ("dd", "dir_buz", "dirbuz", "dip_dir", "dipdir", "dip_direction", "direccion_buzamiento",
           "azimut_buzamiento", "dirección_buzamiento"),
    "strike": ("strike", "rumbo", "rumbo_az", "direccion", "dirección"),
    "dip": ("dip", "buz", "buzamiento", "inclinacion", "inclinación"),
    "id": ("id", "estacion", "estación", "punto"),
}


def read_field_data(path):
    """CSV/XLSX con columnas tipo + (dir. de buzamiento | rumbo) + buzamiento.
    Devuelve columnas: id, tipo, dd, dip, strike (regla de la mano derecha)."""
    import pandas as pd
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, sep=None, engine="python", encoding="utf-8-sig")
    cols = {c.lower().strip(): c for c in df.columns}

    def pick(key):
        for a in _ALIASES[key]:
            if a in cols:
                return df[cols[a]]
        return None
    dip = pick("dip")
    if dip is None:
        raise ValueError("No se encontró columna de buzamiento (dip / buzamiento).")
    dd = pick("dd")
    if dd is None:
        st = pick("strike")
        if st is None:
            raise ValueError("Falta dirección de buzamiento (dd) o rumbo (strike).")
        dd = (st.astype(float) + 90) % 360          # regla de la mano derecha
    tipo = pick("tipo")
    out = pd.DataFrame({
        "id": pick("id") if pick("id") is not None else np.arange(1, len(df) + 1),
        "tipo": (tipo.astype(str).str.strip().str.lower() if tipo is not None else "diaclasa"),
        "dd": dd.astype(float) % 360, "dip": dip.astype(float).clip(0, 90)})
    out["tipo"] = out["tipo"].replace({"foliación": "foliacion", "diaclasas": "diaclasa",
                                       "fallas": "falla", "joint": "diaclasa", "fault": "falla"})
    out["strike"] = (out["dd"] - 90) % 360
    out["rumbo_axial"] = out["strike"] % 180
    return out
