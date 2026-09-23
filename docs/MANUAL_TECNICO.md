# Lineamientos PRO

Extracción automática de lineamientos estructurales desde un DEM, con varios
modelos en un solo programa y la réplica del módulo LINE de PCI Geomatica para
comparar.

## Uso rápido

**Interfaz:** doble clic en `Lineamientos_PRO.bat` (o `python gui_pro/app.py`).
1. Elija el DEM (GeoTIFF en un sistema proyectado en metros).
2. Opcional: el sombreado que usó en PCI (para que la réplica sea idéntica) y un
   shapefile de referencia (validación).
3. Elija un preajuste y los modelos. Pulse **Ejecutar**.
4. Explore el mapa (zoom con la rueda, clic en una línea para ver atributos),
   el resumen, las figuras y el reporte HTML.

**Línea de comandos:**
```
python -m lineamientos_pro --dem dem.tif
python -m lineamientos_pro --dem dem.tif --preset completo --res 2 --ref catalyst.shp --hillshade 0.tif
python -m lineamientos_pro --dem dem.tif --campo datos_estructurales.csv --zona "La Cruz"
python -m lineamientos_pro --config parametros.json      # repite una corrida exacta
python -m lineamientos_pro.figures <carpeta_corrida> --campo datos.csv   # regenera las láminas
python -m lineamientos_pro --benchmark 6                 # banco de pruebas sintético
```

## Modelos

| Modelo | Qué detecta |
|---|---|
| **consenso** (recomendado) | Fusión ponderada de las cuatro evidencias; donde coinciden en posición y rumbo se refuerzan |
| valles | Concavidades lineales (Hessiano multiescala): valles de falla, drenaje controlado |
| crestas | Convexidades lineales: crestas y diques |
| escarpes | Crestas del mapa de pendiente: escarpes, facetas, quiebres de pendiente |
| bordes | Bordes de sombreado en 8 azimuts (lo que ve el intérprete, sin sesgo de iluminación) |
| pci_line | Réplica de PCI LINE (RADI, GTHR, LTHR, FTHR, ATHR, DTHR) sobre un sombreado |
| pci_multi | PCI LINE en 4 azimuts con los duplicados fusionados |

## Por qué funciona mejor que PCI LINE

PCI LINE detecta **bordes de brillo** en **un** sombreado: depende del azimut
del sol, marca cualquier borde (textura, carreteras, límites del DEM) y
entrega fragmentos cortos. Lineamientos PRO:

1. **Evidencia topográfica nativa**: mide la forma del relieve (curvaturas y
   pendiente a varias escalas), no el brillo de una imagen.
2. **Filtro orientado de rectitud multi-longitud**: para 18 rumbos integra la
   evidencia que concuerda en dirección a lo largo de 40 m y 200 m. Un
   lineamiento es recto y continuo; los cauces sinuosos y la textura no
   acumulan respuesta. Salta huecos (erosión, vegetación).
3. **Enlace colineal** con índice espacial, huecos proporcionales a la longitud
   y control del desfase lateral.
4. **Atributos geológicos** y **confianza calibrada** por cada línea.

### Resultados medidos

Banco sintético (6 terrenos de 2 km con fallas conocidas de rumbo aleatorio y
cauces sinuosos como distractores; tolerancia 8 m / 15°):

| Modelo | Precisión | Recall | F1 |
|---|---|---|---|
| **consenso** | **0,72** | **0,54** | **0,62** |
| bordes | 0,65 | 0,58 | 0,61 |
| PCI LINE con el mejor GTHR posible* | 0,20 | 0,26 | 0,22 |
| PCI LINE con valores por defecto | 0,13 | 0,31 | 0,18 |

\* Umbral elegido conociendo la respuesta correcta, una ventaja que en la práctica no existe.

Confianza del consenso: la clase **Alta** acertó el 86 % de su longitud, la
**Media** el 40 % y la **Baja** el 0 %.

### Fidelidad de la réplica PCI

Calibrada contra una corrida real de PCI Geomatica 2018 (LINE con parámetros
por defecto sobre `Ejemplos/0.tif`, salida `Ejemplos/pci_line_original.shp`, 2.264
polilíneas), siguiendo la ayuda oficial (`Ejemplos/geomatica.pdf`):
partición de polilíneas en ATHR, descarte de piezas menores que LTHR, enlace
solo de tramos que se enfrentan, entrada de 8 bits sin reescalar y vértices en
esquinas de píxel. Las incógnitas se ajustaron con
`python -m lineamientos_pro.calibrate_pci Ejemplos/0.tif Ejemplos/pci_line_original.shp`:
sigma = RADI/2,5 y ganancia del gradiente 22.

| Réplica vs PCI real | Valor |
|---|---|
| Polilíneas | 2.015 (PCI: 2.264) |
| Mediana de longitud | 45 m (PCI: 47,7 m) |
| Precisión / recall / F1 (15 m, 20°) | 0,91 / 0,79 / **0,85** |
| F1 con tolerancia estricta (1,5 m) | 0,68 |
| Correlación de rosetas | 0,98 |

Si cambia de versión de PCI u otra configuración, repita la calibración con
una corrida propia.

### Sobre La Cruz

La salida de PCI está dominada por rumbos E-W por el sesgo de iluminación del
sombreado. El consenso, que no depende de la iluminación, da como familia
principal **N48E (42 % de la longitud)**, que coincide con los grandes valles
NE.

## Salidas (carpeta `Lineamientos_<dem>_<fecha>`)

| Archivo | Contenido |
|---|---|
| `lineamientos.gpkg` | Capa `lin_<modelo>` por modelo y `cruces_<principal>` (intersecciones) |
| `shp/` | Las mismas capas en shapefile |
| `rasters/` | Sombreado multidireccional, respuesta de línea por modelo, densidad de longitud (km/km²), de frecuencia (lin./km²) y de cruces (núcleo cuártico), red de drenaje |
| `figuras/` | Láminas de publicación a 300 dpi (ver abajo) y un mapa por modelo |
| `tablas/estadisticas.xlsx` | Libro con resumen, estadística estructural, familias por modelo, atributos de cada lineamiento, validación, acuerdo entre modelos y campo vs DEM |
| `tablas/*.csv` | Las mismas tablas en CSV (`;`) |
| `reporte.html` | Reporte autocontenido con todo lo anterior |
| `parametros.json` | Configuración exacta; se puede recargar para repetir la corrida |

### Láminas (se generan solas en cada corrida)

| Lámina | Paneles |
|---|---|
| 01 Mapa de lineamientos | Mapa por familias sobre sombreado, norte, escala, roseta insertada |
| 02 Análisis direccional | Roseta por longitud, roseta de frecuencia, histograma axial con densidad von Mises, tabla de familias (azimut medio, desviación circular, n, km, %, tipo, confianza) y prueba de Rayleigh |
| 03 Longitudes | Histograma, ley de potencia N(>L) ∝ L^-a con R², rumbo vs longitud, cajas por familia |
| 04 Densidad | Densidad de longitud y de frecuencia con cruces |
| 05 Confianza y tipo | Mapa por confianza, longitud por clase, evidencia dominante y % sobre drenaje por familia |
| 06 Comparación de modelos | Rosetas de todos los modelos (y de la referencia), densidades, matriz de acuerdo F1 |
| 07 Campo vs DEM | Estereograma (polos de diaclasas + densidad Kamb, falla, foliaciones), roseta de diaclasas subverticales, superposición con los lineamientos, tabla de correspondencia de familias, histograma de buzamientos |

**Datos de campo** (opcional, CSV o XLSX): columnas `tipo` (diaclasa / foliacion /
falla), `dir_buz` (dirección de buzamiento) o `rumbo` (regla de la mano
derecha), y `buzamiento`. Ejemplo: `Ejemplos/datos_estructurales_lacruz.csv`.
Para comparar con los lineamientos se usan las diaclasas subverticales
(buzamiento ≥ 60°), porque son las que producen trazas en el relieve.

**Atributos de cada lineamiento:** `longitud_m`, `azimut` (0–180°), `rumbo`
(N48E), `sector`, `familia` y `fam_rumbo`, `tipo` (valle, cresta, escarpe o
borde), `n_evid` (evidencias que lo respaldan), `fuerza`, `soporte`,
`drenaje` (fracción sobre cauces), `sigue_dren`, `confianza` y `clase_conf`.

`sigue_dren = True` no invalida el lineamiento: los tramos de drenaje rectos
suelen estar controlados por fracturas. El atributo permite separarlos al
interpretar.

## Recomendaciones de parámetros

- **DEM LiDAR de 1 m**: `--res 2` da el mismo resultado unas 4 veces más rápido
  (La Cruz: unos 35 s con consenso y PCI). Para rasgos muy finos use el
  preajuste *detalle*.
- **Estudios regionales (SRTM/ALOS 12–30 m)**: preajuste *regional*.
- **Sensibilidad**: 0,5 es conservador y 0,8 exploratorio. Filtre después por
  `clase_conf`.
- Los parámetros de la réplica PCI están en píxeles de la imagen, igual que en
  Geomatica. La réplica siempre corre a la resolución nativa.

## Estructura del código

```
lineamientos_pro/
  config.py      parámetros y preajustes
  raster.py      lectura/escritura, relleno de huecos
  evidence.py    Hessiano multiescala, escarpes, bordes multiazimut, sombreado
  linefilter.py  filtro orientado de rectitud, NMS, histéresis
  vectorize.py   trazado del esqueleto, enlace colineal, recorte por soporte
  models.py      modelos (consenso, evidencias, réplica PCI)
  pci_line.py    réplica del algoritmo LINE de PCI
  calibrate_pci.py  calibra la réplica contra una corrida real de PCI
  attributes.py  familias, tipo, confianza, drenaje
  products.py    densidades (núcleo cuártico), cruces, reporte HTML
  stats.py       estadística circular, ley de potencia, lectura de datos de campo
  figures.py     láminas de publicación
  validation.py  precisión/recall/F1 por longitud contra una referencia
  synthetic.py   generador de terrenos con fallas conocidas
  bench.py       benchmark
  pipeline.py    orquestador
gui_pro/         interfaz (FastAPI + pywebview, sin internet)
```

`build_exe.bat` genera un ejecutable con PyInstaller.
