# Extractor de Lineamientos La Cruz

Sistema de extracción automática de lineamientos geológicos a partir de Modelos Digitales de Elevación (DEM). Este proyecto utiliza algoritmos avanzados de procesamiento de imágenes para identificar estructuras geológicas lineales de forma eficiente.

## 🚀 Características
- Procesamiento automático de archivos GeoTIFF (.tif).
- Interfaz gráfica intuitiva para facilitar el uso a geólogos y analistas.
- Basado en algoritmos optimizados de detección de bordes y filtrado espacial.
- Capacidad de exportar resultados para análisis posterior en SIG.

## 📂 Estructura del Proyecto
- **Codigo_Fuente/**: Contiene la lógica del algoritmo (`lineamientos_lacruz`) y la interfaz gráfica (`gui`).
- **ExtractorLineamentos/**: Versión ejecutable lista para usar en Windows.
- **Ejemplos/**: Archivos DEM de prueba para validar el funcionamiento.

## 💻 Uso para Usuarios
1. Navega a la carpeta `ExtractorLineamentos`.
2. Ejecuta `ExtractorLineamentos.exe`.
3. Carga un archivo `.tif` desde la carpeta `Ejemplos` y ajusta los parámetros según sea necesario.

## 🛠 Desarrollo
Si deseas modificar el código o ejecutarlo desde la fuente, asegúrate de tener Python 3.10+ instalado. Las dependencias principales incluyen:
- `numpy`
- `rasterio`
- `geopandas`
- `matplotlib`
- `opencv-python`

---
Desarrollado con el apoyo de **Antigravity AI**.
