import sys
import os
import io
from pathlib import Path
import threading
import time
import uvicorn
import webview
import json
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
import asyncio

# --- LÓGICA PARA EMPAQUETADO (PyInstaller) ---
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
    sys.path.append(str(BASE_DIR))
else:
    BASE_DIR = Path(__file__).parent.parent
    sys.path.append(str(BASE_DIR))

from lineamientos_lacruz import config as cfg
from lineamientos_lacruz.main import run_pipeline as start_extraction

app = FastAPI()
GUI_DIR = BASE_DIR / "gui"

app.mount("/static", StaticFiles(directory=str(GUI_DIR / "static")), name="static")

log_capture_queue = asyncio.Queue()

class OutputCapture(io.TextIOBase):
    def write(self, s):
        if s.strip():
            asyncio.run_coroutine_threadsafe(log_capture_queue.put(s.strip()), loop)
        return len(s)

@app.get("/", response_class=HTMLResponse)
async def index():
    with open(GUI_DIR / "static" / "index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/config")
async def get_config():
    return {
        "DEM_PATH": str(cfg.DEM_PATH),
        "MODE": cfg.MODE,
        "DETECTOR": cfg.DETECTOR,
        "GTHR_PERCENTILE": cfg.GTHR_PERCENTILE,
        "MIN_LENGTH_M": cfg.MIN_LENGTH_M,
        "OUTPUT_DIR": str(cfg.OUTPUT_DIR) if cfg.OUTPUT_DIR else "Resultados/",
        "FILTER_DRAINAGE": cfg.FILTER_DRAINAGE,
        "CATALYST_AZIMUTH": cfg.CATALYST_AZIMUTH,
        "LTHR": cfg.LTHR, "FTHR": cfg.FTHR, "ATHR": cfg.ATHR, "DTHR": cfg.DTHR,
        "DILATION_RADIUS": cfg.DILATION_RADIUS,
        "PCA_COMPONENT": cfg.PCA_COMPONENT,
        "Z_FACTOR": cfg.Z_FACTOR,
        "INCLUDE_SLOPE": cfg.INCLUDE_SLOPE,
        "INCLUDE_CURVATURE": cfg.INCLUDE_CURVATURE,
    }

@app.post("/api/run")
async def run_cmd(data: dict):
    try:
        overrides = {
            "DEM_PATH": data.get("dem_path"),
            "MODE": data.get("mode"),
            "DETECTOR": data.get("detector"),
            "GTHR_PERCENTILE": data.get("gthr_percentile"),
            "MIN_LENGTH_M": data.get("min_length_m"),
            "OUTPUT_DIR": data.get("output_dir"),
            "CATALYST_AZIMUTH": data.get("azimuth"),
            "FILTER_DRAINAGE": data.get("filter_drainage") == "true",
            "LTHR": data.get("lthr"), "FTHR": data.get("fthr"),
            "ATHR": data.get("athr"), "DTHR": data.get("dthr"),
            "DILATION_RADIUS": data.get("dilation_radius"),
            "PCA_COMPONENT": int(data.get("pca_component", 2)),
            "Z_FACTOR": float(data.get("z_factor", 1.0)),
            "INCLUDE_SLOPE": data.get("include_slope") == True,
            "INCLUDE_CURVATURE": data.get("include_curvature") == True,
        }
        for k, v in overrides.items():
            if k == "DEM_PATH" and v: v = Path(v)
            setattr(cfg, k, v)

        def worker():
            old_stdout = sys.stdout
            sys.stdout = OutputCapture()
            try:
                start_extraction()
                # Tras terminar, intentar generar preview si existe matplotlib
                try: generate_preview_internal()
                except: pass
            except Exception as e: print(f"ERROR: {e}")
            finally:
                sys.stdout = old_stdout
                asyncio.run_coroutine_threadsafe(log_capture_queue.put("[EOF]"), loop)

        threading.Thread(target=worker, daemon=True).start()
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/logs")
async def get_logs():
    async def event_generator():
        while True:
            msg = await log_capture_queue.get()
            yield f"data: {msg}\n\n"
            if msg == "[EOF]": break
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/preview")
async def get_preview():
    """Retorna la imagen de previsualización más reciente"""
    out_dir = Path(cfg.OUTPUT_DIR) if cfg.OUTPUT_DIR else Path(cfg.DEM_PATH).parent / "Resultados"
    preview_file = out_dir / "preview_last_run.png"
    if preview_file.exists():
        return FileResponse(preview_file)
    return {"error": "No preview available"}

def generate_preview_internal():
    """Dibuja los lineamientos sobre el hillshade y guarda en Resultados/preview_last_run.png"""
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import rasterio
    from rasterio.plot import show
    
    out_dir = Path(cfg.OUTPUT_DIR) if cfg.OUTPUT_DIR else Path(cfg.DEM_PATH).parent / "Resultados"
    shp_files = list(out_dir.glob("*.shp"))
    if not shp_files: return
    
    # Tomar el shp más reciente
    shp_path = max(shp_files, key=os.path.getmtime)
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    
    try:
        # Cargar hillshade original si es posible, sino solo el SHP
        gdf = gpd.read_file(shp_path)
        gdf.plot(ax=ax, color='#2dd4bf', linewidth=0.5, alpha=0.8)
        ax.set_title(f"Resultados: {shp_path.name}", color='white')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / "preview_last_run.png", facecolor=fig.get_facecolor())
    except:
        pass
    finally:
        plt.close()

def start_fastapi():
    global loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    config = uvicorn.Config(app, host="127.0.0.1", port=23456, log_level="error")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())

if __name__ == "__main__":
    t = threading.Thread(target=start_fastapi, daemon=True)
    t.start()
    time.sleep(1)
    webview.create_window(
        'Lineamientos La Cruz - Desktop', 'http://127.0.0.1:23456',
        width=1200, height=850, background_color='#0f172a'
    )
    webview.start()
