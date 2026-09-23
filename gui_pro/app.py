"""
Interfaz de escritorio de Lineamientos PRO.

  python gui_pro/app.py            (ventana nativa con pywebview)
  python gui_pro/app.py --browser  (abre en el navegador)

Backend FastAPI local (127.0.0.1) + frontend HTML/JS sin dependencias externas.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

if getattr(sys, "frozen", False):
    BASE = Path(sys._MEIPASS)
else:
    BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import uvicorn  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from lineamientos_pro import __version__  # noqa: E402
from lineamientos_pro.config import MODELOS, PRESETS, Params, params_from_preset  # noqa: E402

STATIC = BASE / "gui_pro" / "static"
PORT = int(os.environ.get("LINPRO_PORT", "23457"))

app = FastAPI(title="Lineamientos PRO")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.running = False
        self.cancel = False
        self.progress = 0.0
        self.stage = ""
        self.logs: list[str] = []
        self.result = None
        self.error = None
        self.started = None

    def log(self, msg):
        with self.lock:
            for line in str(msg).splitlines():
                self.logs.append(line)


JOB = Job()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/defaults")
def defaults():
    return {"params": Params().to_dict(), "models": MODELOS, "version": __version__,
            "presets": {k: {"label": v["label"], "params": params_from_preset(k).to_dict()}
                        for k, v in PRESETS.items()}}


@app.post("/api/run")
def run(params: dict):
    if JOB.running:
        raise HTTPException(409, "Ya hay un proceso en ejecución.")
    P = Params.from_dict(params)
    if not P.dem_path or not Path(P.dem_path).exists():
        raise HTTPException(400, f"No se encuentra el DEM: {P.dem_path or '(vacío)'}")
    for k in ("hillshade_path", "reference_path", "field_path"):
        v = getattr(P, k)
        if v and not Path(v).exists():
            raise HTTPException(400, f"No se encuentra el archivo: {v}")
    JOB.reset()
    JOB.running = True
    JOB.started = time.time()

    def prog(f, m=""):
        JOB.progress = float(f)
        if m:
            JOB.stage = m.strip()

    def worker():
        from lineamientos_pro.pipeline import Cancelled, run as run_pipeline
        try:
            JOB.result = run_pipeline(P, log=JOB.log, progress=prog, cancel=lambda: JOB.cancel)
        except Cancelled:
            JOB.error = "Proceso cancelado por el usuario."
            JOB.log(JOB.error)
        except Exception as e:  # noqa: BLE001
            JOB.error = str(e)
            JOB.log("ERROR: " + str(e))
            JOB.log(traceback.format_exc())
        finally:
            JOB.running = False

    threading.Thread(target=worker, daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def status(since: int = 0):
    with JOB.lock:
        logs = JOB.logs[since:]
        n = len(JOB.logs)
    return {"running": JOB.running, "progress": JOB.progress, "stage": JOB.stage,
            "logs": logs, "next": n, "result": JOB.result, "error": JOB.error,
            "elapsed": (time.time() - JOB.started) if JOB.started else 0}


@app.post("/api/cancel")
def cancel():
    JOB.cancel = True
    return {"ok": True}


def _out():
    if not JOB.result:
        raise HTTPException(404, "Sin resultados")
    return Path(JOB.result["output_dir"])


@app.get("/api/result/{sub}/{name}")
def result_file(sub: str, name: str):
    if sub not in ("preview", "figuras", "tablas"):
        raise HTTPException(404)
    p = (_out() / sub / name).resolve()
    if not str(p).startswith(str(_out().resolve())) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/open")
def open_path(data: dict):
    what = data.get("what", "folder")
    out = _out()
    target = {"report": out / "reporte.html",
              "excel": out / "tablas" / "estadisticas.xlsx"}.get(what, out)
    if what == "figure":
        target = (out / "figuras" / Path(data.get("name", "")).name)
    if not target.exists():
        raise HTTPException(404, "No existe")
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])
    return {"ok": True}


@app.post("/api/load_result")
def load_result(data: dict):
    """Abre una corrida anterior (carpeta Lineamientos_...)."""
    import json
    folder = Path(data.get("folder", ""))
    f = folder / "preview" / "resumen.json"
    if not f.exists():
        raise HTTPException(400, "La carpeta no contiene una corrida de Lineamientos PRO.")
    if JOB.running:
        raise HTTPException(409, "Hay un proceso en ejecución.")
    JOB.reset()
    JOB.result = json.loads(f.read_text(encoding="utf-8"))
    JOB.result["output_dir"] = str(folder)
    JOB.progress = 1.0
    return {"ok": True, "result": JOB.result}


@app.post("/api/config/save")
def config_save(data: dict):
    path = data.get("path")
    Params.from_dict(data.get("params", {})).save(path)
    return {"ok": True}


@app.post("/api/config/load")
def config_load(data: dict):
    return {"params": Params.load(data["path"]).to_dict()}


# ----------------------------------------------------------------------------
def _dialog(name):
    import webview
    fd = getattr(webview, "FileDialog", None)       # pywebview >= 5
    return getattr(fd, name) if fd else getattr(webview, f"{name}_DIALOG")


class NativeAPI:
    """Diálogos nativos expuestos a JavaScript por pywebview."""

    def __init__(self):
        self._window = None

    def pick_file(self, kind="raster", save=False):
        types = {
            "raster": ("Rásteres (*.tif;*.tiff;*.img;*.vrt)", "Todos (*.*)"),
            "vector": ("Vectores (*.shp;*.gpkg;*.geojson)", "Todos (*.*)"),
            "json": ("Configuración (*.json)", "Todos (*.*)"),
            "table": ("Tablas (*.csv;*.xlsx;*.xls;*.txt)", "Todos (*.*)"),
        }.get(kind, ("Todos (*.*)",))
        mode = _dialog("SAVE") if save else _dialog("OPEN")
        kw = {"save_filename": "parametros.json"} if save else {}
        r = self._window.create_file_dialog(mode, file_types=types, **kw)
        if not r:
            return ""
        return r if isinstance(r, str) else r[0]

    def pick_folder(self):
        r = self._window.create_file_dialog(_dialog("FOLDER"))
        return (r if isinstance(r, str) else r[0]) if r else ""


def _serve():
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")


def main():
    if "--server" in sys.argv:          # solo backend (pruebas / uso remoto)
        _serve()
        return
    threading.Thread(target=_serve, daemon=True).start()
    url = f"http://127.0.0.1:{PORT}"
    for _ in range(50):
        try:
            import urllib.request
            urllib.request.urlopen(url + "/api/defaults", timeout=0.5)
            break
        except Exception:
            time.sleep(0.1)
    if "--browser" in sys.argv:
        webbrowser.open(url)
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return
    try:
        import webview
    except ImportError:
        webbrowser.open(url)
        while True:
            time.sleep(1)
    api = NativeAPI()
    api._window = webview.create_window(f"Lineamientos PRO {__version__}", url, js_api=api,
                                       width=1440, height=920, min_size=(1100, 700),
                                       background_color="#0f1720")
    webview.start()


if __name__ == "__main__":
    main()
