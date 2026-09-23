@echo off
REM Genera dist\LineamientosPRO\LineamientosPRO.exe con PyInstaller
cd /d "%~dp0"
python -m PyInstaller --noconfirm --clean --windowed --name LineamientosPRO ^
  --add-data "gui_pro\static;gui_pro\static" ^
  --collect-all rasterio --collect-all pyogrio --collect-all pyproj --collect-all shapely ^
  --collect-submodules skimage --collect-submodules sklearn ^
  --collect-all mplstereonet --hidden-import openpyxl --hidden-import PIL ^
  --exclude-module torch --exclude-module torchvision --exclude-module tensorflow ^
  --exclude-module numba --exclude-module llvmlite --exclude-module pysheds ^
  --exclude-module IPython --exclude-module notebook --exclude-module boto3 --exclude-module botocore ^
  --hidden-import lineamientos_pro.pipeline --hidden-import uvicorn.logging ^
  --hidden-import uvicorn.loops.auto --hidden-import uvicorn.protocols.http.auto ^
  --hidden-import uvicorn.lifespan.on ^
  gui_pro\app.py
echo.
echo Ejecutable en dist\LineamientosPRO\LineamientosPRO.exe
pause
