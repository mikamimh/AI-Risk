@echo off
setlocal
cd /d "%~dp0"

echo Instalando PyInstaller...
python -m pip install pyinstaller --quiet
if errorlevel 1 (
    echo ERRO: falha ao instalar PyInstaller.
    pause
    exit /b 1
)

echo Compilando AI Risk.exe...
python -m PyInstaller ^
    --onefile ^
    --console ^
    --name "AI Risk" ^
    --distpath "." ^
    --workpath "build\_pyinstaller" ^
    --specpath "build\_pyinstaller" ^
    --clean ^
    launcher.py

if errorlevel 1 (
    echo ERRO: compilacao falhou.
    pause
    exit /b 1
)

echo.
echo Pronto! "AI Risk.exe" criado nesta pasta.
echo Done!  "AI Risk.exe" created in this folder.
pause
endlocal
