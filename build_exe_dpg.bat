@echo off
set "PYTHON_EXE=D:\Python3\python.exe"
set "APP_NAME=ShapeKeyFaceTracker"

echo [1/4] Installing PyInstaller...
"%PYTHON_EXE%" -m pip install pyinstaller

echo [2/4] Cleaning old builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [3/4] Building EXE with Assets...
REM --add-data "source;destination"
REM In Windows, the separator is ;
REM Adding --collect-all mediapipe to fix "No module named 'mediapipe.tasks.c'"
"%PYTHON_EXE%" -m PyInstaller --onefile --noconsole ^
    --name "%APP_NAME%" ^
    --add-data "face_landmarker.task;." ^
    --add-data "face_mesh.png;." ^
    --collect-all mediapipe ^
    main_dpg.py

echo.
echo [4/4] Done!
echo EXE is in: %cd%\dist\%APP_NAME%.exe
echo Logs will be created at: tracker_log.txt (next to the EXE)
pause
