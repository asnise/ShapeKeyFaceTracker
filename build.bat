@echo off
echo Building ShapeKeyFaceTracker Executable...
echo This might take a few minutes...

pyinstaller --noconfirm --onefile --windowed --name "ShapeKeyFaceTracker" --collect-data mediapipe --collect-all customtkinter main.py

echo.
echo =========================================================
echo Build Complete!
echo You can find your ShapeKeyFaceTracker.exe in the "dist" folder.
echo =========================================================
pause
