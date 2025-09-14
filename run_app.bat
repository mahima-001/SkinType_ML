@echo off
echo Starting SkiScan Application...
echo.

:: Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

:: Install requirements
echo Installing requirements...
pip install -r requirements.txt
echo.

:: Start the Flask application
echo Starting Flask server...
echo Your application will be available at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
python app.py

pause
