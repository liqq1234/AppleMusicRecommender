@echo off
echo ==========================================
echo   Apple Music Recommendation System
echo       Initialization Script
echo ==========================================
echo.
cd App
echo [1/3] Creating Database Migrations...
python manage.py makemigrations user
echo [2/3] Applying Migrations...
python manage.py migrate
cd ..
echo [3/3] Importing Real CSV Data (100k+ Music + 140k+ Interactions)...
python Scripts\import_real_data.py
echo.
echo Database initialization complete!
pause
