# Apple Music Recommendation System - Final Deliverable

This folder contains the complete, ready-to-run recommendation system.

## 🚀 Quick Start

### 1. Environment Setup
Make sure you have Python 3.10+ installed. Install the dependencies first:
```bash
pip install -r requirements.txt
```

### 2. Initialize Database (First time only)
Double-click `Init_Database.bat` or run:
```bash
python App/manage.py migrate
python Scripts/populate_user_rate.py
```

### 3. Run the System
Double-click `Run_Server.bat` or run:
```bash
cd App
python manage.py runserver
```
Visit: **http://127.0.0.1:8000**

## 📂 Folder Structure
- `App/`: The Django backend and frontend.
- `Scripts/`: Data simulation tools.
- `requirements.txt`: Project dependencies.
- `Init_Database.bat`: One-click data initialization.
- `Run_Server.bat`: One-click system startup.
