# AI-Based Power-Line Inspection with a Drone
## Run (Windows PowerShell)
    python -m venv venv; venv\Scripts\activate; pip install -r requirements.txt; uvicorn main:app --reload
## Run (Linux / macOS)
    python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && uvicorn main:app --reload
Open http://127.0.0.1:8000 (API docs: /docs). Data persists in app.db and storage/.
