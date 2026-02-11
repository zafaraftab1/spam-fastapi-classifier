from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .routes.predict import router as predict_router
from .routes.ui import router as ui_router
from .database import Base, engine
from .config import APP_NAME

app = FastAPI(title=APP_NAME, version="3.0")

# ✅ Create DB tables automatically (engine is safe to create a fallback SQLite in database.py)
Base.metadata.create_all(bind=engine)

# ✅ Static files (use package-relative path)
static_dir = str(Path(__file__).resolve().parent / "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# ✅ Routes
app.include_router(ui_router)
app.include_router(predict_router)

@app.get("/health")
def health():
    return {"status": "running ✅"}