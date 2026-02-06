from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.routes.predict import router as predict_router
from app.routes.ui import router as ui_router
from app.database import Base, engine
from app.config import APP_NAME

app = FastAPI(title=APP_NAME, version="3.0")

# ✅ Create DB tables automatically
Base.metadata.create_all(bind=engine)

# ✅ Static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ✅ Routes
app.include_router(ui_router)
app.include_router(predict_router)

@app.get("/health")
def health():
    return {"status": "running ✅"}