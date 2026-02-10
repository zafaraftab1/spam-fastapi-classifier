# 🚀 Quick Reference Guide

## Start the App (30 seconds)

```bash
cd /Users/zafaraftab/SpamEmailClassifier
uvicorn src.main:app --reload --port 8000
```

Then open: **http://127.0.0.1:8000**

---

## Common Commands

### First Time Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn src.main:app --reload --port 8000
```

### Test the API
```bash
# Health check
curl http://127.0.0.1:8000/health

# Predict spam
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"Free money!!!"}'

# Predict legitimate
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"message":"Hi, how are you?"}'
```

### Train Model
```bash
python scripts/train_model.py
```

### View API Documentation
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## Key Files

| File | Purpose |
|------|---------|
| `src/main.py` | FastAPI app entry point |
| `src/routes/predict.py` | ML prediction endpoint |
| `src/routes/ui.py` | Web UI endpoint |
| `src/templates/index.html` | Web interface |
| `src/static/style.css` | Styling |
| `src/ML/model_utils.py` | Model loading |
| `src/database.py` | Database setup |
| `scripts/train_model.py` | Model training |

---

## Environment Setup (Optional)

Create `.env` file:
```env
APP_NAME=Spam Email Classifier
DB_URL=sqlite:///./dev.db
REDIS_URL=redis://localhost:6379/0
```

**Or use defaults (recommended for local dev)**

---

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Web UI |
| GET | `/health` | Health check |
| POST | `/api/predict` | Classify email |
| GET | `/docs` | API documentation |

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Module not found | `cd` to project root, run `uvicorn src.main:app --reload` |
| Port in use | Use different port: `--port 8001` |
| CSS not loading | Ensure you're in project root |
| Model not found | Run `python scripts/train_model.py` |
| Redis error | Just remove `REDIS_URL` from `.env` |

---

## Project Status

✅ All errors fixed
✅ Frontend enhanced with modern UI
✅ Documentation complete
✅ Ready for production

**Version:** 3.0
**Status:** 🟢 Production Ready

