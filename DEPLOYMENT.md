# 🚀 Deployment Guide

## Local Development Setup

### Prerequisites
- Python 3.8+
- pip or conda
- Virtual environment (recommended)

### Step 1: Clone Repository
```bash
cd /Users/zafaraftab/SpamEmailClassifier
```

### Step 2: Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Server
```bash
uvicorn src.main:app --reload --port 8000
```

### Step 5: Access Application
- **Web UI:** http://127.0.0.1:8000
- **API Docs:** http://127.0.0.1:8000/docs

---

## Docker Deployment

### Build Docker Image

Create `Dockerfile` in project root:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build and Run

```bash
# Build image
docker build -t spam-classifier:latest .

# Run container
docker run -d \
  --name spam-classifier \
  -p 8000:8000 \
  -v ./dev.db:/app/dev.db \
  spam-classifier:latest

# Check logs
docker logs spam-classifier

# Stop container
docker stop spam-classifier
```

### Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_NAME=Spam Email Classifier
      - DB_URL=postgresql://user:password@postgres:5432/spam_classifier
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - postgres
      - redis
    volumes:
      - ./artifacts:/app/artifacts
    restart: unless-stopped

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: spam_classifier
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

volumes:
  postgres_data:
```

### Run with Docker Compose
```bash
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

---

## Heroku Deployment

### Prerequisites
- Heroku CLI installed
- Heroku account

### Step 1: Login to Heroku
```bash
heroku login
```

### Step 2: Create Heroku App
```bash
heroku create spam-email-classifier
```

### Step 3: Add Buildpack
```bash
heroku buildpacks:add heroku/python
```

### Step 4: Set Environment Variables
```bash
heroku config:set APP_NAME="Spam Classifier"
heroku config:set REDIS_URL=<your-redis-url>  # Optional
```

### Step 5: Deploy
```bash
git push heroku main
```

### Step 6: View Logs
```bash
heroku logs --tail
```

### Step 7: Open App
```bash
heroku open
```

---

## AWS Deployment

### Option 1: Elastic Beanstalk

#### Step 1: Install EB CLI
```bash
pip install awsebcli --upgrade --user
```

#### Step 2: Initialize EB Application
```bash
eb init -p python-3.11 spam-classifier
```

#### Step 3: Create `.ebextensions/app.config`
```yaml
option_settings:
  aws:autoscaling:launchconfiguration:
    IamInstanceProfile: aws-elasticbeanstalk-ec2-role
  aws:elasticbeanstalk:application:environment:
    PYTHONPATH: /var/app/current:$PYTHONPATH
    DJANGO_SETTINGS_MODULE: config.settings
commands:
  01_migrate:
    command: "source /var/app/venv/*/bin/activate && python scripts/train_model.py || true"
    leader_only: true
```

#### Step 4: Create Environment
```bash
eb create production --single
```

#### Step 5: Set Environment Variables
```bash
eb setenv APP_NAME="Spam Classifier"
```

#### Step 6: Deploy
```bash
eb deploy
```

#### Step 7: View Logs
```bash
eb logs
```

### Option 2: ECS (Elastic Container Service)

```bash
# Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag spam-classifier:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/spam-classifier:latest

docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/spam-classifier:latest

# Create ECS task definition, service, and cluster via AWS Console
```

---

## Google Cloud Deployment

### Cloud Run Deployment

```bash
# Build image
gcloud builds submit --tag gcr.io/PROJECT_ID/spam-classifier

# Deploy to Cloud Run
gcloud run deploy spam-classifier \
  --image gcr.io/PROJECT_ID/spam-classifier \
  --platform managed \
  --region us-central1 \
  --set-env-vars APP_NAME="Spam Classifier"

# View URL
gcloud run services describe spam-classifier --region us-central1
```

---

## Azure Deployment

### App Service Deployment

```bash
# Create resource group
az group create --name myResourceGroup --location eastus

# Create App Service plan
az appservice plan create \
  --name myAppServicePlan \
  --resource-group myResourceGroup \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group myResourceGroup \
  --plan myAppServicePlan \
  --name spam-classifier \
  --runtime "PYTHON:3.11"

# Configure deployment
az webapp up \
  --resource-group myResourceGroup \
  --name spam-classifier \
  --runtime "PYTHON:3.11" \
  --logs

# Set environment variables
az webapp config appsettings set \
  --resource-group myResourceGroup \
  --name spam-classifier \
  --settings APP_NAME="Spam Classifier"
```

---

## DigitalOcean Deployment

### App Platform

```bash
# Login to doctl
doctl auth init

# Create app
doctl apps create --spec app.yaml

# View app
doctl apps list
```

Create `app.yaml`:
```yaml
name: spam-classifier
services:
- name: web
  github:
    repo: zafaraftab/SpamEmailClassifier
    branch: main
  build_command: pip install -r requirements.txt
  run_command: uvicorn src.main:app --host 0.0.0.0 --port 8080
  http_port: 8080
  envs:
  - key: APP_NAME
    value: "Spam Classifier"
  - key: PYTHONUNBUFFERED
    value: "1"
```

---

## Production Best Practices

### 1. Use Environment Variables
```bash
export APP_NAME="Spam Classifier Production"
export DB_URL="postgresql://user:pass@prod-db:5432/spam"
export REDIS_URL="redis://prod-redis:6379/0"
```

### 2. Enable HTTPS
```python
# Use SSL certificates from Let's Encrypt
# Configure in Nginx/CloudFlare
```

### 3. Database Setup
```sql
-- PostgreSQL
CREATE DATABASE spam_classifier;
CREATE USER spam_user WITH PASSWORD 'secure_password';
ALTER ROLE spam_user SET client_encoding TO 'utf8';
GRANT ALL PRIVILEGES ON DATABASE spam_classifier TO spam_user;
```

### 4. Redis Configuration
```bash
# Set requirepass in redis.conf
requirepass your_secure_password

# Update connection string
REDIS_URL=redis://:your_secure_password@redis:6379/0
```

### 5. Monitoring & Logging
```python
# Add in production:
import logging
logging.basicConfig(level=logging.INFO)

# Use services like:
# - DataDog
# - New Relic
# - Sentry
# - CloudWatch
```

### 6. Load Balancing
```nginx
upstream spam_classifier {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    server_name spam-classifier.com;

    location / {
        proxy_pass http://spam_classifier;
        proxy_set_header Host $host;
    }
}
```

### 7. Security Headers
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 8. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/predict")
@limiter.limit("100/minute")
def predict_api(request: MessageRequest):
    # ...
```

---

## Monitoring Checklist

- [ ] Set up error tracking (Sentry)
- [ ] Configure logging (CloudWatch, DataDog)
- [ ] Monitor database performance
- [ ] Monitor cache hit rates
- [ ] Set up alerts for downtime
- [ ] Regular backup of database
- [ ] Monitor prediction accuracy drift
- [ ] Track API response times

---

## Performance Optimization

### 1. Model Caching
```python
# Load model once at startup
model = None
vectorizer = None

@app.on_event("startup")
async def load_model():
    global model, vectorizer
    model, vectorizer = load_artifacts()
```

### 2. Database Connection Pooling
```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40
)
```

### 3. Redis Connection Pooling
```python
import redis
from redis import ConnectionPool

pool = ConnectionPool.from_url(REDIS_URL)
redis_client = redis.Redis(connection_pool=pool)
```

### 4. Async/Await for I/O Operations
```python
@app.post("/api/predict")
async def predict_api(request: MessageRequest, 
                     db: Session = Depends(get_db)):
    # Use async operations where possible
    pass
```

---

## Rollback Strategy

```bash
# Git rollback
git revert <commit-hash>
git push

# Docker rollback
docker stop spam-classifier
docker run -d spam-classifier:previous-version

# Heroku rollback
heroku releases
heroku rollback v42

# ECS rollback
aws ecs update-service --cluster production \
  --service spam-classifier \
  --task-definition spam-classifier:previous-version
```

---

## Health Checks

### Local Check
```bash
curl http://127.0.0.1:8000/health
```

### Remote Check
```bash
curl https://your-domain.com/health
```

### Automated Checks
```bash
# Add to monitoring tool
GET /health every 30 seconds
Response code: 200
Body contains: "running"
```

---

**Deployment Version:** 3.0  
**Last Updated:** February 10, 2026  
**Status:** ✅ Ready for Production

