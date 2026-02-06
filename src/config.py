import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "Spam Classifier")
DB_URL = os.getenv("DB_URL")
REDIS_URL = os.getenv("REDIS_URL")