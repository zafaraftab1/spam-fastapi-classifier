from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DB_URL

# Provide a safe fallback for local development
if DB_URL:
    engine = create_engine(DB_URL)
else:
    # Fallback to a local file-based SQLite database
    engine = create_engine("sqlite:///./dev.db", connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()