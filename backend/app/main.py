import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from backend.app.api import router as api_router
from backend.app.db.session import init_db

# configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# disable tensorflow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

load_dotenv()  # take environment variables from .env

# ensure data directories exist
os.makedirs("backend/app/data/uploads", exist_ok=True)
os.makedirs("backend/app/data/vectorstore", exist_ok=True)

# initialize database
init_db()

app = FastAPI(
    title="Intelligent Document Assistant",
    description="An AI-powered document assistant that helps you manage and query your documents efficiently.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {
        "message": "Welcome to the Intelligent Document Assistant API. Visit /docs for API documentation.",
        "docs": "/docs",
        "version": "1.0.0",
        "features": [
            "User Authentication (Registration, Login, Password Reset)",
            "Document Upload and Management (PDF, DOCX, TXT)",
            "AI-Powered Chat Interface for Document Querying",
            "Contextual Responses with Source Citations",
            "Secure and Scalable Architecture",
        ]
    }
    
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the Intelligent Document Assistant API...")
    logger.info(f"Database Initialized and ready.")
    logger.info(f"Ollama model: {os.getenv('OLLAMA_MODEL', 'qwen3:8b')} at {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down the Intelligent Document Assistant API...")