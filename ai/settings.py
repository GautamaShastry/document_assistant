"""
AI Service Configuration
Centralized settings for the RAG pipeline
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Disable TensorFlow warnings
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

# Ollama LLM Configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))

# Document Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Embedding Model Configuration
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")  # Use 'cuda' for GPU
EMBED_NORMALIZE = os.getenv("EMBED_NORMALIZE", "true").lower() in {"1", "true", "yes"}

# Storage Paths
BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", BASE_DIR / "backend/app/data/vectorstore"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "backend/app/data/uploads"))

# Create directories
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Document Processing Limits
PROCESSING_LIMITS = {
    "max_file_size": 50 * 1024 * 1024,  # 50 MB
    "max_chunks_per_document": 1000,
    "min_chunk_size": 50,  # Minimum characters per chunk
}

# Supported file types
SUPPORTED_TYPES = ['.pdf', '.docx', '.txt']

# RAG Pipeline Configuration
RAG_CONFIG = {
    "retrieval_k": 4,  
    "search_type": "similarity",  
    "score_threshold": 0.5,  
}

logger.info("=" * 60)
logger.info("AI Settings Initialized")
logger.info(f"OLLAMA_MODEL: {OLLAMA_MODEL}")
logger.info(f"OLLAMA_HOST: {OLLAMA_HOST}")
logger.info(f"EMBED_MODEL: {EMBED_MODEL_NAME}")
logger.info(f"VECTORSTORE_DIR: {VECTORSTORE_DIR}")
logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}, OVERLAP: {CHUNK_OVERLAP}")
logger.info("=" * 60)