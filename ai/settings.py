import os
import logging

# configure logging for AI service
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# disable tensorflow
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

CHUNK_SIZE = os.getenv("CHUNK_SIZE", 900)
CHUNK_OVERLAP = os.getenv("CHUNK_OVERLAP", 100)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu")
EMBED_NORMALIZE = os.getenv("EMBED_NORMALIZE", "false").lower() in {"1", "true", "yes"}

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "backend/app/data/vectorstore")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "backend/app/data/uploads")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# RAG Pipeline configurations
RAG_CONFIG = {
    "temperature": 0.2,
    "max_tokens": 2048,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
}

# Document processing limits
PROCESSING_LIMITS = {
    "max_file_size": 50 * 1024 * 1024,  # 50 MB
    "max_chunks_per_document": 1000,
    "max_chunks_per_user": 100,
}

SUPPORTED_DOCS = ['.pdf', '.docx', '.txt']

logger.info(f"AI Settings initialized:")
logger.info(f" - OLLAMA_MODEL: {OLLAMA_MODEL}")
logger.info(f" - OLLAMA_HOST: {OLLAMA_HOST}")
logger.info(f" - VECTORSTORE_DIR: {VECTORSTORE_DIR}")
logger.info(f" - UPLOAD_DIR: {UPLOAD_DIR}")
logger.info(f" - CHUNK_SIZE: {CHUNK_SIZE}")
logger.info(f" - CHUNK_OVERLAP: {CHUNK_OVERLAP}")
logger.info(f" - EMBED_MODEL_NAME: {EMBED_MODEL_NAME}")