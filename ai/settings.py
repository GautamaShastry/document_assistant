import os

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