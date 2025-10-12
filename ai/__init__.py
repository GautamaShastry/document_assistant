"""
AI Services Package for RAG Pipeline

This package contains all AI-related functionality:
- Document processing and loading
- Vector store management with FAISS
- RAG pipeline implementation
- Configuration and settings
"""

from ai.settings import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL_NAME,
    VECTORSTORE_DIR,
    UPLOAD_DIR
)

from ai.services.document import (
    DocumentProcessor,
    load_document,
    split_documents
)

from ai.services.vector_store import (
    VectorStoreManager,
    create_vector_store,
    load_vector_store,
    get_retriever
)

from ai.services.rag_pipeline import (
    RAGPipeline,
    answer,
    stream_answer
)

__version__ = "1.0.0"

__all__ = [
    # Settings
    "OLLAMA_MODEL",
    "OLLAMA_HOST",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBED_MODEL_NAME",
    "VECTORSTORE_DIR",
    "UPLOAD_DIR",
    
    # Classes
    "DocumentProcessor",
    "VectorStoreManager",
    "RAGPipeline",
    
    # Functions
    "load_document",
    "split_documents",
    "create_vector_store",
    "load_vector_store",
    "get_retriever",
    "answer",
    "stream_answer",
]