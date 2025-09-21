"""
AI Services Package for Intelligent Document Assistant

This package contains all AI-related functionality:
- Document processing and loading
- Vector store management with FAISS
- RAG pipeline with conversational context
- Configuration and settings
"""

from .settings import (
    OLLAMA_MODEL,
    OLLAMA_HOST,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBED_MODEL_NAME,
    EMBED_DEVICE,
    EMBED_NORMALIZE,
    VECTORSTORE_DIR,
    UPLOAD_DIR
)

from .services.document import (
    load_document,
    split_documents,
    SUPPORTED_TYPES
)

from .services.vector_store import (
    create_vector_store,
    upsert_documents_to_vector_store,
    load_vector_store,
    get_retriever,
    delete_vector_store
)

from .services.rag_pipeline import (
    answer,
    stream_answer,
    conversational_answer,
    stream_conversational_answer,
    format_chat_messages_for_context,
    get_conversation_summary
)

__all__ = [
    # Settings
    "OLLAMA_MODEL",
    "OLLAMA_HOST", 
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "EMBED_MODEL_NAME",
    "EMBED_DEVICE",
    "EMBED_NORMALIZE",
    "VECTORSTORE_DIR",
    "UPLOAD_DIR",
    
    # Document processing
    "load_document",
    "split_documents",
    "SUPPORTED_TYPES",
    
    # Vector store
    "create_vector_store",
    "upsert_documents_to_vector_store", 
    "load_vector_store",
    "get_retriever",
    "delete_vector_store",
    
    # RAG pipeline
    "answer",
    "stream_answer",
    "conversational_answer",
    "stream_conversational_answer",
    "format_chat_messages_for_context",
    "get_conversation_summary"
]