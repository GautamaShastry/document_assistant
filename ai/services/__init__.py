"""
AI Services Module

Contains the core AI functionality:
- Document loading and text splitting
- Vector store operations with FAISS
- RAG pipeline implementation
"""

from ai.services.document import DocumentProcessor, load_document, split_documents
from ai.services.vector_store import VectorStoreManager, create_vector_store
from ai.services.rag_pipeline import RAGPipeline, answer, stream_answer

__all__ = [
    "DocumentProcessor",
    "VectorStoreManager",
    "RAGPipeline",
    "load_document",
    "split_documents",
    "create_vector_store",
    "answer",
    "stream_answer",
]