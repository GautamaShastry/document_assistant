"""
AI Services Module

Contains the core AI functionality:
- Document loading and text splitting
- Vector store operations with FAISS
- RAG pipeline with conversational context
"""

from .document import load_document, split_documents