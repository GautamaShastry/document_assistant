"""
Vector Store Management Module
Handles FAISS vector store operations with performance tracking
"""
import os
import re
import time
import shutil
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ai.settings import (
    EMBED_MODEL_NAME,
    EMBED_DEVICE,
    EMBED_NORMALIZE,
    VECTORSTORE_DIR
)

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": EMBED_DEVICE},
            encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE}
        )
        self.metrics = {
            "stores_created": 0,
            "stores_loaded": 0,
            "documents_indexed": 0,
            "total_embedding_time": 0.0,
            "total_retrieval_time": 0.0,
            "retrieval_count": 0
        }
        logger.info(f"Initialized VectorStoreManager with {EMBED_MODEL_NAME}")
    
    def _store_path(self, store_name: str) -> Path:
        """Get full path for a vector store"""
        if not isinstance(store_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", store_name):
            raise ValueError("Invalid store name; use 1-80 ASCII letters, digits, hyphens, or underscores")
        root = VECTORSTORE_DIR.resolve()
        path = (root / store_name).resolve()
        if path.parent != root:
            raise ValueError("Store name escapes the index directory")
        return path
    
    def create_vector_store(
        self,
        chunks: List[Document],
        store_name: str,
        overwrite: bool = False
    ) -> str:
        """
        Create a new FAISS vector store
        
        Args:
            chunks: Document chunks to embed
            store_name: Unique name for the store
            overwrite: Whether to overwrite existing store
            
        Returns:
            Path to created vector store
        """
        if not chunks:
            raise ValueError("No document chunks provided")
        
        start_time = time.time()
        path = self._store_path(store_name)
        
        # Handle existing store
        if path.exists():
            if overwrite:
                logger.info(f"Removing existing store: {store_name}")
                shutil.rmtree(path)
            else:
                raise FileExistsError(
                    f"Store '{store_name}' already exists. "
                    "Set overwrite=True to replace."
                )
        
        path.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Creating vector store '{store_name}' with {len(chunks)} chunks")
            
            # Create FAISS index
            db = FAISS.from_documents(chunks, self.embedding_model)
            db.save_local(str(path))
            
            embedding_time = time.time() - start_time
            
            # Update metrics
            self.metrics["stores_created"] += 1
            self.metrics["documents_indexed"] += len(chunks)
            self.metrics["total_embedding_time"] += embedding_time
            
            logger.info(
                f"Created vector store '{store_name}' at {path} "
                f"in {embedding_time:.2f}s "
                f"({len(chunks)/embedding_time:.1f} chunks/s)"
            )
            
            return str(path)
            
        except Exception as e:
            logger.error(f"Error creating vector store '{store_name}': {e}")
            # Cleanup on failure
            if path.exists():
                shutil.rmtree(path)
            raise RuntimeError(f"Could not create vector store: {e}")
    
    def load_vector_store(self, store_name: str) -> FAISS:
        """
        Load a vector store from disk
        
        Args:
            store_name: Name of the store to load
            
        Returns:
            Loaded FAISS vector store
        """
        start_time = time.time()
        path = self._store_path(store_name)
        
        if not path.exists() or not any(path.iterdir()):
            raise FileNotFoundError(f"Vector store '{store_name}' not found at {path}")
        
        try:
            db = FAISS.load_local(
                str(path),
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
            
            load_time = time.time() - start_time
            self.metrics["stores_loaded"] += 1
            
            logger.info(
                f"Loaded vector store '{store_name}' in {load_time:.2f}s"
            )
            
            return db
            
        except Exception as e:
            logger.error(f"Error loading vector store '{store_name}': {e}")
            raise RuntimeError(f"Could not load vector store: {e}")
    
    def add_documents(
        self,
        documents: List[Document],
        store_name: str
    ) -> str:
        """
        Add documents to existing store (or create new)
        
        Args:
            documents: Documents to add
            store_name: Name of the store
            
        Returns:
            Path to the store
        """
        if not documents:
            logger.warning("No documents provided for addition")
            return str(self._store_path(store_name))
        
        start_time = time.time()
        path = self._store_path(store_name)
        
        try:
            if path.exists() and any(path.iterdir()):
                # Load and add to existing store
                db = self.load_vector_store(store_name)
                db.add_documents(documents)
                db.save_local(str(path))
                logger.info(
                    f"Added {len(documents)} documents to existing store '{store_name}'"
                )
            else:
                # Create new store
                path.mkdir(parents=True, exist_ok=True)
                db = FAISS.from_documents(documents, self.embedding_model)
                db.save_local(str(path))
                self.metrics["stores_created"] += 1
                logger.info(
                    f"Created new store '{store_name}' with {len(documents)} documents"
                )
            
            add_time = time.time() - start_time
            self.metrics["documents_indexed"] += len(documents)
            self.metrics["total_embedding_time"] += add_time
            
            logger.info(f"Document addition completed in {add_time:.2f}s")
            
            return str(path)
            
        except Exception as e:
            logger.error(f"Error adding documents to store '{store_name}': {e}")
            raise RuntimeError(f"Could not add documents: {e}")
    
    def similarity_search(
        self,
        store_name: str,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            store_name: Name of the store
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        try:
            db = self.load_vector_store(store_name)
            
            if score_threshold is not None:
                # FAISS raw scores are distances (lower is better).
                # Relevance scores are higher-is-better.
                if not 0.0 <= score_threshold <= 1.0:
                    raise ValueError("score_threshold must be between 0 and 1")
                docs_and_scores = db.similarity_search_with_relevance_scores(query, k=k)
                docs = [
                    doc for doc, relevance in docs_and_scores
                    if relevance >= score_threshold
                ]
            else:
                docs = db.similarity_search(query, k=k)
            
            retrieval_time = time.time() - start_time
            
            # Update metrics
            self.metrics["retrieval_count"] += 1
            self.metrics["total_retrieval_time"] += retrieval_time
            
            logger.info(
                f"Retrieved {len(docs)} documents for query in {retrieval_time:.3f}s"
            )
            
            return docs
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            raise RuntimeError(f"Search failed: {e}")
    
    def delete_vector_store(self, store_name: str) -> None:
        """Delete a vector store"""
        path = self._store_path(store_name)
        
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info(f"Deleted vector store '{store_name}' at {path}")
            except Exception as e:
                logger.error(f"Error deleting vector store '{store_name}': {e}")
                raise RuntimeError(f"Could not delete vector store: {e}")
        else:
            logger.warning(f"Vector store '{store_name}' does not exist")
    
    def list_stores(self) -> List[Dict[str, Any]]:
        """List all available vector stores"""
        stores = []
        
        if not VECTORSTORE_DIR.exists():
            return stores
        
        for store_path in VECTORSTORE_DIR.iterdir():
            if store_path.is_dir() and any(store_path.iterdir()):
                try:
                    stat = store_path.stat()
                    
                    # Calculate total size
                    total_size = sum(
                        f.stat().st_size
                        for f in store_path.rglob('*')
                        if f.is_file()
                    )
                    
                    stores.append({
                        "name": store_path.name,
                        "path": str(store_path),
                        "created": stat.st_ctime,
                        "modified": stat.st_mtime,
                        "size_bytes": total_size
                    })
                except Exception as e:
                    logger.warning(f"Failed to get info for store '{store_path.name}': {e}")
        
        return sorted(stores, key=lambda x: x["modified"], reverse=True)
    
    def get_store_stats(self, store_name: str) -> Dict[str, Any]:
        """Get statistics for a specific vector store"""
        try:
            db = self.load_vector_store(store_name)
            path = self._store_path(store_name)
            
            # Calculate directory size
            total_size = sum(
                f.stat().st_size
                for f in path.rglob('*')
                if f.is_file()
            )
            
            file_count = len(list(path.rglob('*')))
            
            return {
                "name": store_name,
                "path": str(path),
                "vector_count": db.index.ntotal if hasattr(db.index, 'ntotal') else None,
                "file_count": file_count,
                "total_size_bytes": total_size,
                "embedding_dimension": len(self.embedding_model.embed_query("test"))
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for store '{store_name}': {e}")
            return {"error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_embedding_time = (
            self.metrics["total_embedding_time"] / 
            self.metrics["stores_created"]
            if self.metrics["stores_created"] > 0 else 0
        )
        
        avg_retrieval_time = (
            self.metrics["total_retrieval_time"] / 
            self.metrics["retrieval_count"]
            if self.metrics["retrieval_count"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "avg_embedding_time_per_store": avg_embedding_time,
            "avg_retrieval_time": avg_retrieval_time
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "stores_created": 0,
            "stores_loaded": 0,
            "documents_indexed": 0,
            "total_embedding_time": 0.0,
            "total_retrieval_time": 0.0,
            "retrieval_count": 0
        }


# Global instance
_vector_store_manager = None


def get_vector_store_manager() -> VectorStoreManager:
    """Get or create the global VectorStoreManager instance"""
    global _vector_store_manager
    if _vector_store_manager is None:
        _vector_store_manager = VectorStoreManager()
    return _vector_store_manager


# Convenience functions
def create_vector_store(chunks: List[Document], store_name: str) -> str:
    """Create a vector store (convenience function)"""
    manager = get_vector_store_manager()
    return manager.create_vector_store(chunks, store_name)


def load_vector_store(store_name: str) -> FAISS:
    """Load a vector store (convenience function)"""
    manager = get_vector_store_manager()
    return manager.load_vector_store(store_name)


@lru_cache(maxsize=128)
def get_retriever(store_name: str, k: int = 4, search_type: str = "similarity"):
    """
    Get cached retriever for a vector store
    
    Args:
        store_name: Name of the store
        k: Number of documents to retrieve
        search_type: 'similarity' or 'mmr'
        
    Returns:
        Configured retriever
    """
    db = load_vector_store(store_name)
    
    if search_type == "mmr":
        return db.as_retriever(search_type="mmr", search_kwargs={"k": k})
    
    return db.as_retriever(search_kwargs={"k": k})


def clear_retriever_cache():
    """Clear the retriever cache"""
    get_retriever.cache_clear()
    logger.info("Retriever cache cleared")