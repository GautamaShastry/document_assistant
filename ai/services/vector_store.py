import os
import logging
from functools import lru_cache
from typing import List, Optional, Dict, Any, Generator

from langchain_core.documents import Document 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ai.settings import EMBED_DEVICE, EMBED_MODEL_NAME, EMBED_NORMALIZE, VECTORSTORE_DIR

logger = logging.getLogger(__name__)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": EMBED_DEVICE},
    encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE},
)

def _store_path(store_name: str) -> str:
    """Get the full path for a vector store."""
    if not isinstance(store_name, (str, os.PathLike)):
        raise TypeError(f"store_name must be str/path, got {type(store_name).__name__}")
    return os.path.join(VECTORSTORE_DIR, str(store_name))

def create_vector_store(chunks: List[Document], store_name: str) -> str:
    """
    Create a new FAISS vector store from document chunks.
    
    Args:
        chunks: List of document chunks to embed
        store_name: Unique name for the vector store
        
    Returns:
        Path to the created vector store
        
    Raises:
        ValueError: If chunks list is empty
        RuntimeError: If vector store creation fails
    """
    if not chunks:
        raise ValueError("No document chunks provided for vector store creation.")
    
    path = _store_path(store_name)
    os.makedirs(path, exist_ok=True)
    
    try:
        for f in os.listdir(path):
            file_path = os.path.join(path, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.warning(f"Could not remove existing file/dir {file_path}: {e}")
                
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(path)
        
        logger.info(f"Created vector store '{store_name}' at {path} with {len(chunks)} chunks.")
        return path
    except Exception as e:
        logger.error(f"Error creating vector store '{store_name}': {e}")
        raise RuntimeError(f"Could not create vector store: {e}")

def upsert_documents_to_vector_store(documents: List[Document], store_name: str) -> str:
    """
    Append new chunks to an existing store (creates if missing).
    
    Args:
        documents: List of documents to add
        store_name: Name of the vector store
        
    Returns:
        Path to the vector store
    """
    if not documents:
        logger.warning("No documents provided for upsert; skipping.")
        return _store_path(store_name)
    path = _store_path(store_name)
    os.makedirs(path, exist_ok=True)
    
    try:
        if os.listdir(path):
            try:
                db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
                db.add_documents(documents)
                logger.info(f"Appended {len(documents)} documents to existing vector store '{store_name}'.")
            except Exception as e:
                logger.warning(f"Could not load existing vector store '{store_name}': {e}. Creating new store.")
                db = FAISS.from_documents(documents, embedding_model)
        else:
            db = FAISS.from_documents(documents, embedding_model)
            logger.info(f"Created new vector store '{store_name}' with {len(documents)} documents.")
            
        db.save_local(path)
        return path
    except Exception as e:
        logger.error(f"Error upserting documents to vector store '{store_name}': {e}")
        raise RuntimeError(f"Could not upsert documents: {e}")

def load_vector_store(store_name: str) -> FAISS:
    """
    Load a vector store from the specified path.
    
    Args:
        store_name: Name of the vector store
        
    Returns:
        Loaded FAISS vector store
        
    Raises:
        FileNotFoundError: If vector store doesn't exist
        RuntimeError: If loading fails
    """
    path = _store_path(store_name)
    if not os.path.isdir(path) or not os.listdir(path):
        raise FileNotFoundError(f"Vector store '{store_name}' not found at {path}")
    try:
        db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"Loaded vector store '{store_name}' from {path}.")
        return db
    except Exception as e:
        logger.error(f"Error loading vector store '{store_name}': {e}")
        raise RuntimeError(f"Could not load vector store: {e}")

@lru_cache(maxsize=128)
def get_retriever(store_name: str, k: int = 4, search_type: str = "similarity", **kwargs):
    """
    Cached retriever for speed. 
    
    Args:
        store_name: Name of the vector store
        k: Number of documents to retrieve
        search_type: 'similarity' | 'mmr'
        **kwargs: Additional search parameters
        
    Returns:
        Configured retriever object
    """
    try:
        db = load_vector_store(store_name)
        kwargs = {"k": k, **kwargs}
        
        if search_type == "mmr":
            return db.as_retriever(search_type="mmr", search_kwargs=kwargs)
        return db.as_retriever(search_kwargs=kwargs)
    except Exception as e:
        logger.error(f"Error getting retriever for store '{store_name}': {e}")
        raise RuntimeError(f"Could not get retriever: {e}")

def delete_vector_store(store_name: str) -> None:
    """
    Delete a vector store from the specified path.
    
    Args:
        store_name: Name of the vector store to delete
    """
    import shutil
    path = _store_path(store_name)
    if os.path.isdir(path):
        try:
            shutil.rmtree(path)
            logger.info(f"Deleted vector store '{store_name}' at {path}.")
        except Exception as e:
            logger.error(f"Error deleting vector store '{store_name}': {e}")
            raise RuntimeError(f"Could not delete vector store: {e}")
    else:
        logger.warning(f"Vector store '{store_name}' at {path} does not exist; nothing to delete.")
        
def list_vector_stores() -> List[Dict[str, Any]]:
    """
    List all available vector stores.
    
    Returns:
        List of dictionaries with store information
    """
    stores = []
    
    if not os.path.exists(VECTORSTORE_DIR):
        return stores
    
    try:
        for store_name in os.listdir(VECTORSTORE_DIR):
            store_path = os.path.join(VECTORSTORE_DIR, store_name)
            
            if os.path.isdir(store_path) and os.listdir(store_path):
                try:
                    # Get basic info
                    stat = os.stat(store_path)
                    stores.append({
                        "name": store_name,
                        "path": store_path,
                        "created": stat.st_ctime,
                        "modified": stat.st_mtime,
                        "size_bytes": sum(
                            os.path.getsize(os.path.join(store_path, f))
                            for f in os.listdir(store_path)
                            if os.path.isfile(os.path.join(store_path, f))
                        )
                    })
                except Exception as e:
                    logger.warning(f"Failed to get info for store '{store_name}': {e}")
                    
    except Exception as e:
        logger.error(f"Failed to list vector stores: {e}")
    
    return stores

def get_store_stats(store_name: str) -> Dict[str, Any]:
    """
    Get statistics for a specific vector store.
    
    Args:
        store_name: Name of the vector store
        
    Returns:
        Dictionary with store statistics
    """
    try:
        db = load_vector_store(store_name)
        path = _store_path(store_name)
        
        # Calculate directory size
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                total_size += os.path.getsize(file_path)
                file_count += 1
        
        return {
            "name": store_name,
            "path": path,
            "vector_count": db.index.ntotal if hasattr(db.index, 'ntotal') else None,
            "file_count": file_count,
            "total_size_bytes": total_size,
            "embedding_dimension": len(embedding_model.embed_query("test")) if embedding_model else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get stats for store '{store_name}': {e}")
        return {"error": str(e)}

# Clear cache when needed
def clear_retriever_cache():
    """Clear the retriever cache"""
    get_retriever.cache_clear()
    logger.info("Retriever cache cleared")