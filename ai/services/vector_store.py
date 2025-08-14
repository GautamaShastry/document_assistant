import os
from functools import lru_cache
from typing import List, Optional

from langchain_core.documents import Document 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from ai.settings import EMBED_DEVICE, EMBED_MODEL_NAME, EMBED_NORMALIZE, VECTORSTORE_DIR

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={"device": EMBED_DEVICE},
    encode_kwargs={"normalize_embeddings": EMBED_NORMALIZE},
)

def _store_path(store_name: str) -> str:
    if not isinstance(store_name, (str, os.PathLike)):
        raise TypeError(f"store_name must be str/path, got {type(store_name).__name__}")
    return os.path.join(VECTORSTORE_DIR, str(store_name))

def create_vector_store(chunks: List[Document], store_name: str) -> FAISS:
    """
    Create a new FAISS vector store from document chunks.
    """
    path = _store_path(store_name)
    os.makedirs(path, exist_ok=True)
    
    for f in os.listdir(path):
        try:
            os.remove(os.path.join(path, f))
        except IsADirectoryError:
            pass
        
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(path)
    return path

def upsert_documents_to_vector_store(documents: List[Document], store_name: str) -> str:
    """
    Append new chunks to an existing store (creates if missing).
    """
    path = _store_path(store_name)
    os.makedirs(path, exist_ok=True)
    if os.listdir(path):
        db = FAISS.load_local(path, embedding_model, allow_dangerous_serialization=True)
        db.add_documents(documents)
        db.save_local(path)
    else:
        db = FAISS.from_documents(documents, embedding_model)
        db.save_local(path)
    return path

def load_vector_store(store_name: str) -> FAISS:
    """
    Load a vector store from the specified path.
    """
    path = _store_path(store_name)
    if not os.path.isdir(path) or not os.listdir(path):
        raise FileNotFoundError(f"Vector store '{store_name}' not found at {path}")
    return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

@lru_cache(maxsize=128)
def get_retriever(store_name: str, k: int = 4, search_type: str = "similarity"):
    """
    Cached retriever for speed. search_type: 'similarity' | 'mmr'
    """
    db = load_vector_store(store_name)
    kwargs = {"k": k}
    
    if search_type == "mmr":
        return db.as_retriever(search_type="mmr", search_kwargs=kwargs)
    return db.as_retriever(search_kwargs=kwargs)

def delete_vector_store(store_name: str) -> None:
    """
    Delete a vector store from the specified path.
    """
    import shutil
    path = _store_path(store_name)
    if os.path.isdir(path):
        shutil.rmtree(path)