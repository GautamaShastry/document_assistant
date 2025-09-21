import os
import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document 
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ai.settings import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSING_LIMITS

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = ['.pdf', '.docx', '.txt']

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors."""
    pass

def validate_file(file_path: str) -> Dict[str, Any]:
    """
    Validate file before processing.
    
    Args:
        file_path: Path to the file to validate
        
    Returns:
        Dictionary with validation results
        
    Raises:
        FileNotFoundError: If file doesn't exist
        DocumentProcessingError: If file is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_size = os.path.getsize(file_path)
    if file_size > PROCESSING_LIMITS["max_file_size"]:
        raise DocumentProcessingError(f"File size {file_size} exceeds maximum allowed size of {PROCESSING_LIMITS['max_file_size']}")
    
    extension = os.path.splitext(file_path)[-1].lower()
    if extension not in SUPPORTED_TYPES:
        raise DocumentProcessingError(f"Unsupported file type: {extension}")
    
    # check if file is readable
    try:
        with open(file_path, 'rb') as f:
            f.read(1024)  # try reading first 1KB
    except Exception as e:
        raise DocumentProcessingError(f"File is not readable: {e}")
    
    return {
        "file_path": file_path,
        "file_size": file_size,
        "file_extension": extension,
        "is_valid": True,
    }

def load_document(file_path: str, add_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """
    Load a document into LangChain Document objects.
    Adds normalized metadata for consistent citations.
    
    Args:
        file_path: Path to the document file
        add_metadata: Additional metadata to include
        
    Returns:
        List of Document objects
        
    Raises:
        FileNotFoundError: If file doesn't exist
        DocumentProcessingError: If document processing fails
    """
    try:
        validation = validate_file(file_path)
        extension = validation["file_extension"]
        
        logger.info(f"Loading document: {file_path} (type: {extension}, size: {validation['file_size']})")
        
        if extension == '.pdf':
            loader = PyPDFLoader(file_path)
        elif extension == '.docx':
            loader = Docx2txtLoader(file_path)
        elif extension == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise DocumentProcessingError(f"Unsupported file type: {extension}")
        
        # load documents
        docs = loader.load()
        
        if not docs:
            raise DocumentProcessingError("No content found in document.")
        
        # add normalized metadata
        for doc in docs:
            base_metadata = {
                "source": file_path,
                "file_name": os.path.basename(file_path),
                "file_size": validation["file_size"],
                "file_type": extension,
                "loader_type": loader.__class__.__name__,
            }
            
            # merge existing metadata
            doc.metadata = {
                **base_metadata,
                **(doc.metadata or {}),
                **(add_metadata or {}),
            }
            
            if doc.page_content:
                doc.page_content = " ".join(doc.page_content.split())  # normalize whitespace
                
            docs = [doc for doc in docs if doc.page_content.strip()]  # filter empty
            
            if not docs:
                raise DocumentProcessingError("All document chunks are empty after processing.")
            
        logger.info(f"Loaded {len(docs)} document chunks from {file_path}")
        return docs
    except (FileNotFoundError, DocumentProcessingError):
        raise
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {e}")
        raise DocumentProcessingError(f"Error loading document: {e}")

def split_documents(docs: List[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, add_chunk_metadata: bool = True) -> List[Document]:
    """
    Split a list of documents into smaller chunks.
    
    Args:
        docs: List of documents to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        add_chunk_metadata: Whether to add chunk-specific metadata
        
    Returns:
        List of document chunks
        
    Raises:
        DocumentProcessingError: If chunking fails
    """
    if not docs:
        logger.warning("No documents provided for splitting.")
        return []
    
    try:
        logger.info(f"Splitting {len(docs)} documents into chunks (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})")
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""],
            keep_separator=True,
        )
            
        chunks = splitter.split_documents(docs)
        
        # check chunk limit before processing
        max_chunks = PROCESSING_LIMITS["max_chunks_per_document"]
        if len(chunks) > max_chunks:
            logger.warning(f"Document chunk count {len(chunks)} exceeds limit of {max_chunks}. Truncating.")
            chunks = chunks[:max_chunks]
        
        # filter out very small chunks
        min_chunk_size = max(50, chunk_size // 10)
        original_count = len(chunks)
        chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= min_chunk_size]
        
        if len(chunks) < original_count:
            logger.info(f"Filtered out {original_count - len(chunks)} small chunks")
        
        # add chunk specific metadata
        if add_chunk_metadata:
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_index": i,
                    "chunk_size": len(chunk.page_content),
                    "total_chunks": len(chunks),
                })
        
        if not chunks:
            raise DocumentProcessingError("No valid chunks created after splitting and filtering.")
        
        logger.info(f"Successfully created {len(chunks)} chunks")
        return chunks
        
    except DocumentProcessingError:
        raise
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise DocumentProcessingError(f"Error splitting documents: {e}")