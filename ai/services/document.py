"""
Document Processing Module
Handles document loading, validation, and text chunking
"""
import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ai.settings import CHUNK_SIZE, CHUNK_OVERLAP, PROCESSING_LIMITS, SUPPORTED_TYPES

logger = logging.getLogger(__name__)


class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass


class DocumentProcessor:
    """Main class for document processing operations"""
    
    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metrics = {
            "documents_processed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
    
    def validate_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate file before processing
        
        Args:
            file_path: Path to file
            
        Returns:
            Dictionary with validation results
            
        Raises:
            FileNotFoundError: If file doesn't exist
            DocumentProcessingError: If file is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size > PROCESSING_LIMITS["max_file_size"]:
            raise DocumentProcessingError(
                f"File size {file_size} bytes exceeds maximum "
                f"{PROCESSING_LIMITS['max_file_size']} bytes"
            )
        
        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_TYPES:
            raise DocumentProcessingError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(SUPPORTED_TYPES)}"
            )
        
        # Test file readability
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)
        except Exception as e:
            raise DocumentProcessingError(f"File is not readable: {e}")
        
        return {
            "file_path": str(file_path),
            "file_size": file_size,
            "file_extension": extension,
            "is_valid": True
        }
    
    def load_document(
        self,
        file_path: str,
        add_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Load document into LangChain Document objects
        
        Args:
            file_path: Path to document
            add_metadata: Additional metadata to include
            
        Returns:
            List of Document objects
        """
        start_time = time.time()
        
        try:
            validation = self.validate_file(file_path)
            extension = validation["file_extension"]
            
            logger.info(
                f"Loading document: {Path(file_path).name} "
                f"(type: {extension}, size: {validation['file_size']} bytes)"
            )
            
            # Select appropriate loader
            if extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif extension == '.docx':
                loader = Docx2txtLoader(file_path)
            elif extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                raise DocumentProcessingError(f"Unsupported file type: {extension}")
            
            # Load documents
            docs = loader.load()
            
            if not docs:
                raise DocumentProcessingError("No content found in document")
            
            # Add metadata to each document
            for doc in docs:
                base_metadata = {
                    "source": str(file_path),
                    "file_name": Path(file_path).name,
                    "file_size": validation["file_size"],
                    "file_type": extension,
                    "loader_type": loader.__class__.__name__
                }
                
                # Merge metadata
                doc.metadata = {
                    **base_metadata,
                    **(doc.metadata or {}),
                    **(add_metadata or {})
                }
                
                # Normalize whitespace
                if doc.page_content:
                    doc.page_content = " ".join(doc.page_content.split())
            
            # Filter empty documents
            docs = [doc for doc in docs if doc.page_content.strip()]
            
            if not docs:
                raise DocumentProcessingError(
                    "All document chunks are empty after processing"
                )
            
            processing_time = time.time() - start_time
            self.metrics["documents_processed"] += 1
            self.metrics["total_processing_time"] += processing_time
            
            logger.info(
                f"Loaded {len(docs)} document chunks from {Path(file_path).name} "
                f"in {processing_time:.2f}s"
            )
            
            return docs
            
        except (FileNotFoundError, DocumentProcessingError):
            self.metrics["errors"] += 1
            raise
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"Error loading document {file_path}: {e}")
            raise DocumentProcessingError(f"Error loading document: {e}")
    
    def split_documents(
        self,
        docs: List[Document],
        add_chunk_metadata: bool = True
    ) -> List[Document]:
        """
        Split documents into smaller chunks
        
        Args:
            docs: List of documents to split
            add_chunk_metadata: Whether to add chunk-specific metadata
            
        Returns:
            List of document chunks
        """
        if not docs:
            logger.warning("No documents provided for splitting")
            return []
        
        start_time = time.time()
        
        try:
            logger.info(
                f"Splitting {len(docs)} documents "
                f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
            )
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""],
                keep_separator=True,
                length_function=len
            )
            
            chunks = splitter.split_documents(docs)
            
            # Apply chunk limit
            max_chunks = PROCESSING_LIMITS["max_chunks_per_document"]
            if len(chunks) > max_chunks:
                logger.warning(
                    f"Chunk count {len(chunks)} exceeds limit {max_chunks}. Truncating."
                )
                chunks = chunks[:max_chunks]
            
            # Filter small chunks
            min_chunk_size = PROCESSING_LIMITS["min_chunk_size"]
            original_count = len(chunks)
            chunks = [
                chunk for chunk in chunks 
                if len(chunk.page_content.strip()) >= min_chunk_size
            ]
            
            if len(chunks) < original_count:
                logger.info(f"Filtered out {original_count - len(chunks)} small chunks")
            
            # Add chunk metadata
            if add_chunk_metadata:
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_index": i,
                        "chunk_size": len(chunk.page_content),
                        "total_chunks": len(chunks)
                    })
            
            if not chunks:
                raise DocumentProcessingError(
                    "No valid chunks created after splitting and filtering"
                )
            
            processing_time = time.time() - start_time
            self.metrics["total_chunks_created"] += len(chunks)
            
            logger.info(
                f"Created {len(chunks)} chunks in {processing_time:.2f}s "
                f"(avg: {len(chunks)/len(docs):.1f} chunks/doc)"
            )
            
            return chunks
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise DocumentProcessingError(f"Error splitting documents: {e}")
    
    def process_file(
        self,
        file_path: str,
        add_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Complete pipeline: load and split a document
        
        Args:
            file_path: Path to document
            add_metadata: Additional metadata
            
        Returns:
            List of document chunks ready for embedding
        """
        docs = self.load_document(file_path, add_metadata)
        chunks = self.split_documents(docs)
        return chunks
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get processing metrics"""
        avg_time = (
            self.metrics["total_processing_time"] / 
            self.metrics["documents_processed"]
            if self.metrics["documents_processed"] > 0 else 0
        )
        
        return {
            **self.metrics,
            "avg_processing_time": avg_time
        }
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics = {
            "documents_processed": 0,
            "total_chunks_created": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }


# Convenience functions for backward compatibility
def load_document(file_path: str, add_metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
    """Load a document (convenience function)"""
    processor = DocumentProcessor()
    return processor.load_document(file_path, add_metadata)


def split_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    add_chunk_metadata: bool = True
) -> List[Document]:
    """Split documents (convenience function)"""
    processor = DocumentProcessor(chunk_size, chunk_overlap)
    return processor.split_documents(docs, add_chunk_metadata)