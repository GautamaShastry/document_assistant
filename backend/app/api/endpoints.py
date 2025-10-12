"""
FastAPI Endpoints for RAG Pipeline
Simple endpoints for testing and demonstration
"""
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ai.services.document import DocumentProcessor
from ai.services.vector_store import get_vector_store_manager
from ai.services.rag_pipeline import get_rag_pipeline

router = APIRouter(tags=["RAG Pipeline"])

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask")
    store_name: str = Field(..., description="Vector store to query")
    k: int = Field(4, ge=1, le=20, description="Number of documents to retrieve")
    include_sources: bool = Field(True, description="Include source information")


class QueryResponse(BaseModel):
    answer: str
    sources: list
    metadata: dict

class StreamQueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask")
    store_name: str = Field(..., description="Vector store to query")
    k: int = Field(4, ge=1, le=20, description="Number of documents to retrieve")
    
class StreamingResponse(BaseModel):
    answer: str

class HealthResponse(BaseModel):
    status: str
    message: str
    ollama_model: str
    ollama_host: str
    vectorstore_dir: str
    embed_model: str


class UploadResponse(BaseModel):
    status: str
    store_name: str
    filename: str
    chunks_created: int
    processing_time: float
    message: str


class MetricsResponse(BaseModel):
    document_processor: dict
    vector_store: dict
    rag_pipeline: dict


# Initialize managers
doc_processor = DocumentProcessor()
vector_manager = get_vector_store_manager()
rag_pipeline = get_rag_pipeline()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from ai.settings import OLLAMA_MODEL, OLLAMA_HOST, VECTORSTORE_DIR, EMBED_MODEL_NAME
    
    return HealthResponse(
        status="healthy",
        message="RAG Pipeline is operational",
        ollama_model=OLLAMA_MODEL,
        ollama_host=OLLAMA_HOST,
        vectorstore_dir=str(VECTORSTORE_DIR),
        embed_model=EMBED_MODEL_NAME
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(..., description="Document file (PDF, DOCX, TXT)"),
    store_name: Optional[str] = Form(None, description="Custom store name"),
    chunk_size: int = Form(900, ge=100, le=2000),
    chunk_overlap: int = Form(100, ge=0, le=500)
):
    """
    Upload and process a document
    
    - Validates and loads the document
    - Splits into chunks
    - Creates vector embeddings
    - Stores in FAISS index
    """
    import time
    
    # Save uploaded file
    upload_dir = Path("backend/app/data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = upload_dir / f"{uuid.uuid4().hex}_{file.filename}"
    
    try:
        # Save file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        start_time = time.time()
        
        # Process document
        processor = DocumentProcessor(chunk_size, chunk_overlap)
        chunks = processor.process_file(str(file_path))
        
        # Generate store name if not provided
        if not store_name:
            base_name = Path(file.filename).stem
            # Clean the filename for use as store name
            clean_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)
            store_name = f"{clean_name}_{uuid.uuid4().hex[:8]}"
        
        # Create vector store
        vector_manager.create_vector_store(chunks, store_name)
        
        processing_time = time.time() - start_time
        
        return UploadResponse(
            status="success",
            store_name=store_name,
            filename=file.filename,
            chunks_created=len(chunks),
            processing_time=round(processing_time, 2),
            message=f"Document processed successfully. Use store_name '{store_name}' to query."
        )
        
    except Exception as e:
        # Clean up file on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query the RAG pipeline
    
    - Retrieves relevant document chunks
    - Generates answer using LLM
    - Returns answer with sources and metadata
    """
    try:
        result = rag_pipeline.query(
            question=request.query,
            store_name=request.store_name,
            k=request.k,
            include_sources=request.include_sources
        )
        
        return QueryResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            metadata=result.get("metadata", {})
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store '{request.store_name}' not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/stream-query")
async def stream_query(request: StreamQueryRequest):
    """
    Stream query response for real-time experience
    
    - Returns response as a stream
    - Useful for chat-like interfaces
    
    Example request body:
    ```json
    {
        "query": "What is the main topic?",
        "store_name": "my_document",
        "k": 4
    }
    ```
    """
    try:
        def generate():
            try:
                for chunk in rag_pipeline.stream_query(
                    request.query, 
                    request.store_name, 
                    request.k
                ):
                    yield chunk
            except Exception as e:
                yield f"\n[ERROR] {str(e)}"
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "X-Content-Type-Options": "nosniff"
            }
        )
        
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Vector store '{request.store_name}' not found"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming failed: {str(e)}")


@router.get("/stores")
async def list_stores():
    """List all available vector stores"""
    try:
        stores = vector_manager.list_stores()
        return {
            "count": len(stores),
            "stores": stores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list stores: {str(e)}")


@router.get("/stores/{store_name}/stats")
async def get_store_stats(store_name: str):
    """Get statistics for a specific vector store"""
    try:
        stats = vector_manager.get_store_stats(store_name)
        if "error" in stats:
            raise HTTPException(status_code=404, detail=stats["error"])
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.delete("/stores/{store_name}")
async def delete_store(store_name: str):
    """Delete a vector store"""
    try:
        vector_manager.delete_vector_store(store_name)
        return {"message": f"Vector store '{store_name}' deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete store: {str(e)}")


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get comprehensive performance metrics
    
    Returns metrics for:
    - Document processing
    - Vector store operations
    - RAG pipeline performance
    """
    return MetricsResponse(
        document_processor=doc_processor.get_metrics(),
        vector_store=vector_manager.get_metrics(),
        rag_pipeline=rag_pipeline.get_metrics()
    )


@router.post("/metrics/reset")
async def reset_metrics():
    """Reset all metrics counters"""
    doc_processor.reset_metrics()
    vector_manager.reset_metrics()
    rag_pipeline.reset_metrics()
    return {"message": "All metrics have been reset"}