import os, re
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.utils.file import save_upload_file
from backend.app.utils.registry import register_store, resolve_store

from ai.services.document import load_document, split_documents
from ai.services.vector_store import create_vector_store, load_vector_store
from ai.services.rag_pipeline import answer, stream_answer

router = APIRouter(tags=["core"])

@router.get("/health")
def health():
    return {
        "status": "ok",
        "message": "RAG API is healthy",
        "vectorstores_dir": os.path.abspath("backend/app/data/vectorstore"),
        "ollama_model": os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"),
    }
    
@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    label: str = Form("default"),
    chunk_size: int = Form(int(os.getenv("CHUNK_SIZE", 900))),
    chunk_overlap: int = Form(int(os.getenv("CHUNK_OVERLAP", 100))),
):
    """Upload one file, index it into a new FAISS store, return an opaque index_id."""
    path = await save_upload_file(file, dest_dir="backend/app/data/uploads")
    
    try:
        docs = load_document(path)
        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load document: {e}")
    
    # store name won't be exposed
    base = os.path.splitext(os.path.basename(path))[0]
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{label}_{base}")
    try:
        store_path = create_vector_store(chunks, safe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not create vector store: {e}")
    
    index_id = register_store(safe, meta={"label": label, "filename": file.filename})
    return {"status": "indexed", "index_id": index_id, "documents": len(docs), "chunks": len(chunks), "store_path": store_path}

class QueryRequest(BaseModel):
    query: str
    index_id: str
    k: int = 4
    include_sources: bool = True

@router.post("/query")
def query(req: QueryRequest):
    """Query a given index_id, return answer and sources."""
    store_name = resolve_store(req.index_id)
    if store_name is None:
        raise HTTPException(status_code=404, detail=f"Unknown index_id: {req.index_id}")
    try:
        _ = load_vector_store(store_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load vector store: {e}")
    try:
        out = answer(req.query, store_name, k=req.k, include_sources=req.include_sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not process query: {e}")
    return out

@router.post("/stream_query")
def stream_query(index_id: str, q: str, k: int = 4):
    store_name = resolve_store(index_id)
    if not store_name:
        raise HTTPException(status_code=404, detail=f"Unknown index_id: {index_id}")
    try:
        _ = load_vector_store(store_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load vector store: {e}")
    
    def token():
        try:
            for chunk in stream_answer(q, store_name, k=k):
                yield chunk
        except Exception as e:
            yield f"\n\n[ERROR] Could not process query: {e}\n"
    return StreamingResponse(token(), media_type="text/plain")