"""
FastAPI Main Application
Simple RAG Pipeline API for testing and demonstration
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from backend.app.api.endpoints import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation Pipeline for Document Q&A",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("=" * 60)
    logger.info("RAG Pipeline API Starting Up")
    logger.info("=" * 60)
    logger.info("Documentation available at: /docs")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("RAG Pipeline API Shutting Down")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )