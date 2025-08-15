import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from backend.app.api import router as api_router

# disable tensorflow globally
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

load_dotenv()

# ensure the data dirs exist
os.makedirs("backend/app/data/uploads", exist_ok=True)
os.makedirs("backend/app/data/vectorstore", exist_ok=True)

app = FastAPI(title="Intelligent Document Assistant", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "Welcome to the Intelligent Document Assistant API. Visit /docs for API documentation."}