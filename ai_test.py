import os
from pathlib import Path

# ---- Paths aligned with ai/settings.py defaults ----
UPLOADS_DIR = Path("backend/app/data/uploads")
STORE_NAME = "sample-store"
SAMPLE_PATH = UPLOADS_DIR / "sample.txt"

def ensure_sample_file():
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.write_text(
            "Project Atlas Technical Notes\n"
            "The analytics stack uses PostgreSQL as the primary database.\n"
            "Dashboards are built in Apache Superset and refreshed nightly.\n",
            encoding="utf-8",
        )
    return str(SAMPLE_PATH)

def main():
    from ai.services.document import load_document, split_documents
    from ai.services.vector_store import create_vector_store
    from ai.services.rag_pipeline import answer

    path = ensure_sample_file()
    print(f"[1/4] Sample file: {path}")

    docs = load_document(path)
    print(f"[2/4] Loaded {len(docs)} document(s)")

    chunks = split_documents(docs)
    print(f"[3/4] Split into {len(chunks)} chunk(s)")

    store_path = create_vector_store(chunks, STORE_NAME)
    print(f"[4/4] Vector store created at: {store_path}")

    print("\n--- RAG Query ---")
    q = "Which database is used by the analytics stack?"
    try:
        out = answer(q, STORE_NAME, k=4, include_sources=True)
        print("Q:", q)
        print("A:", out["answer"])
        print("\nSources:")
        for s in out.get("sources", []):
            print("-", s["source"], "page=", s.get("page"), "\n  ", s["snippet"][:120], "...")
    except Exception as e:
        print("\n[ERROR] Could not run the LLM step:", e)
        print("Tips:")
        print(" - Ensure Ollama is running and OLLAMA_HOST is reachable")
        print(" - Pull a model, e.g.:  ollama pull llama3")
        print(" - Or set OLLAMA_MODEL to one you already have")

if __name__ == "__main__":
    main()